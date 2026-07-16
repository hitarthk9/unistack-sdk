"""
OpenTelemetry tracing for UniStack — instance-scoped, vendor-neutral, fail-open.

`Telemetry` owns a TracerProvider that is either passed in by the caller or built here
from an explicit OTLP endpoint; it is NEVER installed globally (`trace.set_tracer_provider`
is never called) and nothing is written to os.environ. Point the endpoint at any OTLP/HTTP
backend — a collector, a hyperscaler agent, or Langfuse (`.../api/public/otel`).

`OTelCallbackHandler` maps LangChain/LangGraph callback events (graph run, nodes, LLM
calls, tools) to OTel spans with GenAI semantic-convention attributes. It parents spans
through an explicit run_id → span map — never thread-local context — so parallel node
fan-out on executor threads keeps correct lineage.

Every entry point here is best-effort: a telemetry failure logs one warning and never
raises into the run.
"""

import json
import logging
import threading
from contextlib import contextmanager
from datetime import timezone
from urllib.parse import unquote

from langchain_core.callbacks import BaseCallbackHandler
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import (
    NonRecordingSpan,
    SpanContext,
    Status,
    StatusCode,
    TraceFlags,
    format_span_id,
    format_trace_id,
    set_span_in_context,
)

logger = logging.getLogger("unistack")

_CLIP = 4000  # max chars for any input/output attribute value


def _clip(value, limit: int = _CLIP) -> str:
    """Serialize a value for a span attribute, truncated so no attribute is unbounded."""
    try:
        text = value if isinstance(value, str) else json.dumps(value, default=str)
    except Exception:
        text = repr(value)
    return text if len(text) <= limit else text[:limit] + "…[clipped]"


def parse_otlp_headers(raw) -> dict | None:
    """Parse the standard OTLP header format 'k=v,k2=v2' (values URL-decoded)."""
    if raw is None or isinstance(raw, dict):
        return dict(raw) if raw else None
    headers = {}
    for pair in str(raw).split(","):
        key, sep, value = pair.strip().partition("=")
        if sep and key:
            headers[key.strip()] = unquote(value.strip())
    return headers or None


def _normalize_endpoint(endpoint: str) -> str:
    """The OTLP exporter appends /v1/traces only for env-supplied endpoints — do it here."""
    endpoint = endpoint.rstrip("/")
    return endpoint if endpoint.endswith("/v1/traces") else endpoint + "/v1/traces"


def _remote_context(trace_id_hex: str, span_id_hex: str):
    parent = SpanContext(
        trace_id=int(trace_id_hex, 16), span_id=int(span_id_hex, 16),
        is_remote=True, trace_flags=TraceFlags(TraceFlags.SAMPLED))
    return set_span_in_context(NonRecordingSpan(parent))


class Telemetry:
    """
    Instance-scoped OTel tracing. Enabled by a `tracer_provider` (caller-owned; also the
    test seam) or an OTLP `endpoint` (provider built and owned here). Disabled → every
    method is a no-op, so callers never branch on `enabled`.
    """

    def __init__(self, workflow: str, *, tracer_provider=None, endpoint: str | None = None,
                 headers=None, service_name: str | None = None):
        self._workflow = workflow
        self._provider = None
        self._owns_provider = False
        self._tracer = None
        self._warned: set[str] = set()
        self.enabled = False
        try:
            if tracer_provider is not None:
                self._provider = tracer_provider
            elif endpoint:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                from opentelemetry.sdk.resources import Resource
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.export import BatchSpanProcessor
                provider = TracerProvider(resource=Resource.create(
                    {"service.name": service_name or f"unistack-{workflow}"}))
                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
                    endpoint=_normalize_endpoint(endpoint),
                    headers=parse_otlp_headers(headers))))
                self._provider = provider
                self._owns_provider = True
            if self._provider is not None:
                self._tracer = self._provider.get_tracer("unistack")
                self.enabled = True
        except Exception as exc:
            logger.warning("telemetry disabled — could not initialise OpenTelemetry: %s", exc)
            self.enabled = False

    # ── span primitives (all fail-open) ─────────────────────────────────────────

    def _warn(self, what: str, exc: Exception) -> None:
        if what not in self._warned:
            self._warned.add(what)
            logger.warning("telemetry: %s failed (suppressing repeats): %s", what, exc)

    def _session_attrs(self, activity_id: str | None) -> dict:
        attrs = {"unistack.workflow": self._workflow}
        if activity_id:
            # session.id is the vendor-neutral key; langfuse.session.id is an additive
            # rendering hint (drives Langfuse's Sessions view, ignored elsewhere).
            attrs.update({"session.id": activity_id, "langfuse.session.id": activity_id,
                          "unistack.activity_id": activity_id})
        return attrs

    @contextmanager
    def _span_cm(self, name: str, attributes: dict | None):
        """Start a span as current; real exceptions from the body still propagate
        (recorded on the span), but telemetry's own failures never do."""
        span = token = None
        try:
            span = self._tracer.start_span(
                name, attributes={k: v for k, v in (attributes or {}).items() if v is not None})
            token = otel_context.attach(set_span_in_context(span))
        except Exception as exc:
            self._warn(f"start span {name}", exc)
        try:
            yield span
        except Exception as exc:
            if span is not None:
                try:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                except Exception:
                    pass
            raise
        finally:
            if token is not None:
                try:
                    otel_context.detach(token)
                except Exception:
                    pass
            if span is not None:
                try:
                    span.end()
                except Exception as exc:
                    self._warn(f"end span {name}", exc)

    @contextmanager
    def _noop_cm(self):
        yield None

    def leg(self, leg: str, activity_id: str, attributes: dict | None = None):
        """Root span for one run leg (`unistack.start` / `unistack.resume`)."""
        if not self.enabled:
            return self._noop_cm()
        return self._span_cm(f"unistack.{leg}",
                             {**self._session_attrs(activity_id), "unistack.leg": leg,
                              **(attributes or {})})

    def span(self, name: str, attributes: dict | None = None, activity_id: str | None = None):
        """Generic child span of whatever is current (or a new root when nothing is)."""
        if not self.enabled:
            return self._noop_cm()
        return self._span_cm(name, {**self._session_attrs(activity_id), **(attributes or {})})

    def llm_span(self, model: str, input_value: str | None = None):
        """A GenAI chat span; the caller stamps usage/response attrs via set_attrs()."""
        if not self.enabled:
            return self._noop_cm()
        return self._span_cm(f"chat {model}", {
            "gen_ai.operation.name": "chat",
            "gen_ai.system": "anthropic",
            "gen_ai.request.model": model,
            "langfuse.observation.type": "generation",
            "input.value": _clip(input_value) if input_value is not None else None,
        })

    def set_attrs(self, span, attributes: dict | None) -> None:
        if span is None or not attributes:
            return
        try:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, value)
        except Exception as exc:
            self._warn("set attributes", exc)

    def stamp_result(self, span, result) -> None:
        """Stamp a leg root with the run outcome (status + pause details)."""
        self.set_attrs(span, {
            "unistack.status": result.status,
            "unistack.pause.node": result.node,
            "unistack.pause.message": _clip(result.message, 1000) if result.message else None,
        })

    def add_event(self, name: str, attributes: dict | None = None) -> None:
        """Add an event to the current span (a leg root, during a run)."""
        if not self.enabled:
            return
        try:
            span = trace.get_current_span()
            if span.get_span_context().is_valid:
                span.add_event(name, {k: v for k, v in (attributes or {}).items()
                                      if v is not None})
        except Exception as exc:
            self._warn("add event", exc)

    def current_ids(self) -> tuple[str, str] | None:
        """(trace_id, span_id) of the current span as hex — persisted at pause time so
        the hitl_pause span can later be emitted into this same trace."""
        if not self.enabled:
            return None
        try:
            ctx = trace.get_current_span().get_span_context()
            if not ctx.is_valid:
                return None
            return format_trace_id(ctx.trace_id), format_span_id(ctx.span_id)
        except Exception as exc:
            self._warn("read current span ids", exc)
            return None

    def emit_closed_span(self, name: str, trace_id_hex: str, span_id_hex: str,
                         start, attributes: dict | None = None) -> None:
        """
        Emit an already-finished span retroactively into an earlier trace: parent is the
        persisted (trace_id, span_id), start_time is explicit, end is now. This is how a
        HITL pause (which can outlive the process) becomes a span whose duration is the
        real human wait — OTLP cannot export open spans, so it is emitted at resolve time.
        """
        if not self.enabled:
            return
        try:
            if start.tzinfo is None:                # Mongo returns naive UTC datetimes
                start = start.replace(tzinfo=timezone.utc)
            span = self._tracer.start_span(
                name, context=_remote_context(trace_id_hex, span_id_hex),
                start_time=int(start.timestamp() * 1e9),
                attributes={k: v for k, v in (attributes or {}).items() if v is not None})
            span.end()
        except Exception as exc:
            self._warn(f"emit {name} span", exc)

    def link_current_to(self, trace_id_hex: str, span_id_hex: str) -> None:
        """Link the current span (a resume leg root) to the leg that paused."""
        if not self.enabled:
            return
        try:
            trace.get_current_span().add_link(SpanContext(
                trace_id=int(trace_id_hex, 16), span_id=int(span_id_hex, 16),
                is_remote=True, trace_flags=TraceFlags(TraceFlags.SAMPLED)))
        except Exception as exc:
            self._warn("link spans", exc)

    def handler(self, activity_id: str) -> "OTelCallbackHandler | None":
        """A per-run LangChain callback handler; captures the current context (the leg
        root) at construction so node spans parent deterministically even when LangGraph
        runs them on executor threads."""
        if not self.enabled:
            return None
        try:
            return OTelCallbackHandler(self._tracer, otel_context.get_current(),
                                       self._session_attrs(activity_id), self._warn)
        except Exception as exc:
            self._warn("create callback handler", exc)
            return None

    def shutdown(self) -> None:
        """Flush + shut down the provider — only when this instance built it. A
        caller-supplied provider is the caller's to close."""
        if self._owns_provider and self._provider is not None:
            try:
                self._provider.shutdown()
            except Exception as exc:
                self._warn("provider shutdown", exc)
            self._provider = None
            self._tracer = None
            self.enabled = False


class OTelCallbackHandler(BaseCallbackHandler):
    """
    LangChain/LangGraph callback events → OTel spans (graph run + nodes as chain spans,
    LLM calls as GenAI chat spans, tools as tool spans). Node-internal `llm.invoke()`
    without an explicit config still lands here via langchain-core's contextvar
    config propagation. Spans that never receive their end event are simply never
    exported. Every hook is fail-open.
    """

    def __init__(self, tracer, root_context, session_attrs: dict, warn):
        self._tracer = tracer
        self._root_context = root_context
        self._session_attrs = dict(session_attrs)
        self._warn = warn
        self._spans: dict = {}                     # run_id -> Span
        self._lock = threading.Lock()

    # ── internals ────────────────────────────────────────────────────────────────

    @staticmethod
    def _name(serialized, kwargs) -> str:
        if kwargs.get("name"):
            return kwargs["name"]
        serialized = serialized or {}
        if serialized.get("name"):
            return serialized["name"]
        ident = serialized.get("id")
        if isinstance(ident, list) and ident:
            return str(ident[-1])
        return "unknown"

    def _start(self, run_id, parent_run_id, name: str, attributes: dict) -> None:
        with self._lock:
            parent = self._spans.get(parent_run_id) if parent_run_id else None
        context = set_span_in_context(parent) if parent is not None else self._root_context
        span = self._tracer.start_span(name, context=context, attributes={
            **self._session_attrs,
            **{k: v for k, v in attributes.items() if v is not None}})
        with self._lock:
            self._spans[run_id] = span

    def _end(self, run_id, attributes: dict | None = None, error=None) -> None:
        with self._lock:
            span = self._spans.pop(run_id, None)
        if span is None:
            return
        for key, value in (attributes or {}).items():
            if value is not None:
                span.set_attribute(key, value)
        if error is not None:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
        span.end()

    @staticmethod
    def _usage(response) -> dict:
        """Token usage: AIMessage.usage_metadata first, llm_output as fallback."""
        try:
            message = getattr(response.generations[0][0], "message", None)
            usage = getattr(message, "usage_metadata", None)
            if usage:
                return {"input": usage.get("input_tokens"), "output": usage.get("output_tokens")}
        except (IndexError, AttributeError, TypeError):
            pass
        raw = ((getattr(response, "llm_output", None) or {}).get("usage")
               or (getattr(response, "llm_output", None) or {}).get("token_usage") or {})
        return {"input": raw.get("input_tokens") or raw.get("prompt_tokens"),
                "output": raw.get("output_tokens") or raw.get("completion_tokens")}

    @staticmethod
    def _response_payload(response):
        text, model = [], None
        try:
            for batch in response.generations:
                for gen in batch:
                    message = getattr(gen, "message", None)
                    text.append(getattr(message, "content", None) or gen.text)
                    meta = getattr(message, "response_metadata", None) or {}
                    model = model or meta.get("model") or meta.get("model_name")
            model = model or (getattr(response, "llm_output", None) or {}).get("model_name")
        except Exception:
            pass
        return text, model

    # ── chains (the graph run + every node) ──────────────────────────────────────

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None,
                       tags=None, metadata=None, **kwargs):
        try:
            self._start(run_id, parent_run_id, self._name(serialized, kwargs), {
                "langchain.run_type": "chain",
                "input.value": _clip(inputs),
            })
        except Exception as exc:
            self._warn("on_chain_start", exc)

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        try:
            self._end(run_id, {"output.value": _clip(outputs)})
        except Exception as exc:
            self._warn("on_chain_end", exc)

    def on_chain_error(self, error, *, run_id, **kwargs):
        try:
            self._end(run_id, error=error)
        except Exception as exc:
            self._warn("on_chain_error", exc)

    # ── LLM calls ────────────────────────────────────────────────────────────────

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None,
                            tags=None, metadata=None, **kwargs):
        try:
            params = kwargs.get("invocation_params") or {}
            model = (params.get("model") or params.get("model_name")
                     or (metadata or {}).get("ls_model_name") or "unknown")
            payload = [[{"role": getattr(m, "type", type(m).__name__), "content": m.content}
                        for m in batch] for batch in messages]
            self._start(run_id, parent_run_id, f"chat {model}", {
                "gen_ai.operation.name": "chat",
                "gen_ai.system": (metadata or {}).get("ls_provider"),
                "gen_ai.request.model": model,
                "langfuse.observation.type": "generation",
                "input.value": _clip(payload),
            })
        except Exception as exc:
            self._warn("on_chat_model_start", exc)

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None,
                     tags=None, metadata=None, **kwargs):
        try:
            params = kwargs.get("invocation_params") or {}
            model = (params.get("model") or params.get("model_name")
                     or (metadata or {}).get("ls_model_name") or "unknown")
            self._start(run_id, parent_run_id, f"chat {model}", {
                "gen_ai.operation.name": "chat",
                "gen_ai.system": (metadata or {}).get("ls_provider"),
                "gen_ai.request.model": model,
                "langfuse.observation.type": "generation",
                "input.value": _clip(prompts),
            })
        except Exception as exc:
            self._warn("on_llm_start", exc)

    def on_llm_end(self, response, *, run_id, **kwargs):
        try:
            usage = self._usage(response)
            text, model = self._response_payload(response)
            self._end(run_id, {
                "output.value": _clip(text if len(text) != 1 else text[0]),
                "gen_ai.response.model": model,
                "gen_ai.usage.input_tokens": usage.get("input"),
                "gen_ai.usage.output_tokens": usage.get("output"),
            })
        except Exception as exc:
            self._warn("on_llm_end", exc)

    def on_llm_error(self, error, *, run_id, **kwargs):
        try:
            self._end(run_id, error=error)
        except Exception as exc:
            self._warn("on_llm_error", exc)

    # ── tools ────────────────────────────────────────────────────────────────────

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None,
                      tags=None, metadata=None, **kwargs):
        try:
            self._start(run_id, parent_run_id, self._name(serialized, kwargs), {
                "langchain.run_type": "tool",
                "input.value": _clip(input_str),
            })
        except Exception as exc:
            self._warn("on_tool_start", exc)

    def on_tool_end(self, output, *, run_id, **kwargs):
        try:
            self._end(run_id, {"output.value": _clip(output)})
        except Exception as exc:
            self._warn("on_tool_end", exc)

    def on_tool_error(self, error, *, run_id, **kwargs):
        try:
            self._end(run_id, error=error)
        except Exception as exc:
            self._warn("on_tool_error", exc)


__all__ = ["Telemetry", "OTelCallbackHandler", "parse_otlp_headers"]
