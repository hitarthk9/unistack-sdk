import time
from contextvars import ContextVar
from datetime import datetime, timezone

from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode, set_span_in_context
from pymongo import MongoClient

from unistack._exporter import MongoDBSpanExporter

_current_activity: ContextVar[str] = ContextVar("unistack_activity", default="unknown")
_current_workflow: ContextVar[str] = ContextVar("unistack_workflow", default="unknown")
# Module-level dicts (not ContextVars) so mutations in one node are visible to the next
# within the same process.
_resume_pending: dict[str, bool] = {}
# Set by run() when a guardrail-breach interrupt is approved. Consumed by the guardrail
# wrapper on the resume pass to skip re-evaluating the same output a second time.
_guardrail_approved: dict[str, bool] = {}

# Sentinel: prevents patching the Anthropic client more than once per process.
_ANTHROPIC_PATCHED = False


def _instrument_anthropic() -> None:
    """
    Monkey-patch anthropic.resources.messages.Messages.create to auto-capture
    token usage (input_tokens, output_tokens, cost_usd, model) on the active
    OTel span, if one exists.

    This is idempotent — safe to call on every UniStack.init().
    If the anthropic package is not installed, this is a no-op.
    """
    global _ANTHROPIC_PATCHED
    if _ANTHROPIC_PATCHED:
        return
    try:
        import anthropic.resources.messages as _amsg
    except ImportError:
        return  # anthropic not installed — cost tracking simply won't run

    _original = _amsg.Messages.create
    _PRICES: dict[str, tuple[float, float]] = {
        "claude-haiku-4-5-20251001": (0.0000008, 0.000004),   # $0.80/$4.00 per MTok
        "claude-haiku-4-5":          (0.0000008, 0.000004),
        "claude-sonnet-4-6":         (0.000003,  0.000015),
        "claude-opus-4-8":           (0.000015,  0.000075),
    }

    def _capturing_create(self_client, *args, **kwargs):
        resp = _original(self_client, *args, **kwargs)
        try:
            span = trace.get_current_span()
            if span is not None and hasattr(resp, "usage") and resp.usage:
                model = kwargs.get("model", "")
                inp, out = _PRICES.get(model, (0.000001, 0.000005))
                cost = resp.usage.input_tokens * inp + resp.usage.output_tokens * out
                span.set_attribute("llm.model",         model)
                span.set_attribute("llm.input_tokens",  resp.usage.input_tokens)
                span.set_attribute("llm.output_tokens", resp.usage.output_tokens)
                span.set_attribute("llm.cost_usd",      round(cost, 6))
        except Exception:
            pass  # never let instrumentation crash a user's LLM call
        return resp

    _capturing_create.__name__ = _original.__name__
    _amsg.Messages.create = _capturing_create
    _ANTHROPIC_PATCHED = True


class RunResult:
    def __init__(self, activity_id: str, state: dict, status: str = "completed"):
        self.activity_id = activity_id
        self.state = state or {}
        self.status = status  # "completed" | "hitl_rejected" | "guardrail_breached" | "failed"


class UniStackCore:
    """
    Framework-agnostic core: MongoDB connection, OTel setup, HITL queue,
    guardrail context loading, and polling. Framework adapters extend this class
    and implement node(), guardrail(), hitl(), and run().
    """

    def __init__(
        self,
        mongo_uri: str,
        workflow: str,
        db_name: str = "unistack",
        hitl_poll_interval: float = 2.0,
        context: str | None = None,
        context_file: str | None = None,
    ):
        self._workflow = workflow
        self._mongo_uri = mongo_uri
        self._db_name = db_name
        self._hitl_poll_interval = hitl_poll_interval
        self._client = MongoClient(mongo_uri)
        self._db = self._client[db_name]
        self._provider = self._setup_otel()
        self._tracer = trace.get_tracer("unistack")
        self._guardrail_context = self._resolve_context(context, context_file)
        _instrument_anthropic()

    @classmethod
    def init(
        cls,
        mongo_uri: str,
        workflow: str,
        db_name: str = "unistack",
        hitl_poll_interval: float = 2.0,
        context: str | None = None,
        context_file: str | None = None,
    ) -> "UniStackCore":
        return cls(mongo_uri, workflow, db_name, hitl_poll_interval, context, context_file)

    def _setup_otel(self) -> TracerProvider:
        exporter = MongoDBSpanExporter(self._mongo_uri, self._db_name)
        processor = BatchSpanProcessor(exporter)
        current = trace.get_tracer_provider()
        if isinstance(current, TracerProvider):
            current.add_span_processor(processor)
            return current
        provider = TracerProvider()
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        return provider

    @staticmethod
    def _resolve_context(context: str | None, context_file: str | None) -> str | None:
        """Resolve guardrail context from inline string or YAML file."""
        if context:
            return context
        if context_file:
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "pyyaml is required for context_file support: pip install pyyaml"
                )
            with open(context_file) as f:
                data = yaml.safe_load(f)
            return data.get("guardrail_context")
        return None

    def _write_guardrail_entry(
        self,
        activity_id: str,
        *,
        node: str,
        policy: str,
        reason: str,
        status: str = "pending",
    ) -> None:
        """Write a guardrail breach event to unistack.guardrails."""
        self._db.guardrails.replace_one(
            {"_id": f"{activity_id}::{node}"},
            {
                "_id": f"{activity_id}::{node}",
                "activity_id": activity_id,
                "workflow": self._workflow,
                "node": node,
                "policy": policy,
                "reason": reason,
                "status": status,
                "detected_at": datetime.now(tz=timezone.utc),
                "resolved_at": None,
                "resolved_by": None,
            },
            upsert=True,
        )

    def _resolve_guardrail_entry(
        self,
        activity_id: str,
        *,
        node: str,
        status: str,
        resolved_by: str | None,
    ) -> None:
        """Update a guardrail breach record with the human decision."""
        self._db.guardrails.update_one(
            {"_id": f"{activity_id}::{node}"},
            {"$set": {
                "status": status,
                "resolved_at": datetime.now(tz=timezone.utc),
                "resolved_by": resolved_by,
            }},
        )

    def _write_hitl_entry(
        self,
        activity_id: str,
        *,
        type_: str,
        message: str,
        data: dict,
        policy: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Upsert a hitl_queue document. Called by both adapters."""
        self._db.hitl_queue.replace_one(
            {"activity_id": activity_id},
            {
                "activity_id": activity_id,
                "workflow": self._workflow,
                "type": type_,
                "status": "pending",
                "created_at": datetime.now(tz=timezone.utc),
                "resolved_at": None,
                "resolved_by": None,
                "resolution_comment": None,
                "message": message,
                "data": data,
                "guardrail_policy": policy,
                "guardrail_reason": reason,
            },
            upsert=True,
        )

    def _poll_for_decision(self, activity_id: str) -> str:
        """Block until hitl_queue entry is resolved. Returns 'approved' or 'rejected'."""
        print(f"\n[UniStack HITL] Activity paused — waiting for human decision.")
        print(f"  Resolve via API (in a separate terminal):")
        print(f'  curl -X POST http://localhost:8000/hitl/{activity_id}/resolve \\')
        print(f'       -H "Content-Type: application/json" \\')
        print(f'       -d \'{{"decision":"approve","resolved_by":"you@company.com"}}\'')
        print(f"\n  Polling every {self._hitl_poll_interval}s ...")
        while True:
            doc = self._db.hitl_queue.find_one(
                {"activity_id": activity_id, "status": {"$ne": "pending"}}
            )
            if doc:
                print(f"\n[UniStack HITL] Decision received: {doc['status']}")
                return doc["status"]
            time.sleep(self._hitl_poll_interval)

    # ── Adapter interface — subclasses must implement these ──────────────────

    def node(self, fn):
        raise NotImplementedError

    def guardrail(self, policy: str):
        raise NotImplementedError

    def hitl(self, message: str, data: dict = None):
        raise NotImplementedError

    def run(self, *args, **kwargs) -> RunResult:
        raise NotImplementedError
