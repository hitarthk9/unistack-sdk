"""
Native adapter — for raw LLM calls, CrewAI, AutoGen, or any sequential pipeline
that doesn't use LangGraph's interrupt/resume mechanism.

HITL is synchronous and blocking: hitl() writes to the queue and polls until
a human resolves it via the API, then returns the decision to the caller.
Guardrail breaches raise directly (no interrupt pause).

Import path:
    from unistack.adapters.native import UniStack
    # or alias: from unistack.adapters.native import UniStack as UniStackNative

The public interface (node, guardrail, hitl, run, init) is identical to the
LangGraph adapter so that agent code is portable across adapters.
"""
import functools

from opentelemetry import context, trace
from opentelemetry.trace import Status, StatusCode, set_span_in_context

from unistack.core import (
    UniStackCore,
    RunResult,
    _current_activity,
    _current_workflow,
)
from unistack._guardrail import GuardrailBreached, evaluate_guardrail


class UniStack(UniStackCore):
    """
    UniStack SDK — native (framework-agnostic) adapter.

    Works with any Python callable: raw Anthropic/OpenAI calls, CrewAI, AutoGen,
    Pydantic AI, or any sequential pipeline. No LangGraph required.

    Usage::

        from unistack.adapters.native import UniStack

        sdk = UniStack.init("mongodb://localhost:27017", workflow="my-pipeline",
                            context="Domain knowledge for guardrail evaluation.")

        @sdk.node
        def call_llm(state: dict) -> dict: ...

        @sdk.node
        @sdk.guardrail("No PII in output")
        def generate_report(state: dict) -> dict: ...

        @sdk.node
        def approval_step(state: dict) -> dict:
            decision = sdk.hitl("Needs sign-off", data=state)
            # decision is "approved" or "rejected" — handle as needed
            return {"approved": decision == "approved"}

        result = sdk.run(my_pipeline_fn, initial_state, run_id="2026-06-30")
        # my_pipeline_fn(initial_state) is called with a root OTel span open.
        # It can call any @sdk.node-decorated functions internally.
    """

    # ── @sdk.node ────────────────────────────────────────────────────────────

    def node(self, fn):
        """Decorator: wraps any callable with OTel span tracing."""
        @functools.wraps(fn)
        def wrapper(state):
            activity_id = _current_activity.get()
            workflow = _current_workflow.get()

            span = self._tracer.start_span(fn.__name__)
            span.set_attribute("unistack.activity_id", activity_id)
            span.set_attribute("unistack.workflow", workflow)
            ctx_token = context.attach(set_span_in_context(span))
            try:
                result = fn(state)
                span.set_attribute("unistack.status", "completed")
                span.set_status(Status(StatusCode.OK))
                return result
            except GuardrailBreached as e:
                span.set_attribute("unistack.status", "guardrail_breached")
                span.set_attribute("unistack.guardrail_policy", e.policy)
                span.set_attribute("unistack.guardrail_reason", e.reason)
                span.set_status(Status(StatusCode.OK))
                raise
            except Exception:
                span.set_attribute("unistack.status", "failed")
                span.set_status(Status(StatusCode.ERROR))
                raise
            finally:
                span.end()
                context.detach(ctx_token)
        return wrapper

    # ── @sdk.guardrail(policy) ───────────────────────────────────────────────

    def guardrail(self, policy: str):
        """
        Decorator: LLM-evaluated policy check after the function runs.
        Apply BELOW @sdk.node.

        On breach, raises GuardrailBreached immediately (no interrupt pause —
        native pipelines have no resume mechanism). sdk.run() catches this and
        records result.status = "guardrail_breached".
        """
        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(state):
                result = fn(state)
                evaluation = evaluate_guardrail(policy, str(result), self._guardrail_context)
                if not evaluation["passed"]:
                    raise GuardrailBreached(policy, evaluation["reason"])
                return result
            return wrapper
        return decorator

    # ── sdk.hitl() ───────────────────────────────────────────────────────────

    def hitl(self, message: str, data: dict = None) -> str:
        """
        Trigger a blocking HITL pause. Writes to hitl_queue and polls until
        a human resolves it via the HITL API.

        Returns "approved" or "rejected". The calling node is responsible
        for acting on the returned decision.
        """
        activity_id = _current_activity.get()
        self._write_hitl_entry(
            activity_id,
            type_="hitl",
            message=message,
            data=data or {},
        )
        return self._poll_for_decision(activity_id)

    # ── sdk.run() ────────────────────────────────────────────────────────────

    def run(self, fn, initial_input: dict, run_id: str) -> RunResult:
        """
        Execute a pipeline function as a UniStack activity.

        fn(initial_input) is called with a root OTel span active, making all
        @sdk.node child spans automatic descendants. Blocking: returns when fn
        returns (including any HITL blocking inside nodes).

        Unlike the LangGraph adapter, run() here takes a plain callable, not
        a compiled graph. fn can call other @sdk.node-decorated functions.
        """
        activity_id = f"{self._workflow}-{run_id}"
        act_token = _current_activity.set(activity_id)
        wf_token = _current_workflow.set(self._workflow)

        root_span = self._tracer.start_span(self._workflow)
        root_span.set_attribute("unistack.activity_id", activity_id)
        root_span.set_attribute("unistack.workflow", self._workflow)
        root_ctx = context.attach(set_span_in_context(root_span))

        final_state: dict = {}
        result_status = "completed"
        try:
            final_state = fn(initial_input) or {}
            root_span.set_attribute("unistack.status", "completed")
            root_span.set_status(Status(StatusCode.OK))
        except GuardrailBreached as e:
            root_span.set_attribute("unistack.status", "guardrail_breached")
            root_span.set_attribute("unistack.guardrail_reason", e.reason)
            root_span.set_status(Status(StatusCode.OK))
            result_status = "guardrail_breached"
            print(f"\n[UniStack Guardrail] Execution halted: {e.reason}")
        except Exception as exc:
            root_span.set_attribute("unistack.status", "failed")
            root_span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        finally:
            root_span.end()
            context.detach(root_ctx)
            _current_activity.reset(act_token)
            _current_workflow.reset(wf_token)
            self._provider.force_flush()

        return RunResult(activity_id=activity_id, state=final_state, status=result_status)


__all__ = ["UniStack"]
