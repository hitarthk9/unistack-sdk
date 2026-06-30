"""
LangGraph adapter — the default UniStack adapter.

Wraps LangGraph nodes with OTel tracing, supports interrupt-based HITL,
and drives graph.stream() with a blocking poll-resume loop.

Import path (backwards-compatible):
    from unistack import UniStack          # via unistack/__init__.py
    from unistack.adapters.langgraph import UniStack  # direct
"""
import functools

from opentelemetry import context, trace
from opentelemetry.trace import Status, StatusCode, set_span_in_context

from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from unistack.core import (
    UniStackCore,
    RunResult,
    _current_activity,
    _current_workflow,
    _resume_pending,
)
from unistack._guardrail import GuardrailBreached, evaluate_guardrail


class UniStack(UniStackCore):
    """
    UniStack SDK — LangGraph adapter.

    Usage::

        sdk = UniStack.init("mongodb://localhost:27017", workflow="my-workflow")
        # Optional: pass context= or context_file= for guardrail domain knowledge

        @sdk.node
        def my_node(state): ...

        @sdk.node
        @sdk.guardrail("Output must not contain PII")
        def sensitive_step(state): ...

        @sdk.node
        def approval_step(state):
            sdk.hitl("Needs human review", data=state)
            return {}

        graph = builder.compile(checkpointer=sdk.checkpointer)
        result = sdk.run(graph, initial_state, run_id="2026-06-29")
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
        super().__init__(mongo_uri, workflow, db_name, hitl_poll_interval, context, context_file)
        self.checkpointer = MongoDBSaver(self._client, db_name=db_name)

    # ── @sdk.node ────────────────────────────────────────────────────────────

    def node(self, fn):
        """
        Decorator: wraps any LangGraph node with OTel tracing.
        Handles HITL and guardrail review pauses — neither is an OTel error.
        Sets unistack.human_overridden=True on the span that ran after a human resume.
        """
        @functools.wraps(fn)
        def wrapper(state, config=None):
            activity_id = _current_activity.get()
            workflow = _current_workflow.get()

            span = self._tracer.start_span(fn.__name__)
            span.set_attribute("unistack.activity_id", activity_id)
            span.set_attribute("unistack.workflow", workflow)
            ctx_token = context.attach(set_span_in_context(span))
            try:
                result = fn(state)
                if _resume_pending.pop(activity_id, False):
                    span.set_attribute("unistack.human_overridden", True)
                span.set_attribute("unistack.status", "completed")
                span.set_status(Status(StatusCode.OK))
                return result
            except GraphInterrupt as e:
                for intr in e.args[0]:
                    if isinstance(intr.value, dict) and intr.value.get("type") == "guardrail_breach":
                        span.set_attribute("unistack.guardrail_policy", intr.value["policy"])
                        span.set_attribute("unistack.guardrail_reason", intr.value["reason"])
                        break
                span.set_attribute("unistack.status", "hitl_pending")
                span.set_status(Status(StatusCode.OK))
                raise
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
        Decorator: LLM-evaluated policy check after the node runs.
        Apply BELOW @sdk.node.

        On breach, pauses for human review (HITL) instead of halting immediately.
        Human approves → execution continues. Human rejects → guardrail_breached.

        Pass context= or context_file= to UniStack.init() to give the LLM
        evaluator domain-specific business knowledge for more accurate decisions.
        """
        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(state):
                from langgraph.types import interrupt
                result = fn(state)
                evaluation = evaluate_guardrail(policy, str(result), self._guardrail_context)
                if not evaluation["passed"]:
                    decision = interrupt({
                        "type": "guardrail_breach",
                        "policy": policy,
                        "reason": evaluation["reason"],
                    })
                    _REJECT = ("n", "no", "reject", "false")
                    if decision is False or str(decision).lower().strip() in _REJECT:
                        raise GuardrailBreached(policy, evaluation["reason"])
                return result
            return wrapper
        return decorator

    # ── sdk.hitl() ───────────────────────────────────────────────────────────

    def hitl(self, message: str, data: dict = None):
        """Trigger a HITL pause inside a node. Wraps LangGraph's interrupt()."""
        from langgraph.types import interrupt
        return interrupt({"message": message, "data": data or {}})

    # ── sdk.run() ────────────────────────────────────────────────────────────

    def run(self, graph, initial_state: dict, run_id: str) -> RunResult:
        """
        Execute a compiled LangGraph graph as a UniStack activity.

        Blocking: does not return until the workflow fully completes, including
        any number of HITL or guardrail-review pauses. While paused, polls
        unistack.hitl_queue every hitl_poll_interval seconds.
        """
        activity_id = f"{self._workflow}-{run_id}"
        lg_config = {"configurable": {"thread_id": activity_id}}

        act_token = _current_activity.set(activity_id)
        wf_token = _current_workflow.set(self._workflow)

        root_span = self._tracer.start_span(self._workflow)
        root_span.set_attribute("unistack.activity_id", activity_id)
        root_span.set_attribute("unistack.workflow", self._workflow)
        root_ctx = context.attach(set_span_in_context(root_span))

        final_state: dict = {}
        result_status = "completed"
        is_resume = False
        input_val = initial_state

        try:
            while True:
                if is_resume:
                    _resume_pending[activity_id] = True
                try:
                    interrupts = self._stream_to_completion(graph, input_val, lg_config)
                except GuardrailBreached as e:
                    root_span.set_attribute("unistack.status", "guardrail_breached")
                    root_span.set_attribute("unistack.guardrail_reason", e.reason)
                    root_span.set_status(Status(StatusCode.OK))
                    result_status = "guardrail_breached"
                    print(f"\n[UniStack Guardrail] Execution halted: {e.reason}")
                    break
                finally:
                    _resume_pending.pop(activity_id, None)

                if not interrupts:
                    snapshot = graph.get_state(lg_config)
                    final_state = dict(snapshot.values) if snapshot else {}
                    root_span.set_attribute("unistack.status", "completed")
                    root_span.set_status(Status(StatusCode.OK))
                    break

                self._write_hitl_queue(activity_id, interrupts)
                decision = self._poll_for_decision(activity_id)

                if decision == "rejected":
                    root_span.set_attribute("unistack.status", "hitl_rejected")
                    root_span.set_status(Status(StatusCode.OK))
                    result_status = "hitl_rejected"
                    break

                is_resume = True
                input_val = Command(resume=True)

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
            self._db.checkpoints.delete_many({"thread_id": activity_id})
            self._db.checkpoint_writes.delete_many({"thread_id": activity_id})

        return RunResult(activity_id=activity_id, state=final_state, status=result_status)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _stream_to_completion(self, graph, input_val, config) -> list:
        """Stream until completion or first interrupt. Returns interrupt list or []."""
        for event in graph.stream(input_val, config, stream_mode="updates"):
            if "__interrupt__" in event:
                return list(event["__interrupt__"])
        return []

    def _write_hitl_queue(self, activity_id: str, interrupts: list) -> None:
        """Parse LangGraph interrupt objects and write to hitl_queue via core."""
        intr = interrupts[0]
        value = intr.value if hasattr(intr, "value") else {}
        is_guardrail = isinstance(value, dict) and value.get("type") == "guardrail_breach"
        if is_guardrail:
            self._write_hitl_entry(
                activity_id,
                type_="guardrail_breach",
                message=f"Guardrail breach: {value.get('reason', '')}",
                data={},
                policy=value.get("policy"),
                reason=value.get("reason"),
            )
        else:
            msg = value.get("message", "") if isinstance(value, dict) else str(value)
            data = value.get("data", {}) if isinstance(value, dict) else {}
            self._write_hitl_entry(activity_id, type_="hitl", message=msg, data=data)


__all__ = ["UniStack"]
