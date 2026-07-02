"""
LangGraph adapter — the default UniStack adapter.

Wraps LangGraph nodes with OTel tracing, supports interrupt-based HITL,
and drives graph.stream() with a blocking poll-resume loop.

Import path (backwards-compatible):
    from unistack import UniStack          # via unistack/__init__.py
    from unistack.adapters.langgraph import UniStack  # direct
"""
import functools

from opentelemetry import context
from opentelemetry.trace import Status, StatusCode, set_span_in_context

from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.config import get_config as _lg_get_config
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from unistack.core import (
    UniStackCore,
    RunResult,
    _current_activity,
    _current_workflow,
    _resume_pending,
    _guardrail_approved,
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
        def wrapper(state):
            activity_id = _current_activity.get()
            workflow    = _current_workflow.get()

            # When invoked outside sdk.run() (e.g. tests, alternative runners),
            # _current_activity holds its default "unknown" value. Derive a better
            # activity_id from LangGraph's own config context variable.
            _act_token = _wf_token = None
            if activity_id == "unknown":
                try:
                    cfg = _lg_get_config()
                    activity_id = (cfg.get("configurable") or {}).get("thread_id") or "untracked-run"
                except Exception:
                    activity_id = "untracked-run"
                _act_token = _current_activity.set(activity_id)
            if workflow == "unknown":
                workflow = self._workflow
                _wf_token = _current_workflow.set(workflow)

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
                if _act_token: _current_activity.reset(_act_token)
                if _wf_token:  _current_workflow.reset(_wf_token)
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
                # On a guardrail-approved resume, run() sets this flag so we skip
                # re-evaluating the same output that a human already reviewed and approved.
                if _guardrail_approved.pop(_current_activity.get(), False):
                    return result
                evaluation = evaluate_guardrail(policy, str(result), self._guardrail_context)
                if not evaluation["passed"]:
                    activity_id = _current_activity.get()
                    # LangGraph re-runs the node from scratch on resume. Guard against
                    # re-writing the hitl_queue entry if this specific breach was already
                    # resolved (defence-in-depth alongside _guardrail_approved).
                    existing = self._db.hitl_queue.find_one({"activity_id": activity_id})
                    already_resolved = (
                        existing
                        and existing.get("type") == "guardrail_breach"
                        and existing.get("guardrail_policy") == policy
                        and existing.get("status") != "pending"
                    )
                    if not already_resolved:
                        self._write_guardrail_entry(
                            activity_id,
                            node=fn.__name__,
                            policy=policy,
                            reason=evaluation["reason"],
                            status="pending",
                        )
                        self._write_hitl_entry(
                            activity_id,
                            type_="guardrail_breach",
                            message=f"Guardrail breach: {evaluation['reason']}",
                            data={},
                            policy=policy,
                            reason=evaluation["reason"],
                        )
                    decision = interrupt({
                        "type": "guardrail_breach",
                        "policy": policy,
                        "reason": evaluation["reason"],
                        "node": fn.__name__,
                    })
                    _REJECT = ("n", "no", "reject", "false")
                    if decision is False or str(decision).lower().strip() in _REJECT:
                        raise GuardrailBreached(policy, evaluation["reason"])
                return result
            return wrapper
        return decorator

    # ── sdk.hitl() ───────────────────────────────────────────────────────────

    def hitl(self, message: str, data: dict = None):
        """
        Trigger a HITL pause inside a node.

        Writes the hitl_queue entry immediately (before the interrupt is raised)
        so the record is visible to pollers as soon as the pause begins.

        LangGraph re-runs the node from scratch on resume. The write is skipped
        if this exact HITL entry is already resolved so the approved/rejected
        status is not overwritten back to pending on the resume pass.
        """
        from langgraph.types import interrupt
        activity_id = _current_activity.get()
        if activity_id and activity_id != "unknown":
            existing = self._db.hitl_queue.find_one({"activity_id": activity_id})
            already_resolved = (
                existing
                and existing.get("type") == "hitl"
                and existing.get("message") == message
                and existing.get("status") != "pending"
            )
            if not already_resolved:
                self._write_hitl_entry(
                    activity_id,
                    type_="hitl",
                    message=message,
                    data=data or {},
                )
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

                intr_value = interrupts[0].value if hasattr(interrupts[0], "value") else {}
                is_guardrail_breach = (
                    isinstance(intr_value, dict)
                    and intr_value.get("type") == "guardrail_breach"
                )
                breached_node = intr_value.get("node") if is_guardrail_breach else None

                # Flush node spans to MongoDB now so the visualizer reflects the
                # correct pause state (paused node, completed predecessors) before
                # the human is prompted. Without this, BatchSpanProcessor may
                # buffer spans for up to 5 s before exporting.
                self._provider.force_flush(timeout_millis=3000)

                # hitl_queue entry was already written at the interrupt site
                # (in sdk.hitl() or @sdk.guardrail). Just poll for the decision.
                decision = self._poll_for_decision(activity_id)

                # Resolve the guardrail audit record with the human's decision.
                if is_guardrail_breach and breached_node:
                    hitl_doc = self._db.hitl_queue.find_one({"activity_id": activity_id})
                    self._resolve_guardrail_entry(
                        activity_id,
                        node=breached_node,
                        status="approved" if decision != "rejected" else "rejected",
                        resolved_by=hitl_doc.get("resolved_by") if hitl_doc else None,
                    )

                if decision == "rejected":
                    root_span.set_attribute("unistack.status", "hitl_rejected")
                    root_span.set_status(Status(StatusCode.OK))
                    result_status = "hitl_rejected"
                    break

                if is_guardrail_breach:
                    _guardrail_approved[activity_id] = True

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
            _guardrail_approved.pop(activity_id, None)
            self._provider.force_flush()
            self.checkpointer.delete_thread(activity_id)

        return RunResult(activity_id=activity_id, state=final_state, status=result_status)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _stream_to_completion(self, graph, input_val, config) -> list:
        """Stream until completion or first interrupt. Returns interrupt list or []."""
        for event in graph.stream(input_val, config, stream_mode="updates"):
            if "__interrupt__" in event:
                return list(event["__interrupt__"])
        return []


__all__ = ["UniStack"]
