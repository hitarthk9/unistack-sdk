import functools
import time
from contextvars import ContextVar
from datetime import datetime, timezone

from opentelemetry import context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode, set_span_in_context

from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.errors import GraphInterrupt
from pymongo import MongoClient

from unistack._exporter import MongoDBSpanExporter
from unistack._guardrail import GuardrailBreached, evaluate_guardrail

_current_activity: ContextVar[str] = ContextVar("unistack_activity", default="unknown")
_current_workflow: ContextVar[str] = ContextVar("unistack_workflow", default="unknown")
# Set of activity_ids currently in a resume pass.
# Module-level dict (not ContextVar) so mutations in one node are visible to the next.
# The first node to complete after a resume pops its activity_id; subsequent nodes see nothing.
_resume_pending: dict[str, bool] = {}


class RunResult:
    def __init__(self, activity_id: str, state: dict, status: str = "completed"):
        self.activity_id = activity_id
        self.state = state or {}
        self.status = status  # "completed" | "hitl_rejected" | "guardrail_breached" | "failed"


class UniStack:
    """
    Minimal observability SDK for LangGraph agents.

    Usage::

        sdk = UniStack.init("mongodb://localhost:27017", workflow="my-workflow")

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
        # sdk.run() blocks until fully complete — including any HITL pauses.
        # While paused, the SDK polls unistack.hitl_queue.
        # An external API updates the queue; sdk.run() detects it and resumes.
    """

    def __init__(
        self,
        mongo_uri: str,
        workflow: str,
        db_name: str = "unistack",
        hitl_poll_interval: float = 2.0,
    ):
        self._workflow = workflow
        self._mongo_uri = mongo_uri
        self._db_name = db_name
        self._hitl_poll_interval = hitl_poll_interval
        self._client = MongoClient(mongo_uri)
        self._db = self._client[db_name]
        self._provider = self._setup_otel()
        self._tracer = trace.get_tracer("unistack")
        # MongoDB-backed checkpointer so the graph state survives the polling period
        # and can be resumed even if the process is restarted between HITL events.
        self.checkpointer = MongoDBSaver(self._client, db_name=db_name)

    @classmethod
    def init(
        cls,
        mongo_uri: str,
        workflow: str,
        db_name: str = "unistack",
        hitl_poll_interval: float = 2.0,
    ) -> "UniStack":
        return cls(mongo_uri, workflow, db_name, hitl_poll_interval)

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
                # Promote guardrail-breach context from the interrupt payload to the
                # span so the assembler can distinguish guardrail review from plain HITL.
                for intr in e.args[0]:
                    if isinstance(intr.value, dict) and intr.value.get("type") == "guardrail_breach":
                        span.set_attribute("unistack.guardrail_policy", intr.value["policy"])
                        span.set_attribute("unistack.guardrail_reason", intr.value["reason"])
                        break
                span.set_attribute("unistack.status", "hitl_pending")
                span.set_status(Status(StatusCode.OK))   # pause is not an error
                raise
            except GuardrailBreached as e:
                # Human reviewed and rejected — deliberate policy halt, not a bug.
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
        """
        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(state):
                from langgraph.types import interrupt
                result = fn(state)
                evaluation = evaluate_guardrail(policy, str(result))
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
        """
        Trigger a HITL pause inside a node. Wraps LangGraph's interrupt().
        The @sdk.node decorator catches the resulting GraphInterrupt and marks
        the span as hitl_pending (not failed).
        """
        from langgraph.types import interrupt
        return interrupt({"message": message, "data": data or {}})

    # ── sdk.run() ────────────────────────────────────────────────────────────

    def run(self, graph, initial_state: dict, run_id: str) -> RunResult:
        """
        Execute a compiled LangGraph graph as a UniStack activity.

        Blocking: sdk.run() does not return until the workflow fully completes
        (including any number of HITL or guardrail-review pauses). While paused,
        it polls unistack.hitl_queue every hitl_poll_interval seconds.

        The external HITL API writes the human's decision to hitl_queue.
        sdk.run() detects it, resumes the graph with Command(resume=...), and
        continues the polling loop until no more pauses remain.

        Because sdk.run() holds the root OTel span open throughout — including
        during the polling wait — resumed node spans are automatically children
        of the original root. No span reconstruction is needed.
        """
        from langgraph.types import Command

        activity_id = f"{self._workflow}-{run_id}"
        lg_config = {"configurable": {"thread_id": activity_id}}

        act_token = _current_activity.set(activity_id)
        wf_token = _current_workflow.set(self._workflow)

        # Root span stays alive for the full activity, including HITL wait time.
        # Resumed node spans inherit it as parent automatically via OTel context.
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
                    _resume_pending.pop(activity_id, None)  # clean up if node never ran

                if not interrupts:
                    # Workflow completed normally.
                    snapshot = graph.get_state(lg_config)
                    final_state = dict(snapshot.values) if snapshot else {}
                    root_span.set_attribute("unistack.status", "completed")
                    root_span.set_status(Status(StatusCode.OK))
                    break

                # HITL or guardrail-review pause detected.
                self._write_hitl_queue(activity_id, interrupts)

                decision = self._poll_for_decision(activity_id)

                if decision == "rejected":
                    root_span.set_attribute("unistack.status", "hitl_rejected")
                    root_span.set_status(Status(StatusCode.OK))
                    result_status = "hitl_rejected"
                    break

                # Human approved — resume the graph.
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
            # LangGraph checkpoint data is only needed while the graph is live and
            # potentially resuming. Delete it once the activity finishes.
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

    def _poll_for_decision(self, activity_id: str) -> str:
        """
        Block until the hitl_queue entry for this activity is resolved.
        Returns "approved" or "rejected".
        Prints instructions so the developer knows to call the resolve API
        in a separate terminal while this process is waiting.
        """
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

    def _write_hitl_queue(self, activity_id: str, interrupts: list) -> None:
        """
        Upsert a hitl_queue document for this activity.
        Called when sdk.run() detects a pause — before entering the polling wait.
        The API reads from this collection to surface items on the dashboard.
        """
        intr = interrupts[0]
        value = intr.value if hasattr(intr, "value") else {}

        is_guardrail = isinstance(value, dict) and value.get("type") == "guardrail_breach"

        doc = {
            "activity_id": activity_id,
            "workflow": self._workflow,
            "type": "guardrail_breach" if is_guardrail else "hitl",
            "status": "pending",
            "created_at": datetime.now(tz=timezone.utc),
            "resolved_at": None,
            "resolved_by": None,
            "resolution_comment": None,
        }

        if is_guardrail:
            doc["message"] = f"Guardrail breach: {value.get('reason', '')}"
            doc["data"] = {}
            doc["guardrail_policy"] = value.get("policy")
            doc["guardrail_reason"] = value.get("reason")
        else:
            doc["message"] = value.get("message", "") if isinstance(value, dict) else str(value)
            doc["data"] = value.get("data", {}) if isinstance(value, dict) else {}
            doc["guardrail_policy"] = None
            doc["guardrail_reason"] = None

        self._db.hitl_queue.replace_one(
            {"activity_id": activity_id},
            doc,
            upsert=True,
        )


__all__ = ["UniStack"]
