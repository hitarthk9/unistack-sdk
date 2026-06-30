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
