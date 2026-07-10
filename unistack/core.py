"""
UniStack SDK — guardrails + human-in-the-loop for existing LangGraph agents.

The SDK latches onto an already-built LangGraph `StateGraph` without touching the
author's nodes. It compiles the graph with static breakpoints (`interrupt_after`)
after each guarded / reviewed node, then drives the pauses:

  - guard   → after the node runs, an LLM-as-judge evaluates its output against a
              policy. Passed → continue silently. Breached → open a HITL pause.
  - review  → unconditional human sign-off after the node, before continuing.

HITL is **durable and non-blocking**. Graph state is persisted by a durable
checkpointer (MongoDB by default), so an activity can pause for a human and be
resumed later — in a different process, after a restart. `start()` runs until a
pause or END and returns; `resume()` (triggered by the human's decision) loads the
persisted state and continues. There is no in-memory blocking and no Mongo queue:
the checkpointer is the only state store, and LangSmith is the pending-index + audit
(an open `hitl_pause` span = a pending approval; the closed span is the record).

All configuration is passed explicitly at init — the SDK reads no environment
variables and loads no .env file.

Usage::

    from unistack import UniStack
    from my_app.graph import builder          # existing, untouched StateGraph

    sdk   = UniStack.init(workflow="content", mongo_uri="mongodb://localhost:27017",
                          anthropic_api_key="sk-ant-...", langsmith_api_key="ls-...")
    graph = sdk.compile(builder, guards={"generate": "No unverified claims."},
                        reviews=["publish"])

    # Production (durable, non-blocking): a service starts and later resumes.
    r = sdk.start(graph, {"topic": "..."})     # -> status "paused" | "completed"
    r = sdk.resume(graph, r.activity_id, "approved")

    # Local convenience (single process): blocks, asking for the decision at each pause.
    r = sdk.run(graph, {"topic": "..."})
"""

import json
import os
import uuid
from datetime import datetime, timezone

from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient

# Fixed namespace so a pause span's LangSmith run id is deterministic from
# (activity_id, checkpoint_id) — open and close address the exact same run with no
# query (avoids ingestion-latency races when pauses resolve quickly).
_PAUSE_NS = uuid.UUID("7b3f9c2a-1e5d-4a6b-8c0f-2d9e4a1b6c7e")


class RunResult:
    def __init__(self, activity_id: str, state: dict, status: str = "completed",
                 node: str | None = None, message: str | None = None):
        self.activity_id = activity_id
        self.state = state or {}
        self.status = status  # "completed" | "paused" | "hitl_rejected" | "failed"
        self.node = node          # the node a pause fired after (when status == "paused")
        self.message = message    # human-readable reason for the pause


class UniStack:
    """Guardrail + durable HITL orchestration for LangGraph graphs."""

    def __init__(
        self,
        workflow: str,
        mongo_uri: str,
        anthropic_api_key: str | None = None,
        langsmith_api_key: str | None = None,
        langsmith_project: str | None = None,
        context: str | None = None,
        db_name: str = "unistack",
        guardrail_model: str = "claude-haiku-4-5-20251001",
        checkpointer=None,
    ):
        self._workflow = workflow
        self._anthropic_api_key = anthropic_api_key
        self._guardrail_model = guardrail_model
        self._guardrail_context = context
        self._mongo = MongoClient(mongo_uri)
        self._db_name = db_name
        self._checkpointer = checkpointer   # None → build MongoDBSaver in compile()

        self._guards: dict[str, str] = {}   # node -> policy text
        self._reviews: set[str] = set()     # nodes needing unconditional sign-off

        # LangSmith: traces + the HITL pending-index / audit. Enabled when a key is
        # supplied; the SDK's own state/resume never depend on it — only discovery does.
        self._tracing = bool(langsmith_api_key)
        self._project = (langsmith_project or workflow) if self._tracing else None
        self._ls = None
        if langsmith_api_key:
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
            os.environ["LANGSMITH_PROJECT"] = self._project
            from langsmith import Client
            self._ls = Client()

    @classmethod
    def init(cls, *args, **kwargs) -> "UniStack":
        return cls(*args, **kwargs)

    # ── Latch-on ──────────────────────────────────────────────────────────────

    def compile(self, builder, guards: dict | None = None, reviews: list | None = None):
        """
        Compile an EXISTING StateGraph builder with guardrail + HITL breakpoints and a
        durable checkpointer. The author's nodes are untouched — UniStack only adds
        static breakpoints (`interrupt_after`) and drives the pauses.
        """
        self._guards = dict(guards or {})
        self._reviews = set(reviews or [])
        stops = list(self._guards) + [n for n in self._reviews if n not in self._guards]
        checkpointer = self._checkpointer or MongoDBSaver(self._mongo, db_name=self._db_name)
        return builder.compile(checkpointer=checkpointer, interrupt_after=stops)

    # ── Durable, non-blocking run cycle ─────────────────────────────────────────

    def start(self, graph, initial_state: dict, run_id: str | None = None) -> RunResult:
        """
        Begin an activity. Runs until the first human pause (guard breach / review) or
        END, then RETURNS — never blocks. On a pause, opens a `hitl_pause` span (the
        pending-approval marker) and returns status "paused".

        run_id defaults to a unique UTC microsecond timestamp.
        """
        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        activity_id = f"{self._workflow}-{run_id}"
        return self._drive(graph, initial_state, activity_id)

    def resume(self, graph, activity_id: str, decision: str,
               resolved_by: str | None = None) -> RunResult:
        """
        Continue a paused activity with a human decision ("approved" / "rejected").
        Loads the persisted checkpoint — works in a different process, after a restart.

        Reject halts the activity. Approve advances to the next pause or END; when there
        is nothing left to run (a terminal pause, or an already-finished thread) it simply
        returns "completed", so a repeated approve is a harmless no-op.
        """
        config = self._config(activity_id)
        state = graph.get_state(config)
        self._close_pause_span(activity_id, self._checkpoint_id(graph, config), decision, resolved_by)

        if decision != "approved":               # reject always halts the activity
            return RunResult(activity_id, dict(state.values), "hitl_rejected")
        if not state.next:                       # terminal pause or already done → nothing to run
            return RunResult(activity_id, dict(state.values), "completed")
        return self._drive(graph, None, activity_id)   # resume from the checkpoint

    def run(self, graph, initial_state: dict, run_id: str | None = None, decide=None) -> RunResult:
        """
        Local, single-process convenience: start the activity and, at each pause, obtain
        the decision from `decide(result) -> "approved"|"rejected"` (default: an
        interactive terminal prompt), then resume — looping until completion or rejection.

        Production uses start()/resume() driven by a service, not this.
        """
        decide = decide or self._prompt_decision
        result = self.start(graph, initial_state, run_id)
        while result.status == "paused":
            result = self.resume(graph, result.activity_id, decide(result), resolved_by="local")
        return result

    # ── Drive loop (shared by start + resume) ──────────────────────────────────

    def _drive(self, graph, input_val, activity_id: str) -> RunResult:
        """Advance the graph from input_val until a human pause or END; build a RunResult."""
        config = self._config(activity_id)
        while True:
            node, output, interrupted = self._stream_segment(graph, input_val, config)
            if not interrupted:
                return RunResult(activity_id, self._values(graph, config), "completed")

            if node in self._guards:
                verdict = self.evaluate(self._guards[node], json.dumps(output, default=str),
                                        thread_id=activity_id)
                if verdict["passed"]:
                    print(f"[UniStack] guard[{node}] passed — continuing.")
                    input_val = None
                    if not graph.get_state(config).next:
                        return RunResult(activity_id, self._values(graph, config), "completed")
                    continue
                message = f"Guardrail breach after '{node}': {verdict['reason']}"
            elif node in self._reviews:
                message = f"Human sign-off required after '{node}'."
            else:                                            # a breakpoint we don't own
                input_val = None
                if not graph.get_state(config).next:
                    return RunResult(activity_id, self._values(graph, config), "completed")
                continue

            print(f"[UniStack] paused after '{node}': {message}")
            self._open_pause_span(activity_id, node, message, self._checkpoint_id(graph, config))
            return RunResult(activity_id, self._values(graph, config), "paused", node, message)

    @staticmethod
    def _stream_segment(graph, input_val, config):
        """Run until the next breakpoint or END. Returns (last_node, last_output, interrupted)."""
        last_node, last_output, interrupted = None, None, False
        for chunk in graph.stream(input_val, config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                interrupted = True
                continue
            for node, output in chunk.items():
                last_node, last_output = node, output
        return last_node, last_output, interrupted

    def _config(self, activity_id: str) -> dict:
        return {
            "configurable": {"thread_id": activity_id},
            "run_name":     activity_id,
            "metadata":     self._thread_meta(activity_id),
            "tags":         [activity_id, self._workflow],
        }

    def _thread_meta(self, activity_id: str) -> dict:
        """Metadata that groups every run of an activity into one LangSmith thread."""
        return {"thread_id": activity_id, "activity_id": activity_id, "workflow": self._workflow}

    @staticmethod
    def _values(graph, config) -> dict:
        return dict(graph.get_state(config).values)

    # ── Guardrail evaluation ──────────────────────────────────────────────────

    def evaluate(self, policy: str, output: str, thread_id: str | None = None) -> dict:
        """
        Evaluate output against a policy using Claude (LLM-as-judge).
        Returns {"passed": bool, "reason": str}. Falls back to a keyword scan when no
        anthropic_api_key was supplied. When thread_id is given, the guardrail_eval trace
        joins that LangSmith thread.
        """
        from unistack._guardrail import evaluate_guardrail
        extra = {"metadata": self._thread_meta(thread_id)} if thread_id else None
        return evaluate_guardrail(
            policy, output, self._guardrail_context,
            api_key=self._anthropic_api_key, model=self._guardrail_model,
            langsmith_extra=extra,
        )

    # ── HITL pause span (LangSmith = pending index + audit) ─────────────────────

    @staticmethod
    def _checkpoint_id(graph, config) -> str | None:
        return graph.get_state(config).config.get("configurable", {}).get("checkpoint_id")

    def _span_id(self, activity_id: str, checkpoint_id: str | None) -> str:
        return str(uuid.uuid5(_PAUSE_NS, f"{activity_id}:{checkpoint_id}"))

    def _open_pause_span(self, activity_id, node, message, checkpoint_id) -> None:
        """Open a `hitl_pause` run (posted, left un-ended). An open one = a pending approval."""
        if not self._ls:
            return
        rid = self._span_id(activity_id, checkpoint_id)
        start = datetime.now(timezone.utc)
        try:
            self._ls.create_run(
                name="hitl_pause", run_type="tool",
                inputs={"activity_id": activity_id, "node": node, "message": message},
                id=rid, trace_id=rid, start_time=start,
                dotted_order=f"{start.strftime('%Y%m%dT%H%M%S%f')}Z{rid}",
                project_name=self._project, extra={"metadata": self._thread_meta(activity_id)},
                tags=[activity_id, self._workflow],
            )
        except Exception as exc:                             # tracing is best-effort
            print(f"[UniStack] could not open hitl_pause span for {activity_id}: {exc}")

    def _close_pause_span(self, activity_id, checkpoint_id, decision, resolved_by) -> None:
        """Close the `hitl_pause` run for this pause (its duration is the human wait)."""
        if not self._ls:
            return
        try:
            self._ls.update_run(
                self._span_id(activity_id, checkpoint_id), end_time=datetime.now(timezone.utc),
                outputs={"decision": decision, "resolved_by": resolved_by},
            )
        except Exception as exc:
            print(f"[UniStack] could not close hitl_pause span for {activity_id}: {exc}")

    # ── Local interactive decision (run() default) ─────────────────────────────

    @staticmethod
    def _prompt_decision(result: "RunResult") -> str:
        print(f"\n[UniStack] HITL pause on {result.activity_id} after '{result.node}':")
        print(f"  {result.message}")
        answer = input("  Approve? [y/N] ").strip().lower()
        return "approved" if answer in ("y", "yes") else "rejected"


__all__ = ["UniStack", "RunResult"]
