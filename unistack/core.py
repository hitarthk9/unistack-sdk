"""
UniStack SDK — guardrails + human-in-the-loop for existing LangGraph agents.

The SDK latches onto an already-built LangGraph `StateGraph` without touching the
author's nodes. It compiles the graph with static breakpoints (`interrupt_after`)
after each guarded / reviewed node, then drives the pauses:

  - guard   → after the node runs, an LLM-as-judge evaluates its output against a
              policy. Passed → continue silently. Breached → open a HITL pause.
              If the judge itself fails (API error, malformed verdict) the guard
              fails CLOSED: the output is treated as a breach and a human decides.
  - review  → unconditional human sign-off after the node, before continuing.

HITL is **durable and non-blocking**. Graph state is persisted by a durable
checkpointer (MongoDB by default), so an activity can pause for a human and be
resumed later — in a different process, after a restart. `start()` runs until a
pause or END and returns; `resume()` (triggered by the human's decision) loads the
persisted state and continues. The checkpointer is the state store; a small
`hitl_resolutions` collection is the per-pause resolution lock + audit stub (exactly
one resolver wins a pause — concurrent duplicates become no-ops); LangSmith is the
pending-index + trace audit (an open `hitl_pause` span = a pending approval).

All configuration is passed explicitly at init — the SDK reads no environment
variables, writes no environment variables, and loads no .env file.

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
import logging
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

if TYPE_CHECKING:
    from langgraph.graph import StateGraph
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger("unistack")

# Fixed namespace so a pause span's LangSmith run id is deterministic from
# (activity_id, checkpoint_id) — open and close address the exact same run with no
# query (avoids ingestion-latency races when pauses resolve quickly).
_PAUSE_NS = uuid.UUID("7b3f9c2a-1e5d-4a6b-8c0f-2d9e4a1b6c7e")

_DECISIONS = ("approved", "rejected")


class UniStackError(RuntimeError):
    """Raised for graph shapes UniStack cannot drive (e.g. dynamic interrupt())."""


@dataclass
class RunResult:
    activity_id: str
    state: dict = field(default_factory=dict)
    status: str = "completed"  # "completed" | "paused" | "hitl_rejected" | "not_found" | "failed"
    node: str | None = None   # the node a pause fired after (when status == "paused")
    message: str | None = None  # human-readable reason for the pause

    def __post_init__(self):
        self.state = self.state or {}


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
        self._mongo_uri = mongo_uri
        self._mongo: MongoClient | None = None      # lazy — connected on first use
        self._db_name = db_name
        self._checkpointer = checkpointer   # None → build MongoDBSaver in compile()
        self._resolutions_indexed = False

        self._guards: dict[str, str] = {}   # node -> policy text
        self._reviews: set[str] = set()     # nodes needing unconditional sign-off

        # LangSmith: traces + the HITL pending-index / audit. Enabled when a key is
        # supplied; the SDK's own state/resume never depend on it — only discovery does.
        # The client is instance-scoped: nothing is written to os.environ.
        self._tracing = bool(langsmith_api_key)
        self._project = (langsmith_project or workflow) if self._tracing else None
        self._ls = None
        if langsmith_api_key:
            from langsmith import Client
            self._ls = Client(api_key=langsmith_api_key)

    @classmethod
    def init(cls, *args, **kwargs) -> "UniStack":
        return cls(*args, **kwargs)

    # ── Connections ──────────────────────────────────────────────────────────────

    def _client(self) -> MongoClient:
        if self._mongo is None:
            self._mongo = MongoClient(self._mongo_uri)
        return self._mongo

    def close(self) -> None:
        """Close the SDK's own MongoDB connection (a custom checkpointer's client is untouched)."""
        if self._mongo is not None:
            self._mongo.close()
            self._mongo = None

    def __enter__(self) -> "UniStack":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── Latch-on ──────────────────────────────────────────────────────────────

    def compile(self, builder: "StateGraph", guards: dict[str, str] | None = None,
                reviews: list[str] | None = None) -> "CompiledStateGraph":
        """
        Compile an EXISTING StateGraph builder with guardrail + HITL breakpoints and a
        durable checkpointer. The author's nodes are untouched — UniStack only adds
        static breakpoints (`interrupt_after`) and drives the pauses.
        """
        self._guards = dict(guards or {})
        self._reviews = set(reviews or [])
        stops = list(self._guards) + [n for n in self._reviews if n not in self._guards]
        # Keep a handle on the checkpointer so a terminal activity can have its Mongo
        # working-state deleted (see _delete_checkpoint). LangSmith traces are untouched.
        self._checkpointer = self._checkpointer or MongoDBSaver(self._client(), db_name=self._db_name)
        return builder.compile(checkpointer=self._checkpointer, interrupt_after=stops)

    # ── Durable, non-blocking run cycle ─────────────────────────────────────────

    def start(self, graph, initial_state: dict, run_id: str | None = None) -> RunResult:
        """
        Begin an activity. Runs until the first human pause (guard breach / review) or
        END, then RETURNS — never blocks. On a pause, records a pending resolution and
        opens a `hitl_pause` span (the pending-approval marker), then returns "paused".

        run_id defaults to a UTC microsecond timestamp plus a 4-hex-char suffix, so
        concurrent starts (e.g. on different replicas) can never share a thread.
        """
        run_id = run_id or (
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}-{secrets.token_hex(2)}"
        )
        activity_id = f"{self._workflow}-{run_id}"
        return self._drive(graph, initial_state, activity_id)

    def resume(self, graph, activity_id: str, decision: str,
               resolved_by: str | None = None) -> RunResult:
        """
        Continue a paused activity with a human decision ("approved" / "rejected").
        Loads the persisted checkpoint — works in a different process, after a restart.

        The pause's resolution is CLAIMED atomically in `hitl_resolutions`, so exactly
        one resolver wins: a concurrent or repeated resolve of the same pause is a
        recorded no-op, never a second advance. Resolving an id that never existed
        returns status "not_found"; an already-finalized activity returns "completed".

        Reject halts the activity. Approve advances to the next pause or END. Any
        terminal outcome deletes the thread's Mongo checkpoints — nothing will ever
        resume from it again. `state.values` is captured up front, so it is returned
        intact even after the delete.
        """
        if decision not in _DECISIONS:
            raise ValueError(f"decision must be one of {_DECISIONS}, got {decision!r}")
        config = self._config(activity_id)
        snapshot = graph.get_state(config)
        checkpoint_id = snapshot.config.get("configurable", {}).get("checkpoint_id")

        if checkpoint_id is None:
            # No persisted thread: never started, or already finalized (terminal threads
            # are deleted). The resolution records disambiguate the two.
            if self._resolutions().find_one({"activity_id": activity_id}):
                return RunResult(activity_id, {}, "completed",
                                 message="activity already finalized — nothing to resume")
            return RunResult(activity_id, {}, "not_found",
                             message=f"no such activity: '{activity_id}'")

        if not self._claim_resolution(activity_id, checkpoint_id, decision, resolved_by):
            prior = self._resolutions().find_one(
                {"activity_id": activity_id, "checkpoint_id": checkpoint_id}) or {}
            status = "paused" if snapshot.next else "completed"
            return RunResult(
                activity_id, dict(snapshot.values), status,
                message=(f"pause already resolved ({prior.get('decision')} "
                         f"by {prior.get('resolved_by')}) — no-op"))

        self._close_pause_span(activity_id, checkpoint_id, decision, resolved_by)
        if decision != "approved":               # reject always halts the activity
            self._delete_checkpoint(activity_id)
            return RunResult(activity_id, dict(snapshot.values), "hitl_rejected")
        if not snapshot.next:                    # terminal pause → nothing left to run
            self._delete_checkpoint(activity_id)
            return RunResult(activity_id, dict(snapshot.values), "completed")
        return self._drive(graph, None, activity_id)   # resume from the checkpoint

    def run(self, graph, initial_state: dict, run_id: str | None = None,
            decide: Callable[[RunResult], str] | None = None) -> RunResult:
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
        """
        Advance the graph from input_val until a human pause or END; build a RunResult.
        Every guarded node that ran in a segment is judged (parallel fan-out included).
        Reaching END is terminal → the thread's Mongo checkpoints are deleted (nothing will
        resume from it again). A pause keeps its checkpoint (that's what resume() loads).
        """
        config = self._config(activity_id)
        while True:
            try:
                updates, interrupted = self._stream_segment(graph, input_val, config)
            except UniStackError:
                self._delete_checkpoint(activity_id)     # unsupported shape → thread is dead
                raise
            if not interrupted:
                return self._completed(graph, config, activity_id)   # ran to END
            input_val = None

            breaches = []
            for node, output in updates:
                if node not in self._guards:
                    continue
                verdict = self._judge(node, output, activity_id)
                if verdict["passed"]:
                    logger.info("guard[%s] passed — continuing", node)
                else:
                    breaches.append((node, verdict["reason"]))
            review_hits = [n for n, _ in updates
                           if n in self._reviews and n not in self._guards]

            if breaches:
                node = breaches[0][0]
                message = "; ".join(f"Guardrail breach after '{n}': {r}" for n, r in breaches)
                if review_hits:
                    message += f" (also pending sign-off: {', '.join(review_hits)})"
            elif review_hits:
                node = review_hits[0]
                message = f"Human sign-off required after '{node}'."
            else:                                        # all guards passed / unowned stop
                if not graph.get_state(config).next:
                    return self._completed(graph, config, activity_id)
                continue

            snapshot = graph.get_state(config)
            checkpoint_id = snapshot.config.get("configurable", {}).get("checkpoint_id")
            logger.info("paused after '%s': %s", node, message)
            self._record_pending(activity_id, checkpoint_id, node)
            self._open_pause_span(activity_id, node, message, checkpoint_id)
            return RunResult(activity_id, dict(snapshot.values), "paused", node, message)

    def _judge(self, node: str, output, activity_id: str) -> dict:
        """Judge one node's output. Never raises — a failing judge fails closed (breach)."""
        try:
            verdict = self.evaluate(self._guards[node], json.dumps(output, default=str),
                                    thread_id=activity_id)
        except Exception as exc:
            verdict = {"passed": False, "reason": f"guardrail judge error: {exc}"}
        if not isinstance(verdict, dict) or not isinstance(verdict.get("passed"), bool):
            verdict = {"passed": False, "reason": f"malformed guardrail verdict: {verdict!r}"}
        return verdict

    def _completed(self, graph, config, activity_id: str) -> RunResult:
        """Build a completed RunResult, capturing final state BEFORE deleting the checkpoints."""
        values = self._values(graph, config)                 # read state first
        self._delete_checkpoint(activity_id)                 # then wipe the terminal thread
        return RunResult(activity_id, values, "completed")

    @staticmethod
    def _stream_segment(graph, input_val, config):
        """
        Run until the next breakpoint or END. Returns (updates, interrupted) where
        updates is the ordered [(node, output), ...] the segment produced — a parallel
        super-step contributes every node that ran, so no guarded node is skipped.
        """
        updates, interrupted = [], False
        for chunk in graph.stream(input_val, config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                if chunk["__interrupt__"]:   # payload ⇒ dynamic interrupt(), not our breakpoint
                    raise UniStackError(
                        "the graph raised a dynamic interrupt() inside a node — UniStack "
                        "drives static breakpoints only (guards/reviews); remove the "
                        "interrupt() call or handle that node outside UniStack")
                interrupted = True
                continue
            updates.extend(chunk.items())
        return updates, interrupted

    def _config(self, activity_id: str) -> dict:
        config = {
            "configurable": {"thread_id": activity_id},
            "run_name":     activity_id,
            "metadata":     self._thread_meta(activity_id),
            "tags":         [activity_id, self._workflow],
        }
        if self._ls:
            # Instance-scoped tracing: an explicit tracer bound to this SDK's client —
            # nothing global, no os.environ writes.
            from langchain_core.tracers.langchain import LangChainTracer
            config["callbacks"] = [LangChainTracer(client=self._ls, project_name=self._project)]
        return config

    def _thread_meta(self, activity_id: str) -> dict:
        """Metadata that groups every run of an activity into one LangSmith thread."""
        return {"thread_id": activity_id, "activity_id": activity_id, "workflow": self._workflow}

    @staticmethod
    def _values(graph, config) -> dict:
        return dict(graph.get_state(config).values)

    def _delete_checkpoint(self, activity_id: str) -> None:
        """
        Delete all Mongo checkpoint documents for a terminal activity (completed/rejected) —
        nothing will ever resume from this thread again. Uses the checkpointer's own
        delete_thread(); wrapped best-effort so a Mongo hiccup during cleanup can never
        change the returned status/state. LangSmith's thread history is never touched.
        """
        try:
            self._checkpointer.delete_thread(activity_id)
        except Exception as exc:                             # cleanup is best-effort
            logger.warning("could not delete checkpoints for %s: %s", activity_id, exc)

    # ── Resolution claims (hitl_resolutions = per-pause lock + audit stub) ───────

    def _resolutions(self):
        coll = self._client()[self._db_name]["hitl_resolutions"]
        if not self._resolutions_indexed:
            coll.create_index([("activity_id", 1), ("checkpoint_id", 1)], unique=True)
            self._resolutions_indexed = True
        return coll

    def _record_pending(self, activity_id: str, checkpoint_id: str | None, node: str) -> None:
        """Record an open pause. Best-effort: resume() can claim even without this record."""
        try:
            self._resolutions().update_one(
                {"activity_id": activity_id, "checkpoint_id": checkpoint_id},
                {"$setOnInsert": {"status": "pending", "node": node,
                                  "workflow": self._workflow,
                                  "opened_at": datetime.now(timezone.utc)}},
                upsert=True)
        except Exception as exc:
            logger.warning("could not record pending pause for %s: %s", activity_id, exc)

    def _claim_resolution(self, activity_id: str, checkpoint_id: str,
                          decision: str, resolved_by: str | None) -> bool:
        """
        Atomically claim this pause's resolution — exactly one resolver wins. Returns
        False when the pause was already resolved (a concurrent/repeated resolve).
        If the pending record is missing (a crash between checkpoint and record), the
        insert IS the claim; a duplicate-key error means someone else just won.
        """
        coll = self._resolutions()
        fields = {"status": "resolved", "decision": decision,
                  "resolved_by": resolved_by, "resolved_at": datetime.now(timezone.utc)}
        won = coll.find_one_and_update(
            {"activity_id": activity_id, "checkpoint_id": checkpoint_id, "status": "pending"},
            {"$set": fields})
        if won is not None:
            return True
        try:
            coll.insert_one({"activity_id": activity_id, "checkpoint_id": checkpoint_id,
                             "node": None, "workflow": self._workflow,
                             "opened_at": fields["resolved_at"], **fields})
            return True
        except DuplicateKeyError:
            return False

    # ── Guardrail evaluation ──────────────────────────────────────────────────

    def evaluate(self, policy: str, output: str, thread_id: str | None = None) -> dict:
        """
        Evaluate output against a policy using Claude (LLM-as-judge).
        Returns {"passed": bool, "reason": str}. Falls back to a keyword scan when no
        anthropic_api_key was supplied; fails closed (passed=False) if the judge errors.
        When thread_id is given, the guardrail_eval trace joins that LangSmith thread.
        """
        from unistack._guardrail import evaluate_guardrail
        if self._ls:
            from langsmith import tracing_context
            with tracing_context(enabled=True, client=self._ls, project_name=self._project,
                                 metadata=self._thread_meta(thread_id) if thread_id else None):
                return evaluate_guardrail(policy, output, self._guardrail_context,
                                          api_key=self._anthropic_api_key,
                                          model=self._guardrail_model)
        return evaluate_guardrail(policy, output, self._guardrail_context,
                                  api_key=self._anthropic_api_key, model=self._guardrail_model)

    # ── HITL pause span (LangSmith = pending index + audit) ─────────────────────

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
            logger.warning("could not open hitl_pause span for %s: %s", activity_id, exc)

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
            logger.warning("could not close hitl_pause span for %s: %s", activity_id, exc)

    # ── Local interactive decision (run() default) ─────────────────────────────

    @staticmethod
    def _prompt_decision(result: "RunResult") -> str:
        print(f"\n[UniStack] HITL pause on {result.activity_id} after '{result.node}':")
        print(f"  {result.message}")
        answer = input("  Approve? [y/N] ").strip().lower()
        return "approved" if answer in ("y", "yes") else "rejected"


__all__ = ["UniStack", "RunResult", "UniStackError"]
