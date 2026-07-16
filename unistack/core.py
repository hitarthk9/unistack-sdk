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
`hitl_resolutions` collection is the per-pause resolution lock, the pending-approvals
index (`status: "pending"`), and the audit record (exactly one resolver wins a pause —
concurrent duplicates become no-ops).

Tracing is pure OpenTelemetry (OTLP), instance-scoped: each start/resume leg is one
trace, grouped per activity by `session.id`; the human wait is a `hitl_pause` span
emitted retroactively at resolve time into the pausing leg's trace. Point it at any
OTLP/HTTP backend (a collector, a hyperscaler agent, or Langfuse).

All configuration is passed explicitly at init — the SDK reads no environment
variables, writes no environment variables, and loads no .env file.

Usage::

    from unistack import UniStack
    from my_app.graph import builder          # existing, untouched StateGraph

    sdk   = UniStack.init(workflow="content", mongo_uri="mongodb://localhost:27017",
                          anthropic_api_key="sk-ant-...",
                          otel_endpoint="https://langfuse.internal/api/public/otel")
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

from unistack._telemetry import Telemetry, _clip

if TYPE_CHECKING:
    from langgraph.graph import StateGraph
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger("unistack")

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
        otel_endpoint: str | None = None,
        otel_headers: dict | str | None = None,
        otel_service_name: str | None = None,
        context: str | None = None,
        db_name: str = "unistack",
        guardrail_model: str = "claude-haiku-4-5-20251001",
        checkpointer=None,
        tracer_provider=None,
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

        # OpenTelemetry: pure-OTLP tracing, enabled by an endpoint or a caller-supplied
        # provider. Instance-scoped — the provider is never installed globally and
        # nothing is written to os.environ. The SDK's own state/resume never depend on
        # the telemetry backend; every telemetry call is best-effort.
        self._telemetry = Telemetry(workflow, tracer_provider=tracer_provider,
                                    endpoint=otel_endpoint, headers=otel_headers,
                                    service_name=otel_service_name)

    @classmethod
    def init(cls, *args, **kwargs) -> "UniStack":
        return cls(*args, **kwargs)

    # ── Connections ──────────────────────────────────────────────────────────────

    def _client(self) -> MongoClient:
        if self._mongo is None:
            self._mongo = MongoClient(self._mongo_uri)
        return self._mongo

    def close(self) -> None:
        """
        Close the SDK's own MongoDB connection (a custom checkpointer's client is
        untouched) and flush + shut down the SDK-owned tracer provider so buffered
        spans export. A caller-supplied `tracer_provider` is the caller's to close.
        """
        if self._mongo is not None:
            self._mongo.close()
            self._mongo = None
        self._telemetry.shutdown()

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
        # working-state deleted (see _delete_checkpoint). Exported traces are untouched.
        self._checkpointer = self._checkpointer or MongoDBSaver(self._client(), db_name=self._db_name)
        return builder.compile(checkpointer=self._checkpointer, interrupt_after=stops)

    # ── Durable, non-blocking run cycle ─────────────────────────────────────────

    def start(self, graph, initial_state: dict, run_id: str | None = None) -> RunResult:
        """
        Begin an activity. Runs until the first human pause (guard breach / review) or
        END, then RETURNS — never blocks. On a pause, records a pending resolution in
        `hitl_resolutions` (the pending-approval marker, carrying this leg's trace ids),
        then returns "paused".

        run_id defaults to a UTC microsecond timestamp plus a 4-hex-char suffix, so
        concurrent starts (e.g. on different replicas) can never share a thread.
        """
        run_id = run_id or (
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}-{secrets.token_hex(2)}"
        )
        activity_id = f"{self._workflow}-{run_id}"
        with self._telemetry.leg("start", activity_id) as span:
            result = self._drive(graph, initial_state, activity_id)
            self._telemetry.stamp_result(span, result)
        return result

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
        with self._telemetry.leg("resume", activity_id,
                                 {"unistack.decision": decision,
                                  "unistack.resolved_by": resolved_by or ""}) as span:
            result = self._resume(graph, activity_id, decision, resolved_by)
            self._telemetry.stamp_result(span, result)
        return result

    def _resume(self, graph, activity_id: str, decision: str,
                resolved_by: str | None) -> RunResult:
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

        claimed = self._claim_resolution(activity_id, checkpoint_id, decision, resolved_by)
        if claimed is None:
            prior = self._resolutions().find_one(
                {"activity_id": activity_id, "checkpoint_id": checkpoint_id}) or {}
            status = "paused" if snapshot.next else "completed"
            return RunResult(
                activity_id, dict(snapshot.values), status,
                message=(f"pause already resolved ({prior.get('decision')} "
                         f"by {prior.get('resolved_by')}) — no-op"))

        self._telemetry.add_event("resolution_claimed", {
            "decision": decision, "resolved_by": resolved_by or "",
            "checkpoint_id": checkpoint_id or ""})
        self._emit_pause_span(claimed, decision, resolved_by)
        if claimed.get("trace_id") and claimed.get("span_id"):
            self._telemetry.link_current_to(claimed["trace_id"], claimed["span_id"])
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
            self._record_pending(activity_id, checkpoint_id, node, message,
                                 self._telemetry.current_ids())
            self._telemetry.add_event("hitl_pause_opened",
                                      {"node": node, "checkpoint_id": checkpoint_id or ""})
            return RunResult(activity_id, dict(snapshot.values), "paused", node, message)

    def _judge(self, node: str, output, activity_id: str) -> dict:
        """Judge one node's output. Never raises — a failing judge fails closed (breach)."""
        try:
            verdict = self.evaluate(self._guards[node], json.dumps(output, default=str),
                                    thread_id=activity_id, node=node)
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
        handler = self._telemetry.handler(activity_id)
        if handler is not None:
            # Instance-scoped tracing: an explicit OTel callback handler in the graph
            # config — nothing global, no os.environ writes. Node-internal llm.invoke()
            # calls inherit it via langchain-core's contextvar config propagation.
            config["callbacks"] = [handler]
        return config

    def _thread_meta(self, activity_id: str) -> dict:
        """Metadata stamped on every run of an activity (kept on the graph config for
        consumers; the OTel spans carry the same ids as `session.id` attributes)."""
        return {"thread_id": activity_id, "activity_id": activity_id, "workflow": self._workflow}

    @staticmethod
    def _values(graph, config) -> dict:
        return dict(graph.get_state(config).values)

    def _delete_checkpoint(self, activity_id: str) -> None:
        """
        Delete all Mongo checkpoint documents for a terminal activity (completed/rejected) —
        nothing will ever resume from this thread again. Uses the checkpointer's own
        delete_thread(); wrapped best-effort so a Mongo hiccup during cleanup can never
        change the returned status/state. Exported traces are never touched.
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

    def _record_pending(self, activity_id: str, checkpoint_id: str | None, node: str,
                        message: str | None, trace_ids: tuple[str, str] | None) -> None:
        """
        Record an open pause — this doc IS the pending-approvals index (status "pending")
        and carries the pausing leg's trace ids so the `hitl_pause` span can later be
        emitted into that trace. Best-effort: resume() can claim even without it.
        """
        trace_id, span_id = trace_ids or (None, None)
        try:
            self._resolutions().update_one(
                {"activity_id": activity_id, "checkpoint_id": checkpoint_id},
                {"$setOnInsert": {"status": "pending", "node": node, "message": message,
                                  "workflow": self._workflow,
                                  "trace_id": trace_id, "span_id": span_id,
                                  "opened_at": datetime.now(timezone.utc)}},
                upsert=True)
        except Exception as exc:
            logger.warning("could not record pending pause for %s: %s", activity_id, exc)

    def _claim_resolution(self, activity_id: str, checkpoint_id: str,
                          decision: str, resolved_by: str | None) -> dict | None:
        """
        Atomically claim this pause's resolution — exactly one resolver wins. Returns
        the claimed pending doc (its pre-image: node, message, opened_at, trace ids),
        or None when the pause was already resolved (a concurrent/repeated resolve).
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
            return won
        try:
            doc = {"activity_id": activity_id, "checkpoint_id": checkpoint_id,
                   "node": None, "workflow": self._workflow,
                   "opened_at": fields["resolved_at"], **fields}
            coll.insert_one(doc)
            return doc          # crash-path claim: carries no trace ids / opened_at ≈ now
        except DuplicateKeyError:
            return None

    # ── Guardrail evaluation ──────────────────────────────────────────────────

    def evaluate(self, policy: str, output: str, thread_id: str | None = None,
                 node: str | None = None) -> dict:
        """
        Evaluate output against a policy using Claude (LLM-as-judge).
        Returns {"passed": bool, "reason": str}. Falls back to a keyword scan when no
        anthropic_api_key was supplied; fails closed (passed=False) if the judge errors.
        Traced as a `guardrail_eval` span — nested in the running leg's trace when
        called mid-run (thread_id = the activity id), a standalone trace otherwise.
        """
        from unistack._guardrail import evaluate_guardrail
        with self._telemetry.span("guardrail_eval", {
                "unistack.guardrail.node": node,
                "unistack.guardrail.policy": _clip(policy, 1000),
                "unistack.guardrail.mode": "llm" if self._anthropic_api_key else "keyword",
        }, activity_id=thread_id) as span:
            verdict = evaluate_guardrail(policy, output, self._guardrail_context,
                                         api_key=self._anthropic_api_key,
                                         model=self._guardrail_model,
                                         telemetry=self._telemetry)
            self._telemetry.set_attrs(span, {
                "unistack.guardrail.passed": verdict.get("passed"),
                "unistack.guardrail.reason": verdict.get("reason")})
        return verdict

    # ── HITL pause span (emitted retroactively — OTLP cannot export open spans) ──

    def _emit_pause_span(self, doc: dict, decision: str, resolved_by: str | None) -> None:
        """
        Emit the completed `hitl_pause` span into the pausing leg's trace: parent and
        start time come from the pending doc persisted at pause time, the end is now —
        so its duration is the real human wait, across processes and restarts. Called
        only by the claim winner, so it is emitted exactly once. A doc without trace
        ids (crash-path claim, or telemetry was off at pause time) is skipped.
        """
        if not self._telemetry.enabled:
            return
        trace_id, span_id, opened_at = doc.get("trace_id"), doc.get("span_id"), doc.get("opened_at")
        if not (trace_id and span_id and opened_at):
            logger.warning("no trace recorded for pause %s/%s — skipping hitl_pause span",
                           doc.get("activity_id"), doc.get("checkpoint_id"))
            return
        activity_id = doc.get("activity_id")
        self._telemetry.emit_closed_span("hitl_pause", trace_id, span_id, opened_at, {
            "session.id": activity_id,
            "langfuse.session.id": activity_id,
            "unistack.activity_id": activity_id,
            "unistack.workflow": doc.get("workflow") or self._workflow,
            "unistack.pause.node": doc.get("node"),
            "unistack.pause.message": _clip(doc["message"], 1000) if doc.get("message") else None,
            "unistack.decision": decision,
            "unistack.resolved_by": resolved_by or "",
        })

    # ── Local interactive decision (run() default) ─────────────────────────────

    @staticmethod
    def _prompt_decision(result: "RunResult") -> str:
        print(f"\n[UniStack] HITL pause on {result.activity_id} after '{result.node}':")
        print(f"  {result.message}")
        answer = input("  Approve? [y/N] ").strip().lower()
        return "approved" if answer in ("y", "yes") else "rejected"


__all__ = ["UniStack", "RunResult", "UniStackError"]
