"""
UniStack SDK — guardrails + human-in-the-loop for existing LangGraph agents.

The SDK latches onto an already-built LangGraph `StateGraph` without touching the
author's nodes. It compiles the graph with static breakpoints (`interrupt_after`)
after each guarded / reviewed node, then owns the run loop:

  - guard   → after the node runs, an LLM-as-judge evaluates its output against a
              policy. Passed → resume silently. Breached → open a HITL pause.
  - review  → unconditional human sign-off after the node, before continuing.

Usage::

    from unistack import UniStack
    from my_app.graph import builder          # existing, untouched StateGraph

    sdk   = UniStack.init("mongodb://localhost:27017", workflow="content",
                          context="Brand voice: professional, no unverified claims.")
    graph = sdk.compile(builder,
                        guards={"generate": "No unverified medical/financial claims."},
                        reviews=["publish"])
    result = sdk.run(graph, {"topic": "..."}, run_id="2026-07-09")
    # result.status → "completed" | "hitl_rejected" | "failed"
"""

import json
import time
from datetime import datetime, timezone

from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient

from unistack.config import API_URL, MONGO_URI


class RunResult:
    def __init__(self, activity_id: str, state: dict, status: str = "completed"):
        self.activity_id = activity_id
        self.state = state or {}
        self.status = status  # "completed" | "hitl_rejected" | "failed"


class UniStack:
    """Guardrail + HITL orchestration for LangGraph graphs."""

    def __init__(
        self,
        workflow: str,
        db_name: str = "unistack",
        hitl_poll_interval: float = 2.0,
        context: str | None = None,
        context_file: str | None = None,
    ):
        self._workflow = workflow
        self._db_name = db_name
        self._hitl_poll_interval = hitl_poll_interval
        self._client = MongoClient(MONGO_URI)
        self._db = self._client[db_name]
        self.checkpointer = MongoDBSaver(self._client, db_name=db_name)
        self._guardrail_context = self._resolve_context(context, context_file)

        self._guards: dict[str, str] = {}   # node -> policy text
        self._reviews: set[str] = set()     # nodes needing unconditional sign-off

    @classmethod
    def init(cls, *args, **kwargs) -> "UniStack":
        return cls(*args, **kwargs)

    # ── Latch-on ──────────────────────────────────────────────────────────────

    def compile(self, builder, guards: dict | None = None, reviews: list | None = None):
        """
        Compile an EXISTING StateGraph builder with guardrail + HITL breakpoints.

        guards  = {node_name: policy_text}   # LLM-judged after the node runs
        reviews = [node_name, ...]           # unconditional human sign-off after the node

        Returns a compiled graph. The author's nodes are untouched — UniStack only
        adds static breakpoints (`interrupt_after`) and drives the pauses in run().
        """
        self._guards = dict(guards or {})
        self._reviews = set(reviews or [])
        stops = list(self._guards) + [n for n in self._reviews if n not in self._guards]
        return builder.compile(checkpointer=self.checkpointer, interrupt_after=stops)

    def run(self, graph, initial_state: dict, run_id: str) -> RunResult:
        """
        Execute a compiled graph as a UniStack activity.

        Blocking. Streams the graph; each time a static breakpoint fires it gates
        the node that just ran (guard evaluation or unconditional review) and
        resumes on approval. Halts and abandons the graph on rejection.
        """
        activity_id = f"{self._workflow}-{run_id}"
        config = {
            "configurable": {"thread_id": activity_id},
            "run_name":     activity_id,
            "metadata":     {"activity_id": activity_id, "workflow": self._workflow},
            "tags":         [activity_id, self._workflow],
        }

        # Clear any stale checkpoint and hitl_queue from a previous run.
        try:
            self.checkpointer.delete_thread(activity_id)
        except Exception:
            pass
        self._db.hitl_queue.delete_many({"activity_id": activity_id})

        input_val = initial_state
        status = "completed"

        try:
            while True:
                node, output, interrupted = self._stream_segment(graph, input_val, config)
                if not interrupted:
                    break                                # ran to END, nothing to gate
                if self._gate(node, output, activity_id) == "rejected":
                    status = "hitl_rejected"
                    break
                if not graph.get_state(config).next:
                    break                                # gated node was terminal — done
                input_val = None                         # resume from checkpoint
        except Exception:
            status = "failed"
            raise
        finally:
            final_state = dict(graph.get_state(config).values)
            try:
                self.checkpointer.delete_thread(activity_id)
            except Exception:
                pass

        return RunResult(activity_id, final_state, status)

    # ── Gate ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _stream_segment(graph, input_val, config):
        """
        Run the graph until the next breakpoint or END. Returns
        (last_node, last_output, interrupted) — the node the breakpoint fired
        after and its output, plus whether a breakpoint actually fired.
        """
        last_node, last_output, interrupted = None, None, False
        for chunk in graph.stream(input_val, config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                interrupted = True
                continue
            for node, output in chunk.items():
                last_node, last_output = node, output
        return last_node, last_output, interrupted

    def _gate(self, node: str | None, output, activity_id: str) -> str:
        """
        Returns "approved" (continue) or "rejected" (stop).
        A guard that passes auto-approves with no human involved.
        """
        if node in self._guards:
            result = self.evaluate(self._guards[node], json.dumps(output, default=str))
            if result["passed"]:
                print(f"\n[UniStack] guard[{node}] passed — continuing.")
                return "approved"
            print(f"\n[UniStack] guard[{node}] breached: {result['reason']}")
            message = f"Guardrail breach after '{node}': {result['reason']}"
        elif node in self._reviews:
            message = f"Human sign-off required after '{node}'."
        else:
            return "approved"                            # a breakpoint we don't own

        self._write_hitl_entry(activity_id, message)
        return self._poll_for_decision(activity_id)

    # ── Guardrail evaluation ──────────────────────────────────────────────────

    def evaluate(self, policy: str, output: str) -> dict:
        """
        Evaluate output against a policy using Claude Haiku (LLM-as-judge).
        Returns {"passed": bool, "reason": str}.
        Falls back to a keyword scan when ANTHROPIC_API_KEY is not set.
        """
        from unistack._guardrail import evaluate_guardrail
        return evaluate_guardrail(policy, output, self._guardrail_context)

    # ── HITL queue ────────────────────────────────────────────────────────────

    def _write_hitl_entry(self, activity_id: str, message: str) -> None:
        """Upsert a minimal hitl_queue document. Called by run() on every pause."""
        self._db.hitl_queue.replace_one(
            {"activity_id": activity_id},
            {
                "activity_id": activity_id,
                "status":      "pending",
                "message":     message,
                "created_at":  datetime.now(tz=timezone.utc),
                "resolved_at": None,
                "resolved_by": None,
            },
            upsert=True,
        )

    def _poll_for_decision(self, activity_id: str) -> str:
        """Block until the hitl_queue entry is resolved. Returns 'approved' or 'rejected'."""
        print(f"\n[UniStack] Waiting for human decision on {activity_id}")
        print(f"  curl -X POST {API_URL}/hitl/{activity_id}/resolve \\")
        print(f'       -H "Content-Type: application/json" \\')
        print(f'       -d \'{{"decision":"approve","resolved_by":"you@company.com"}}\'')
        while True:
            doc = self._db.hitl_queue.find_one(
                {"activity_id": activity_id, "status": {"$ne": "pending"}}
            )
            if doc:
                print(f"\n[UniStack] Decision: {doc['status']}")
                return doc["status"]
            time.sleep(self._hitl_poll_interval)

    # ── Context loading ───────────────────────────────────────────────────────

    @staticmethod
    def _resolve_context(context: str | None, context_file: str | None) -> str | None:
        """Load guardrail business context from an inline string or a YAML file."""
        if context:
            return context
        if context_file:
            try:
                import yaml
            except ImportError:
                raise ImportError("pyyaml is required for context_file: pip install pyyaml")
            with open(context_file) as f:
                data = yaml.safe_load(f)
            return data.get("guardrail_context")
        return None


__all__ = ["UniStack", "RunResult"]
