"""
Guardrail tests for the LangGraph adapter.

Scenarios covered:
  1. Clean pass        — output is fine, evaluate_guardrail called once, graph completes
  2. Breach → reject   — human rejects, result.status = "guardrail_breached"
  3. Breach → approve  — human approves, result.status = "completed"
                         evaluate_guardrail called EXACTLY ONCE (the double-execution fix)
  4. Keyword fallback  — without an API key the keyword scan catches obvious violations

All tests use a real MongoDB (localhost:27017) with an isolated "unistack_test" database
that is wiped before and after each test. The SDK's poll interval is set to 0.1s so tests
complete quickly.
"""

import threading
import time
from datetime import datetime, timezone
from typing import TypedDict
from unittest.mock import patch

import pytest
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pymongo import MongoClient

from unistack.adapters.langgraph import UniStack

MONGO_URI = "mongodb://localhost:27017"
TEST_DB = "unistack_test"
POLICY = "Output must not contain the word 'forbidden'."


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_db():
    """Wipe the test database before and after every test."""
    client = MongoClient(MONGO_URI)
    db = client[TEST_DB]
    _wipe(db)
    yield db
    _wipe(db)
    client.close()


def _wipe(db):
    for col in ["spans", "hitl_queue", "checkpoints", "checkpoint_writes"]:
        db[col].drop()


def _sdk(workflow: str) -> UniStack:
    return UniStack.init(
        MONGO_URI,
        workflow=workflow,
        db_name=TEST_DB,
        hitl_poll_interval=0.1,
    )


def _graph(sdk: UniStack, node_fn):
    """Compile a minimal single-node LangGraph."""
    class S(TypedDict):
        value: str

    builder = StateGraph(S)
    builder.add_node("guarded", node_fn)
    builder.add_edge(START, "guarded")
    builder.add_edge("guarded", END)
    return builder.compile(checkpointer=sdk.checkpointer)


def _resolve(db, activity_id: str, decision: str, after: float = 0.15):
    """
    Simulate a human resolving the HITL queue entry.
    Polls until the pending entry appears, then writes the decision.
    Runs in a daemon thread so the test's main thread can block on sdk.run().
    """
    def _worker():
        deadline = time.time() + 5
        while time.time() < deadline:
            if db.hitl_queue.find_one({"activity_id": activity_id, "status": "pending"}):
                break
            time.sleep(0.05)
        time.sleep(after)
        db.hitl_queue.update_one(
            {"activity_id": activity_id},
            {"$set": {
                "status": decision,
                "resolved_by": "test-runner",
                "resolved_at": datetime.now(tz=timezone.utc),
            }},
        )

    threading.Thread(target=_worker, daemon=True).start()


# ── Test 1: clean pass ────────────────────────────────────────────────────────

def test_guardrail_clean_pass():
    """
    When the output is clean, evaluate_guardrail is called once and the graph
    completes without any interrupt.
    """
    sdk = _sdk("grd-clean")

    @sdk.node
    @sdk.guardrail(POLICY)
    def guarded(state):
        return {"value": "safe output"}

    graph = _graph(sdk, guarded)

    with patch("unistack.adapters.langgraph.evaluate_guardrail",
               return_value={"passed": True, "reason": "ok"}) as mock_eval:
        result = sdk.run(graph, {"value": ""}, run_id="clean-1")

    assert result.status == "completed"
    mock_eval.assert_called_once()


# ── Test 2: breach → human reject ─────────────────────────────────────────────

def test_guardrail_breach_reject(clean_db):
    """
    When a breach is detected and the human rejects, run() breaks out of the
    loop directly (without resuming the graph), so result.status = "hitl_rejected".
    evaluate_guardrail is called once.
    """
    sdk = _sdk("grd-reject")
    activity_id = "grd-reject-rej-1"

    @sdk.node
    @sdk.guardrail(POLICY)
    def guarded(state):
        return {"value": "forbidden content here"}

    graph = _graph(sdk, guarded)
    _resolve(clean_db, activity_id, "rejected")

    with patch("unistack.adapters.langgraph.evaluate_guardrail",
               return_value={"passed": False, "reason": "contains forbidden"}) as mock_eval:
        result = sdk.run(graph, {"value": ""}, run_id="rej-1")

    # run() handles rejection directly without resuming the graph, so the status
    # is "hitl_rejected" (same as a rejected regular HITL), not "guardrail_breached".
    assert result.status == "hitl_rejected"
    mock_eval.assert_called_once()


# ── Test 3: breach → human approve (the double-evaluation fix) ────────────────

def test_guardrail_breach_approve_no_double_eval(clean_db):
    """
    When a breach is detected and the human approves, the graph resumes and
    completes successfully.

    The critical assertion: evaluate_guardrail must be called EXACTLY ONCE —
    not twice. LangGraph re-runs the node function on resume (unavoidable), but
    the SDK's _guardrail_approved flag must prevent re-evaluation of the same
    output the human already reviewed.
    """
    sdk = _sdk("grd-approve")
    activity_id = "grd-approve-apr-1"

    @sdk.node
    @sdk.guardrail(POLICY)
    def guarded(state):
        return {"value": "questionable content"}

    graph = _graph(sdk, guarded)
    _resolve(clean_db, activity_id, "approved")

    with patch("unistack.adapters.langgraph.evaluate_guardrail",
               return_value={"passed": False, "reason": "borderline"}) as mock_eval:
        result = sdk.run(graph, {"value": ""}, run_id="apr-1")

    assert result.status == "completed", (
        f"Expected 'completed' after human approval, got '{result.status}'"
    )
    assert mock_eval.call_count == 1, (
        f"evaluate_guardrail was called {mock_eval.call_count} times — "
        f"expected 1. The double-evaluation fix is broken."
    )


# ── Test 4: breach → approve → following nodes still run ──────────────────────

def test_guardrail_approve_downstream_nodes_execute(clean_db):
    """
    After a guardrail breach is approved, nodes downstream of the guarded node
    must still execute normally.
    """
    sdk = _sdk("grd-downstream")
    activity_id = "grd-downstream-ds-1"
    downstream_ran = []

    class S(TypedDict):
        value: str
        done: bool

    @sdk.node
    @sdk.guardrail(POLICY)
    def guarded(state):
        return {"value": "borderline", "done": False}

    @sdk.node
    def downstream(state):
        downstream_ran.append(True)
        return {"done": True}

    builder = StateGraph(S)
    builder.add_node("guarded", guarded)
    builder.add_node("downstream", downstream)
    builder.add_edge(START, "guarded")
    builder.add_edge("guarded", "downstream")
    builder.add_edge("downstream", END)
    graph = builder.compile(checkpointer=sdk.checkpointer)

    _resolve(clean_db, activity_id, "approved")

    with patch("unistack.adapters.langgraph.evaluate_guardrail",
               return_value={"passed": False, "reason": "borderline"}):
        result = sdk.run(graph, {"value": "", "done": False}, run_id="ds-1")

    assert result.status == "completed"
    assert downstream_ran, "Downstream node did not execute after guardrail approval"


# ── Test 5: keyword-scan fallback (no API key) ────────────────────────────────

def test_guardrail_keyword_fallback_breach(clean_db):
    """
    Without an API key, the keyword scanner catches explicit violations.
    This exercises the fallback path in evaluate_guardrail directly.
    """
    import os
    from unistack._guardrail import evaluate_guardrail

    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        result = evaluate_guardrail("No illegal activity", "This order is sanctioned and illegal")
        assert result["passed"] is False
        assert "sanctioned" in result["reason"] or "illegal" in result["reason"]

        clean = evaluate_guardrail("No illegal activity", "Standard retail order, low risk")
        assert clean["passed"] is True
    finally:
        if saved:
            os.environ["ANTHROPIC_API_KEY"] = saved


# ── Test 6: multiple sequential guardrail breaches (both approved) ─────────────

def test_two_sequential_guardrail_breaches_both_approved(clean_db):
    """
    A workflow with two guardrail-decorated nodes where both breach and both
    are approved. Each evaluate_guardrail call must happen exactly once per node.
    """
    sdk = _sdk("grd-two")

    class S(TypedDict):
        a: str
        b: str

    call_log = []

    def mock_eval(policy, output, context=None):
        call_log.append(output)
        return {"passed": False, "reason": "test breach"}

    @sdk.node
    @sdk.guardrail("Policy A")
    def node_a(state):
        return {"a": "output-a", "b": ""}

    @sdk.node
    @sdk.guardrail("Policy B")
    def node_b(state):
        return {"b": "output-b"}

    builder = StateGraph(S)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", END)
    graph = builder.compile(checkpointer=sdk.checkpointer)

    activity_id_a = "grd-two-two-1"

    # We need to approve twice — once per node breach.
    # Each approval is detected after the previous interrupt resolves.
    approved = []

    def double_approver():
        db = MongoClient(MONGO_URI)[TEST_DB]
        for _ in range(2):
            deadline = time.time() + 5
            while time.time() < deadline:
                doc = db.hitl_queue.find_one({"activity_id": activity_id_a, "status": "pending"})
                if doc:
                    break
                time.sleep(0.05)
            time.sleep(0.15)
            db.hitl_queue.update_one(
                {"activity_id": activity_id_a},
                {"$set": {
                    "status": "approved",
                    "resolved_by": "test-runner",
                    "resolved_at": datetime.now(tz=timezone.utc),
                }},
            )
            approved.append(True)
            # Give sdk.run() time to process and re-raise the next interrupt
            time.sleep(0.3)

    threading.Thread(target=double_approver, daemon=True).start()

    with patch("unistack.adapters.langgraph.evaluate_guardrail", side_effect=mock_eval):
        result = sdk.run(graph, {"a": "", "b": ""}, run_id="two-1")

    assert result.status == "completed"
    assert len(call_log) == 2, (
        f"evaluate_guardrail called {len(call_log)} times — expected exactly 2 "
        f"(once per node, not twice per node). calls: {call_log}"
    )
