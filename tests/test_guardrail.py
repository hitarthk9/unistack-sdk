"""
Integration tests for UniStack guardrails + HITL (static-breakpoint model).
Requires MongoDB on localhost:27017 (uses isolated "unistack_test" database).
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

from unistack import UniStack
from unistack.config import MONGO_URI

TEST_DB = "unistack_test"
POLICY = "Output must not contain the word 'forbidden'."
EVAL_TARGET = "unistack._guardrail.evaluate_guardrail"


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
    db.hitl_queue.drop()


def _sdk(workflow: str) -> UniStack:
    return UniStack.init(
        workflow=workflow,
        db_name=TEST_DB,
        hitl_poll_interval=0.1,
    )


def _resolve(activity_id: str, decision: str, times: int = 1, after: float = 0.15):
    """
    Simulate a human resolving the HITL queue entry `times` times (once per pause).
    Runs in a daemon thread with its own MongoDB client so it survives fixture teardown.
    """
    def _worker():
        client = MongoClient(MONGO_URI)
        db = client[TEST_DB]
        try:
            for _ in range(times):
                deadline = time.time() + 5
                while time.time() < deadline:
                    if db.hitl_queue.find_one({"activity_id": activity_id, "status": "pending"}):
                        break
                    time.sleep(0.05)
                time.sleep(after)
                db.hitl_queue.update_one(
                    {"activity_id": activity_id, "status": "pending"},
                    {"$set": {
                        "status": decision,
                        "resolved_by": "test-runner",
                        "resolved_at": datetime.now(tz=timezone.utc),
                    }},
                )
                time.sleep(0.25)  # let run() process and reach the next pause
        finally:
            client.close()

    threading.Thread(target=_worker, daemon=True).start()


# ── Test 1: guard clean pass (no human) ───────────────────────────────────────

def test_guard_clean_pass(clean_db):
    sdk = _sdk("grd-clean")

    class S(TypedDict):
        value: str

    def guarded(state):
        return {"value": "safe output"}

    builder = StateGraph(S)
    builder.add_node("guarded", guarded)
    builder.add_edge(START, "guarded")
    builder.add_edge("guarded", END)
    graph = sdk.compile(builder, guards={"guarded": POLICY})

    with patch(EVAL_TARGET, return_value={"passed": True, "reason": "ok"}) as mock_eval:
        result = sdk.run(graph, {"value": ""}, run_id="clean-1")

    assert result.status == "completed"
    mock_eval.assert_called_once()
    assert clean_db.hitl_queue.count_documents({}) == 0


# ── Test 2: guard breach → human reject ───────────────────────────────────────

def test_guard_breach_reject(clean_db):
    sdk = _sdk("grd-reject")
    activity_id = "grd-reject-rej-1"

    class S(TypedDict):
        value: str

    def guarded(state):
        return {"value": "forbidden content here"}

    builder = StateGraph(S)
    builder.add_node("guarded", guarded)
    builder.add_edge(START, "guarded")
    builder.add_edge("guarded", END)
    graph = sdk.compile(builder, guards={"guarded": POLICY})

    _resolve(activity_id, "rejected")

    with patch(EVAL_TARGET, return_value={"passed": False, "reason": "contains forbidden"}) as mock_eval:
        result = sdk.run(graph, {"value": ""}, run_id="rej-1")

    assert result.status == "hitl_rejected"
    mock_eval.assert_called_once()
    doc = clean_db.hitl_queue.find_one({"activity_id": activity_id})
    assert doc is not None
    assert doc["status"] == "rejected"
    assert doc["resolved_at"] is not None


# ── Test 3: guard breach → approve → downstream runs ──────────────────────────

def test_guard_breach_approve_downstream_runs(clean_db):
    sdk = _sdk("grd-approve")
    activity_id = "grd-approve-apr-1"
    downstream_ran = []

    class S(TypedDict):
        value: str
        done: bool

    def guarded(state):
        return {"value": "borderline", "done": False}

    def downstream(state):
        downstream_ran.append(True)
        return {"done": True}

    builder = StateGraph(S)
    builder.add_node("guarded", guarded)
    builder.add_node("downstream", downstream)
    builder.add_edge(START, "guarded")
    builder.add_edge("guarded", "downstream")
    builder.add_edge("downstream", END)
    graph = sdk.compile(builder, guards={"guarded": POLICY})

    _resolve(activity_id, "approved")

    with patch(EVAL_TARGET, return_value={"passed": False, "reason": "borderline"}) as mock_eval:
        result = sdk.run(graph, {"value": "", "done": False}, run_id="apr-1")

    assert result.status == "completed"
    assert mock_eval.call_count == 1, (
        f"evaluate called {mock_eval.call_count}x — expected 1 (resume moves forward, "
        f"the guarded node is not re-run)."
    )
    assert downstream_ran, "Downstream node did not execute after approval"


# ── Test 4a: review node → approve ────────────────────────────────────────────

def test_review_approve(clean_db):
    sdk = _sdk("rev-ap")

    class S(TypedDict):
        value: str

    def work(state):
        return {"value": "output"}

    builder = StateGraph(S)
    builder.add_node("work", work)
    builder.add_edge(START, "work")
    builder.add_edge("work", END)
    graph = sdk.compile(builder, reviews=["work"])

    _resolve("rev-ap-go", "approved")
    result = sdk.run(graph, {"value": ""}, run_id="go")
    assert result.status == "completed"


# ── Test 4b: review node → reject ─────────────────────────────────────────────

def test_review_reject(clean_db):
    sdk = _sdk("rev-rj")

    class S(TypedDict):
        value: str

    def work(state):
        return {"value": "output"}

    builder = StateGraph(S)
    builder.add_node("work", work)
    builder.add_edge(START, "work")
    builder.add_edge("work", END)
    graph = sdk.compile(builder, reviews=["work"])

    _resolve("rev-rj-no", "rejected")
    result = sdk.run(graph, {"value": ""}, run_id="no")
    assert result.status == "hitl_rejected"


# ── Test 5: keyword-scan fallback (no API key) ────────────────────────────────

def test_guardrail_keyword_fallback_breach():
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


# ── Test 6: two sequential guards, both breach, both approved ─────────────────

def test_two_sequential_guards_both_approved(clean_db):
    sdk = _sdk("grd-two")
    activity_id = "grd-two-two-1"

    class S(TypedDict):
        a: str
        b: str

    call_log = []

    def mock_eval(policy, output, context=None):
        call_log.append(output)
        return {"passed": False, "reason": "test breach"}

    def node_a(state):
        return {"a": "output-a", "b": ""}

    def node_b(state):
        return {"b": "output-b"}

    builder = StateGraph(S)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", END)
    graph = sdk.compile(builder, guards={"node_a": "Policy A", "node_b": "Policy B"})

    _resolve(activity_id, "approved", times=2)

    with patch(EVAL_TARGET, side_effect=mock_eval):
        result = sdk.run(graph, {"a": "", "b": ""}, run_id="two-1")

    assert result.status == "completed"
    assert len(call_log) == 2, (
        f"evaluate called {len(call_log)}x — expected exactly 2 (once per node)."
    )
