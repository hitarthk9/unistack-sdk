"""
Integration tests for UniStack guardrails + durable HITL (start / resume model).
Requires MongoDB on localhost:27017 (uses isolated "unistack_test" database).

Tracing is off in tests (no LangSmith key) so the hitl_pause span open/close are no-ops
and nothing hits the network. start()/resume() are synchronous — no threads or polling.
"""

import re
from typing import TypedDict
from unittest.mock import patch

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pymongo import MongoClient

import pytest

from unistack import UniStack

MONGO_URI = "mongodb://localhost:27017"
TEST_DB = "unistack_test"
EVAL_TARGET = "unistack._guardrail.evaluate_guardrail"


@pytest.fixture(autouse=True)
def clean_db():
    client = MongoClient(MONGO_URI)
    db = client[TEST_DB]
    _wipe(db)
    yield db
    _wipe(db)
    client.close()


def _wipe(db):
    for c in ("checkpoints", "checkpoint_writes"):
        db[c].drop()


def _sdk(workflow: str) -> UniStack:
    return UniStack.init(workflow=workflow, mongo_uri=MONGO_URI, db_name=TEST_DB)


class S(TypedDict):
    a: str
    b: str


def _one_node_graph(name: str):
    def node(state):
        return {"a": "output", "b": ""}
    b = StateGraph(S)
    b.add_node(name, node)
    b.add_edge(START, name)
    b.add_edge(name, END)
    return b


def _two_node_graph():
    def node_a(state):
        return {"a": "out-a", "b": ""}

    def node_b(state):
        return {"b": "out-b"}
    b = StateGraph(S)
    b.add_node("node_a", node_a)
    b.add_node("node_b", node_b)
    b.add_edge(START, "node_a")
    b.add_edge("node_a", "node_b")
    b.add_edge("node_b", END)
    return b


def _passes(*a, **k):
    return {"passed": True, "reason": "ok"}


def _breaches(*a, **k):
    return {"passed": False, "reason": "test breach"}


# ── Guards ─────────────────────────────────────────────────────────────────────

def test_guard_clean_pass_completes_without_pause():
    sdk = _sdk("grd-clean")
    graph = sdk.compile(_one_node_graph("gen"), guards={"gen": "policy"})
    with patch(EVAL_TARGET, side_effect=_passes) as mock:
        r = sdk.start(graph, {"a": "", "b": ""})
    assert r.status == "completed"          # a passing guard never pauses
    assert mock.call_count == 1


def test_guard_breach_pauses_then_reject_halts():
    sdk = _sdk("grd-rej")
    graph = sdk.compile(_one_node_graph("gen"), guards={"gen": "policy"})
    with patch(EVAL_TARGET, side_effect=_breaches):
        r = sdk.start(graph, {"a": "", "b": ""})
    assert r.status == "paused" and r.node == "gen"
    r2 = sdk.resume(graph, r.activity_id, "rejected", resolved_by="t")
    assert r2.status == "hitl_rejected"


def test_guard_breach_approve_runs_downstream():
    sdk = _sdk("grd-apr")
    graph = sdk.compile(_two_node_graph(), guards={"node_a": "policy"})
    with patch(EVAL_TARGET, side_effect=_breaches) as mock:
        r = sdk.start(graph, {"a": "", "b": ""})
        assert r.status == "paused" and r.node == "node_a"
        r2 = sdk.resume(graph, r.activity_id, "approved")
    assert r2.status == "completed"
    assert r2.state["b"] == "out-b"          # downstream node_b ran after approval
    assert mock.call_count == 1              # guarded node judged exactly once


# ── Reviews ────────────────────────────────────────────────────────────────────

def test_review_pause_then_approve_completes():
    sdk = _sdk("rev-apr")
    graph = sdk.compile(_one_node_graph("work"), reviews=["work"])
    r = sdk.start(graph, {"a": "", "b": ""})
    assert r.status == "paused" and r.node == "work"
    r2 = sdk.resume(graph, r.activity_id, "approved")
    assert r2.status == "completed"


def test_review_reject_halts():
    sdk = _sdk("rev-rej")
    graph = sdk.compile(_one_node_graph("work"), reviews=["work"])
    r = sdk.start(graph, {"a": "", "b": ""})
    r2 = sdk.resume(graph, r.activity_id, "rejected")
    assert r2.status == "hitl_rejected"


# ── Multiple HITL in one activity ───────────────────────────────────────────────

def test_two_sequential_guards_both_breach_both_approved():
    sdk = _sdk("grd-two")
    graph = sdk.compile(_two_node_graph(), guards={"node_a": "A", "node_b": "B"})
    with patch(EVAL_TARGET, side_effect=_breaches) as mock:
        r = sdk.start(graph, {"a": "", "b": ""})
        assert r.status == "paused" and r.node == "node_a"
        r = sdk.resume(graph, r.activity_id, "approved")
        assert r.status == "paused" and r.node == "node_b"     # advanced to the next pause
        r = sdk.resume(graph, r.activity_id, "approved")
    assert r.status == "completed"
    assert mock.call_count == 2              # once per guarded node


# ── Durability: resume survives a process restart ───────────────────────────────

def test_durability_resume_in_fresh_instance():
    sdk_a = _sdk("dur")
    graph_a = sdk_a.compile(_one_node_graph("work"), reviews=["work"])
    r = sdk_a.start(graph_a, {"a": "", "b": ""})
    assert r.status == "paused"

    # brand-new SDK + compile = a different process after a restart; state loads from Mongo
    sdk_b = _sdk("dur")
    graph_b = sdk_b.compile(_one_node_graph("work"), reviews=["work"])
    r2 = sdk_b.resume(graph_b, r.activity_id, "approved", resolved_by="other-process")
    assert r2.status == "completed"


def test_double_resume_is_idempotent():
    sdk = _sdk("idem")
    graph = sdk.compile(_one_node_graph("work"), reviews=["work"])
    r = sdk.start(graph, {"a": "", "b": ""})
    assert sdk.resume(graph, r.activity_id, "approved").status == "completed"
    # a second resolve on an already-finished thread is a no-op, not a re-run
    assert sdk.resume(graph, r.activity_id, "approved").status == "completed"


# ── run() convenience with an injected (non-interactive) decision provider ──────

def test_run_convenience_drives_pauses_and_ids_are_unique():
    sdk = _sdk("auto")
    graph = sdk.compile(_two_node_graph(), reviews=["node_a", "node_b"])
    r1 = sdk.run(graph, {"a": "", "b": ""}, decide=lambda res: "approved")
    r2 = sdk.run(graph, {"a": "", "b": ""}, decide=lambda res: "approved")
    assert r1.status == "completed" and r2.status == "completed"
    assert re.fullmatch(r"auto-\d{8}T\d{12}", r1.activity_id)   # unique timestamped id
    assert r1.activity_id != r2.activity_id


# ── Guardrail keyword fallback (no API key) ─────────────────────────────────────

def test_guardrail_keyword_fallback_breach():
    from unistack._guardrail import evaluate_guardrail
    result = evaluate_guardrail("No illegal activity", "This order is sanctioned and illegal")
    assert result["passed"] is False
    assert "sanctioned" in result["reason"] or "illegal" in result["reason"]
    clean = evaluate_guardrail("No illegal activity", "Standard retail order, low risk")
    assert clean["passed"] is True
