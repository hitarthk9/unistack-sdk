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

from unistack import UniStack, UniStackError

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
    for c in ("checkpoints", "checkpoint_writes", "hitl_resolutions"):
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
    # unique timestamped id + 4-hex anti-collision suffix
    assert re.fullmatch(r"auto-\d{8}T\d{12}-[0-9a-f]{4}", r1.activity_id)
    assert r1.activity_id != r2.activity_id


# ── Checkpoint cleanup (Mongo working-state trimmed at terminal outcomes) ───────

def _checkpoints(db, activity_id: str) -> int:
    return db.checkpoints.count_documents({"thread_id": activity_id})


def test_checkpoints_kept_while_paused_deleted_on_completion(clean_db):
    sdk = _sdk("ck-cmp")
    graph = sdk.compile(_one_node_graph("work"), reviews=["work"])
    r = sdk.start(graph, {"a": "", "b": ""})
    assert r.status == "paused"
    assert _checkpoints(clean_db, r.activity_id) > 0        # kept while a human is needed
    r2 = sdk.resume(graph, r.activity_id, "approved")
    assert r2.status == "completed"
    assert _checkpoints(clean_db, r.activity_id) == 0       # wiped once terminal


def test_checkpoints_deleted_on_reject(clean_db):
    sdk = _sdk("ck-rej")
    graph = sdk.compile(_one_node_graph("work"), reviews=["work"])
    r = sdk.start(graph, {"a": "", "b": ""})
    sdk.resume(graph, r.activity_id, "rejected")
    assert _checkpoints(clean_db, r.activity_id) == 0


def test_checkpoints_deleted_on_clean_pass(clean_db):
    sdk = _sdk("ck-clean")
    graph = sdk.compile(_one_node_graph("gen"), guards={"gen": "policy"})
    with patch(EVAL_TARGET, side_effect=_passes):
        r = sdk.start(graph, {"a": "", "b": ""})           # passes, never pauses → straight to END
    assert r.status == "completed"
    assert _checkpoints(clean_db, r.activity_id) == 0


def test_double_resume_after_delete_is_safe(clean_db):
    sdk = _sdk("ck-idem")
    graph = sdk.compile(_one_node_graph("work"), reviews=["work"])
    r = sdk.start(graph, {"a": "", "b": ""})
    assert sdk.resume(graph, r.activity_id, "approved").status == "completed"
    assert _checkpoints(clean_db, r.activity_id) == 0
    # resolving an already-deleted thread must not throw; get_state returns an empty snapshot
    again = sdk.resume(graph, r.activity_id, "approved")
    assert again.status == "completed"


def test_messages_state_graph_completes_and_cleans_up(clean_db):
    """The ubiquitous add_messages accumulating reducer runs start→pause→resume→END fine,
    keeps its accumulated state, and leaves zero checkpoints behind."""
    from langgraph.graph import MessagesState

    def greet(state):
        return {"messages": [("ai", "hello")]}

    def finish(state):
        return {"messages": [("ai", "done")]}

    b = StateGraph(MessagesState)
    b.add_node("greet", greet)
    b.add_node("finish", finish)
    b.add_edge(START, "greet")
    b.add_edge("greet", "finish")
    b.add_edge("finish", END)

    sdk = _sdk("ck-msgs")
    graph = sdk.compile(b, reviews=["greet"])
    r = sdk.start(graph, {"messages": [("human", "hi")]})
    assert r.status == "paused"
    r2 = sdk.resume(graph, r.activity_id, "approved")
    assert r2.status == "completed"
    # accumulating reducer preserved the full conversation, and cleanup left nothing behind
    assert len(r2.state["messages"]) == 3                   # human hi + ai hello + ai done
    assert _checkpoints(clean_db, r.activity_id) == 0


# ── Guardrail keyword fallback (no API key) ─────────────────────────────────────

def test_guardrail_keyword_fallback_breach():
    from unistack._guardrail import evaluate_guardrail
    result = evaluate_guardrail("No illegal activity", "This order is sanctioned and illegal")
    assert result["passed"] is False
    assert "sanctioned" in result["reason"] or "illegal" in result["reason"]
    clean = evaluate_guardrail("No illegal activity", "Standard retail order, low risk")
    assert clean["passed"] is True


# ── Judge failure fails CLOSED (never crash, never silently pass) ───────────────

def test_judge_api_error_fails_closed():
    from unistack._guardrail import evaluate_guardrail
    with patch("anthropic.Anthropic") as anthro:
        anthro.return_value.messages.create.side_effect = RuntimeError("api down")
        with patch("langsmith.wrappers.wrap_anthropic", side_effect=lambda c, **k: c):
            result = evaluate_guardrail("policy", "output", api_key="sk-ant-fake")
    assert result["passed"] is False
    assert "judge unavailable" in result["reason"]


def test_judge_malformed_verdict_fails_closed():
    from unittest.mock import MagicMock

    from unistack._guardrail import evaluate_guardrail

    block = MagicMock()
    block.type = "tool_use"
    block.input = {"pass": True}                 # wrong key — malformed verdict
    resp = MagicMock()
    resp.content = [block]
    with patch("anthropic.Anthropic") as anthro:
        anthro.return_value.messages.create.return_value = resp
        with patch("langsmith.wrappers.wrap_anthropic", side_effect=lambda c, **k: c):
            result = evaluate_guardrail("policy", "output", api_key="sk-ant-fake")
    assert result["passed"] is False
    assert "malformed verdict" in result["reason"]


def test_sdk_pauses_when_judge_raises():
    """Even if the whole evaluator blows up, the activity pauses instead of crashing."""
    sdk = _sdk("grd-jerr")
    graph = sdk.compile(_one_node_graph("gen"), guards={"gen": "policy"})
    with patch(EVAL_TARGET, side_effect=RuntimeError("total judge failure")):
        r = sdk.start(graph, {"a": "", "b": ""})
    assert r.status == "paused" and r.node == "gen"
    assert "judge error" in r.message


def test_sdk_pauses_on_malformed_verdict_shape():
    sdk = _sdk("grd-jshape")
    graph = sdk.compile(_one_node_graph("gen"), guards={"gen": "policy"})
    with patch(EVAL_TARGET, return_value={"verdict": "fine"}):     # missing "passed"
        r = sdk.start(graph, {"a": "", "b": ""})
    assert r.status == "paused"
    assert "malformed guardrail verdict" in r.message


# ── Dynamic interrupt() is rejected loudly (was: infinite loop) ─────────────────

def test_dynamic_interrupt_raises_clear_error(clean_db):
    from langgraph.types import interrupt

    def needs_input(state):
        answer = interrupt("what now?")
        return {"a": answer, "b": ""}

    b = StateGraph(S)
    b.add_node("needs_input", needs_input)
    b.add_edge(START, "needs_input")
    b.add_edge("needs_input", END)

    sdk = _sdk("dyn")
    graph = sdk.compile(b)                       # no guards/reviews of our own
    with pytest.raises(UniStackError, match="dynamic interrupt"):
        sdk.start(graph, {"a": "", "b": ""})
    # the dead thread's checkpoints were cleaned up
    assert clean_db.checkpoints.count_documents({}) == 0


# ── Parallel fan-out: every guarded node in a super-step is judged ──────────────

class S3(TypedDict):
    a: str
    b: str
    done: str


def _fanout_graph():
    def pa(state):
        return {"a": "out-a"}

    def pb(state):
        return {"b": "out-b"}

    def join(state):
        return {"done": "yes"}

    b = StateGraph(S3)
    b.add_node("pa", pa)
    b.add_node("pb", pb)
    b.add_node("join", join)
    b.add_edge(START, "pa")
    b.add_edge(START, "pb")
    b.add_edge("pa", "join")
    b.add_edge("pb", "join")
    b.add_edge("join", END)
    return b


def test_fanout_all_guarded_nodes_judged_pass():
    sdk = _sdk("fan-pass")
    graph = sdk.compile(_fanout_graph(), guards={"pa": "A", "pb": "B"})
    with patch(EVAL_TARGET, side_effect=_passes) as mock:
        r = sdk.start(graph, {"a": "", "b": "", "done": ""})
    assert r.status == "completed"
    assert mock.call_count == 2                  # BOTH parallel nodes judged
    assert r.state["done"] == "yes"


def test_fanout_breach_in_parallel_node_pauses_then_completes():
    sdk = _sdk("fan-breach")
    graph = sdk.compile(_fanout_graph(), guards={"pa": "A", "pb": "B"})
    with patch(EVAL_TARGET, side_effect=_breaches) as mock:
        r = sdk.start(graph, {"a": "", "b": "", "done": ""})
    assert r.status == "paused"
    assert mock.call_count == 2
    assert "'pa'" in r.message and "'pb'" in r.message   # both breaches reported
    r2 = sdk.resume(graph, r.activity_id, "approved")
    assert r2.status == "completed" and r2.state["done"] == "yes"


# ── Resolution claims: the double-approve race + unknown ids ────────────────────

def test_concurrent_resolve_loses_claim_and_does_not_advance(clean_db):
    sdk = _sdk("race")
    graph = sdk.compile(_two_node_graph(), reviews=["node_a"])
    r = sdk.start(graph, {"a": "", "b": ""})
    assert r.status == "paused"
    # simulate another process winning the claim first
    claimed = clean_db.hitl_resolutions.update_one(
        {"activity_id": r.activity_id, "status": "pending"},
        {"$set": {"status": "resolved", "decision": "approved", "resolved_by": "other-proc"}})
    assert claimed.modified_count == 1
    r2 = sdk.resume(graph, r.activity_id, "approved")
    assert "already resolved" in r2.message
    assert r2.state.get("b", "") == ""           # downstream node_b did NOT run again
    assert _checkpoints(clean_db, r.activity_id) > 0   # thread untouched by the loser


def test_resolve_claims_even_without_pending_record(clean_db):
    """Crash-recovery: pending record missing → the resolve's insert IS the claim."""
    sdk = _sdk("orphan")
    graph = sdk.compile(_one_node_graph("work"), reviews=["work"])
    r = sdk.start(graph, {"a": "", "b": ""})
    clean_db.hitl_resolutions.delete_many({"activity_id": r.activity_id})
    r2 = sdk.resume(graph, r.activity_id, "approved")
    assert r2.status == "completed"


def test_resume_unknown_activity_is_not_found():
    sdk = _sdk("nf")
    graph = sdk.compile(_one_node_graph("work"), reviews=["work"])
    r = sdk.resume(graph, "nf-19990101T000000000000-dead", "approved")
    assert r.status == "not_found"


def test_resume_invalid_decision_raises():
    sdk = _sdk("baddec")
    graph = sdk.compile(_one_node_graph("work"), reviews=["work"])
    with pytest.raises(ValueError, match="decision"):
        sdk.resume(graph, "whatever", "maybe")


# ── run_id: same-microsecond starts never collide; SDK never touches os.environ ─

def test_same_microsecond_starts_get_distinct_ids():
    from datetime import datetime as real_datetime
    from unittest.mock import MagicMock

    frozen = MagicMock(wraps=real_datetime)
    frozen.now.return_value = real_datetime(2026, 7, 13, 12, 0, 0, 123456)
    sdk = _sdk("frozen")
    graph = sdk.compile(_one_node_graph("gen"))
    with patch("unistack.core.datetime", frozen):
        r1 = sdk.start(graph, {"a": "", "b": ""})
        r2 = sdk.start(graph, {"a": "", "b": ""})
    assert r1.activity_id != r2.activity_id      # hex suffix differs despite equal timestamp
    assert r1.activity_id.startswith("frozen-20260713T120000123456-")


def test_init_never_mutates_environ():
    import os
    before = {k: os.environ.get(k) for k in
              ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT")}
    UniStack.init(workflow="envtest", mongo_uri=MONGO_URI, db_name=TEST_DB,
                  langsmith_api_key="lsv2_pt_fake_key_for_test")
    after = {k: os.environ.get(k) for k in before}
    assert before == after
