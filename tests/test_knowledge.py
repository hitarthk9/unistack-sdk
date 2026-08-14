"""
Knowledge-base-backed guards.

Two halves:
  * pure unit tests over `unistack._knowledge` — no IO, no Mongo, no network;
  * integration tests proving what the feature is FOR: the judge is handed every rule, and the
    rule it cites reaches the human approving the pause.

The Mongo-backed half uses the same isolated database as test_guardrail.py.
"""

import logging
from typing import TypedDict
from unittest.mock import patch

import pytest
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pymongo import MongoClient

from unistack import UniStack, _knowledge
from unistack._knowledge import KnowledgeBaseError

MONGO_URI = "mongodb://localhost:27017"
TEST_DB = "unistack_test"
EVAL_TARGET = "unistack._guardrail.evaluate_guardrail"

KB = {
    "knowledge_base": "brand-policy",
    "version": 3,
    "rules": [
        {"id": "BP-001", "rule": "No unverified medical claims.",
         "examples": {"violating": "boosts your energy", "compliant": "all-day comfort"}},
        {"id": "BP-002", "rule": "No unverified financial claims."},
    ],
}


# ── validate ───────────────────────────────────────────────────────────────────

def test_validate_accepts_a_well_formed_base():
    assert _knowledge.validate(KB) is KB


@pytest.mark.parametrize("kb, expected", [
    ("just a string",                                    "must be a mapping"),
    ({"rules": [{"id": "A", "rule": "r"}]},              "non-empty 'knowledge_base' name"),
    ({"knowledge_base": "k"},                            "has no 'rules'"),
    ({"knowledge_base": "k", "rules": []},               "has no 'rules'"),
    ({"knowledge_base": "k", "rules": [{"rule": "r"}]},  "needs a non-empty 'id'"),
    ({"knowledge_base": "k", "rules": [{"id": "A"}]},    "needs a non-empty 'rule'"),
])
def test_validate_rejects_malformed_bases(kb, expected):
    # Loud at load time, never a judge prompt with silently missing rules.
    with pytest.raises(KnowledgeBaseError, match=expected):
        _knowledge.validate(kb)


def test_validate_rejects_duplicate_ids():
    # Duplicates would make a citation ambiguous, which defeats the point of having ids.
    kb = {"knowledge_base": "k", "rules": [{"id": "A", "rule": "one"}, {"id": "A", "rule": "two"}]}
    with pytest.raises(KnowledgeBaseError, match="duplicate rule id 'A'"):
        _knowledge.validate(kb)


# ── render ─────────────────────────────────────────────────────────────────────

def test_render_includes_every_rule_its_id_and_its_examples():
    out = _knowledge.render(KB)
    assert "[BP-001] No unverified medical claims." in out
    assert "[BP-002] No unverified financial claims." in out     # every rule, always
    assert "violating: boosts your energy" in out
    assert "compliant: all-day comfort" in out
    assert "brand-policy (version 3)" in out                     # which version judged this


def test_render_emits_every_example_label_not_just_a_fixed_pair():
    """A dropped example is an author editing a file that has no effect — silently."""
    kb = {"knowledge_base": "k", "rules": [{"id": "A", "rule": "r", "examples": {
        "violating": "bad", "compliant": "good", "compliant_edge_case": "also good"}}]}
    out = _knowledge.render(kb)
    assert "compliant_edge_case: also good" in out
    # ...in the author's order, so the block stays byte-stable and cacheable.
    assert out.index("violating") < out.index("compliant:") < out.index("compliant_edge_case")


def test_render_is_byte_stable_across_calls():
    # The whole prompt-caching argument rests on this: a non-deterministic render would
    # invalidate the cache on every call and silently cost ~10x.
    assert _knowledge.render(KB) == _knowledge.render(KB)


# ── resolve ────────────────────────────────────────────────────────────────────

def test_resolve_passes_a_string_guard_through_untouched():
    assert _knowledge.resolve("plain policy", {"brand-policy": KB}) == ("plain policy", None)


def test_resolve_accepts_a_mapping_with_both_halves():
    guard = {"policy": "node-specific", "knowledge_base": "brand-policy"}
    assert _knowledge.resolve(guard, {"brand-policy": KB}) == ("node-specific", "brand-policy")


def test_resolve_accepts_a_mapping_with_policy_only():
    assert _knowledge.resolve({"policy": "only prose"}, {}) == ("only prose", None)


def test_resolve_rejects_an_unknown_knowledge_base():
    # The dangerous case: naming a base that was never loaded would judge against an empty
    # policy and pass everything. Refuse instead — and name what IS loaded.
    with pytest.raises(KnowledgeBaseError, match=r"unknown knowledge base 'nope'.*loaded: brand-policy"):
        _knowledge.resolve({"knowledge_base": "nope"}, {"brand-policy": KB})


def test_resolve_rejects_an_empty_mapping():
    with pytest.raises(KnowledgeBaseError, match="needs 'policy', 'knowledge_base', or both"):
        _knowledge.resolve({}, {})


# ── policy_block ───────────────────────────────────────────────────────────────

def test_policy_block_puts_node_prose_ahead_of_the_shared_rules():
    block = _knowledge.policy_block("node prose", KB)
    assert block.index("node prose") < block.index("[BP-001]")


def test_policy_block_without_a_base_is_just_the_prose():
    assert _knowledge.policy_block("only prose", None) == "only prose"


# ── size warning ───────────────────────────────────────────────────────────────

def test_check_size_warns_only_once_over_the_ceiling(caplog):
    with caplog.at_level(logging.WARNING, logger="unistack"):
        _knowledge.check_size("small", "x" * 400)
        assert caplog.records == []                              # well under: silent

        _knowledge.check_size("huge", "x" * (_knowledge.MAX_RULE_TOKENS * 4 + 4))
    assert len(caplog.records) == 1
    # The message must name the real limit — attention, not context — or the reader "fixes" it
    # by reaching for a bigger model, which does nothing.
    assert "attention" in caplog.records[0].getMessage()


# ── integration: the guard, end to end ─────────────────────────────────────────

class S(TypedDict):
    a: str


@pytest.fixture(autouse=True)
def clean_db():
    client = MongoClient(MONGO_URI)
    db = client[TEST_DB]
    _wipe(db)
    yield db
    _wipe(db)
    client.close()


def _wipe(db):
    for c in ("checkpoints", "checkpoint_writes", "hitl_resolutions", "activities"):
        db[c].drop()


def _graph():
    def gen(state):
        return {"a": "boosts your energy and focus all day long"}
    b = StateGraph(S)
    b.add_node("gen", gen)
    b.add_edge(START, "gen")
    b.add_edge("gen", END)
    return b


def _sdk(workflow: str, **kw) -> UniStack:
    return UniStack.init(workflow=workflow, mongo_uri=MONGO_URI, db_name=TEST_DB, **kw)


def test_compile_refuses_a_guard_naming_an_unloaded_base():
    # Compile time, not mid-run: the failure lands on the person deploying, not on an activity.
    sdk = _sdk("kb-unknown", knowledge_bases={"brand-policy": KB})
    with pytest.raises(KnowledgeBaseError, match="unknown knowledge base 'other'"):
        sdk.compile(_graph(), guards={"gen": {"knowledge_base": "other"}})


def test_init_refuses_a_malformed_base():
    with pytest.raises(KnowledgeBaseError, match="has no 'rules'"):
        _sdk("kb-bad", knowledge_bases={"brand-policy": {"knowledge_base": "brand-policy"}})


def test_judge_receives_every_rule_of_the_named_base():
    sdk = _sdk("kb-policy", knowledge_bases={"brand-policy": KB})
    graph = sdk.compile(_graph(), guards={"gen": {"policy": "Be professional.",
                                                  "knowledge_base": "brand-policy"}})
    with patch(EVAL_TARGET, return_value={"passed": True, "reason": "ok", "rule_ids": []}) as m:
        sdk.start(graph, {"a": ""})

    policy = m.call_args.args[0]
    assert "Be professional." in policy          # node-specific prose survives
    assert "[BP-001]" in policy and "[BP-002]" in policy   # ...and EVERY rule is sent


def test_breach_message_cites_the_rule_the_judge_named():
    """The payoff: the human approving the pause is told which rule fired."""
    sdk = _sdk("kb-cite", knowledge_bases={"brand-policy": KB})
    graph = sdk.compile(_graph(), guards={"gen": {"knowledge_base": "brand-policy"}})
    verdict = {"passed": False, "reason": "claims an energy boost with no source",
               "rule_ids": ["BP-001"]}
    with patch(EVAL_TARGET, return_value=verdict):
        r = sdk.start(graph, {"a": ""})

    assert r.status == "paused"
    assert "[BP-001]" in r.message
    # ...and it is durable, not just in the HTTP response — this is what unistack-api lists.
    doc = MongoClient(MONGO_URI)[TEST_DB]["hitl_resolutions"].find_one({"activity_id": r.activity_id})
    assert "[BP-001]" in doc["message"]


def test_breach_message_without_citations_is_unchanged():
    # Backward compatibility for a plain string guard: no ids, no empty "[]" noise.
    sdk = _sdk("kb-plain")
    graph = sdk.compile(_graph(), guards={"gen": "No unverified claims."})
    with patch(EVAL_TARGET, return_value={"passed": False, "reason": "nope", "rule_ids": []}):
        r = sdk.start(graph, {"a": ""})
    assert "[" not in r.message.split("nope")[0]
    assert "nope" in r.message
