"""
Span-level tests for the SDK's OpenTelemetry integration, using an in-memory exporter
(the `tracer_provider` init kwarg is the test seam — nothing hits the network).
Requires MongoDB on localhost:27017 (uses the isolated "unistack_test" database).

What they lock in:
  - one trace per leg (unistack.start / unistack.resume), grouped by session.id;
  - node + LLM spans via the callback handler, including node-internal `llm.invoke()`
    with no config (langchain-core contextvar propagation);
  - the pending doc persists the pausing leg's trace ids, and the `hitl_pause` span is
    emitted retroactively into that trace at resolve time;
  - guardrail_eval + judge GenAI spans with token usage;
  - telemetry failures never change run behaviour (fail-open).
"""

import json
from datetime import timezone
from typing import TypedDict
from unittest.mock import MagicMock, patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import format_span_id, format_trace_id
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
    for c in ("checkpoints", "checkpoint_writes", "hitl_resolutions"):
        db[c].drop()


@pytest.fixture()
def exporter():
    return InMemorySpanExporter()


@pytest.fixture()
def sdk_factory(exporter):
    def make(workflow: str, **kwargs) -> UniStack:
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        return UniStack.init(workflow=workflow, mongo_uri=MONGO_URI, db_name=TEST_DB,
                             tracer_provider=provider, **kwargs)
    return make


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


def _spans(exporter):
    return exporter.get_finished_spans()


def _one(exporter, name: str):
    matches = [s for s in _spans(exporter) if s.name == name]
    assert len(matches) == 1, \
        f"expected exactly one '{name}' span, got {[s.name for s in _spans(exporter)]}"
    return matches[0]


# ── One trace per leg, full span tree ───────────────────────────────────────────

def test_clean_run_emits_one_trace_with_full_span_tree(exporter, sdk_factory):
    sdk = sdk_factory("otel-clean")
    graph = sdk.compile(_one_node_graph("gen"), guards={"gen": "policy"})
    with patch(EVAL_TARGET, side_effect=_passes):
        r = sdk.start(graph, {"a": "", "b": ""})
    assert r.status == "completed"

    root = _one(exporter, "unistack.start")
    graph_span = _one(exporter, r.activity_id)          # run_name = activity_id
    node = _one(exporter, "gen")
    guard = _one(exporter, "guardrail_eval")

    assert root.parent is None
    assert root.attributes["unistack.status"] == "completed"
    assert root.attributes["session.id"] == r.activity_id
    assert root.attributes["langfuse.session.id"] == r.activity_id
    assert root.attributes["unistack.workflow"] == "otel-clean"
    assert graph_span.parent.span_id == root.context.span_id
    assert node.parent.span_id == graph_span.context.span_id
    assert guard.parent.span_id == root.context.span_id
    assert len({s.context.trace_id for s in (root, graph_span, node, guard)}) == 1
    assert guard.attributes["unistack.guardrail.passed"] is True
    assert guard.attributes["unistack.guardrail.node"] == "gen"
    assert guard.attributes["unistack.guardrail.mode"] == "keyword"   # no anthropic key


# ── Pause: pending doc carries the traceparent; leg root records the pause ──────

def test_pause_persists_trace_ids_and_message_in_pending_doc(exporter, sdk_factory, clean_db):
    sdk = sdk_factory("otel-pause")
    graph = sdk.compile(_one_node_graph("gen"), guards={"gen": "policy"})
    with patch(EVAL_TARGET, side_effect=_breaches):
        r = sdk.start(graph, {"a": "", "b": ""})
    assert r.status == "paused"

    root = _one(exporter, "unistack.start")
    assert root.attributes["unistack.status"] == "paused"
    assert root.attributes["unistack.pause.node"] == "gen"
    assert "hitl_pause_opened" in [e.name for e in root.events]

    doc = clean_db["hitl_resolutions"].find_one({"activity_id": r.activity_id})
    assert doc["status"] == "pending"
    assert doc["message"] == r.message
    assert doc["trace_id"] == format_trace_id(root.context.trace_id)
    assert doc["span_id"] == format_span_id(root.context.span_id)


# ── Resolve: retroactive hitl_pause into the pausing trace; legs linked ─────────

def test_approve_emits_retroactive_hitl_pause_into_pausing_trace(exporter, sdk_factory, clean_db):
    sdk = sdk_factory("otel-appr")
    graph = sdk.compile(_two_node_graph(), guards={"node_a": "policy"})
    with patch(EVAL_TARGET, side_effect=_breaches):
        r = sdk.start(graph, {"a": "", "b": ""})
    doc = clean_db["hitl_resolutions"].find_one({"activity_id": r.activity_id})

    r2 = sdk.resume(graph, r.activity_id, "approved", resolved_by="tester@x")
    assert r2.status == "completed"

    start_root = _one(exporter, "unistack.start")
    resume_root = _one(exporter, "unistack.resume")
    pause = _one(exporter, "hitl_pause")

    # the pause span lands in the START leg's trace, parented on its root, spanning
    # the human wait: start = the persisted opened_at (Mongo stores ms precision)
    assert pause.context.trace_id == start_root.context.trace_id
    assert pause.parent.span_id == start_root.context.span_id
    opened_ns = doc["opened_at"].replace(tzinfo=timezone.utc).timestamp() * 1e9
    assert abs(pause.start_time - opened_ns) < 5e6
    assert pause.end_time > pause.start_time
    assert pause.attributes["unistack.decision"] == "approved"
    assert pause.attributes["unistack.resolved_by"] == "tester@x"
    assert pause.attributes["unistack.pause.node"] == "node_a"
    assert pause.attributes["session.id"] == r.activity_id

    # the resume leg is its own trace, linked back to the pausing leg's root
    assert resume_root.context.trace_id != start_root.context.trace_id
    assert any(link.context.trace_id == start_root.context.trace_id
               and link.context.span_id == start_root.context.span_id
               for link in resume_root.links)
    assert "resolution_claimed" in [e.name for e in resume_root.events]
    assert resume_root.attributes["unistack.status"] == "completed"
    assert resume_root.attributes["unistack.decision"] == "approved"
    assert resume_root.attributes["session.id"] == r.activity_id


def test_reject_emits_hitl_pause_with_rejected_decision(exporter, sdk_factory):
    sdk = sdk_factory("otel-rej")
    graph = sdk.compile(_one_node_graph("gen"), guards={"gen": "policy"})
    with patch(EVAL_TARGET, side_effect=_breaches):
        r = sdk.start(graph, {"a": "", "b": ""})
    r2 = sdk.resume(graph, r.activity_id, "rejected", resolved_by="tester@x")
    assert r2.status == "hitl_rejected"

    pause = _one(exporter, "hitl_pause")
    assert pause.attributes["unistack.decision"] == "rejected"
    resume_root = _one(exporter, "unistack.resume")
    assert resume_root.attributes["unistack.status"] == "hitl_rejected"


def test_review_pause_traced_without_guardrail_span(exporter, sdk_factory):
    sdk = sdk_factory("otel-rev")
    graph = sdk.compile(_one_node_graph("work"), reviews=["work"])
    r = sdk.start(graph, {"a": "", "b": ""})
    assert r.status == "paused"
    assert not [s for s in _spans(exporter) if s.name == "guardrail_eval"]
    root = _one(exporter, "unistack.start")
    assert root.attributes["unistack.status"] == "paused"
    assert root.attributes["unistack.pause.node"] == "work"


# ── Disabled telemetry: zero spans, identical behaviour ─────────────────────────

def test_disabled_telemetry_emits_nothing_and_runs_fine(exporter, clean_db):
    sdk = UniStack.init(workflow="otel-off", mongo_uri=MONGO_URI, db_name=TEST_DB)
    graph = sdk.compile(_one_node_graph("gen"), guards={"gen": "policy"})
    with patch(EVAL_TARGET, side_effect=_breaches):
        r = sdk.start(graph, {"a": "", "b": ""})
    assert r.status == "paused"
    doc = clean_db["hitl_resolutions"].find_one({"activity_id": r.activity_id})
    assert doc["trace_id"] is None                       # uniform schema, no ids
    r2 = sdk.resume(graph, r.activity_id, "approved")
    assert r2.status == "completed"
    assert len(exporter.get_finished_spans()) == 0


# ── Judge LLM call: GenAI span with token usage under guardrail_eval ────────────

def test_judge_llm_span_carries_genai_usage(exporter, sdk_factory):
    sdk = sdk_factory("otel-judge", anthropic_api_key="sk-ant-fake")
    block = MagicMock()
    block.type = "tool_use"
    block.input = {"passed": True, "reason": "fine"}
    resp = MagicMock()
    resp.content = [block]
    resp.model = "claude-haiku-4-5-20251001"
    resp.usage.input_tokens = 123
    resp.usage.output_tokens = 45
    with patch("anthropic.Anthropic") as anthro:
        anthro.return_value.messages.create.return_value = resp
        verdict = sdk.evaluate("no fraud", "clean output", node="gen")
    assert verdict == {"passed": True, "reason": "fine"}

    guard = _one(exporter, "guardrail_eval")
    llm = _one(exporter, "chat claude-haiku-4-5-20251001")
    assert llm.parent.span_id == guard.context.span_id
    assert llm.attributes["gen_ai.request.model"] == "claude-haiku-4-5-20251001"
    assert llm.attributes["gen_ai.usage.input_tokens"] == 123
    assert llm.attributes["gen_ai.usage.output_tokens"] == 45
    assert json.loads(llm.attributes["output.value"]) == {"passed": True, "reason": "fine"}
    assert guard.attributes["unistack.guardrail.mode"] == "llm"
    assert guard.attributes["unistack.guardrail.passed"] is True


# ── Node-internal llm.invoke() with no config still lands under the node span ───

def test_node_internal_llm_invoke_without_config_is_traced(exporter, sdk_factory):
    llm = FakeListChatModel(responses=["hello"])

    def gen(state):
        msg = llm.invoke("say hi")                       # no config on purpose
        return {"a": msg.content, "b": ""}

    b = StateGraph(S)
    b.add_node("gen", gen)
    b.add_edge(START, "gen")
    b.add_edge("gen", END)

    sdk = sdk_factory("otel-prop")
    graph = sdk.compile(b)                               # no guards/reviews
    r = sdk.start(graph, {"a": "", "b": ""})
    assert r.status == "completed"
    assert r.state["a"] == "hello"

    node = _one(exporter, "gen")
    chat = [s for s in _spans(exporter) if s.name.startswith("chat ")]
    assert len(chat) == 1, "node-internal llm.invoke() was not traced"
    assert chat[0].parent.span_id == node.context.span_id
    assert chat[0].attributes["gen_ai.operation.name"] == "chat"
    assert "say hi" in chat[0].attributes["input.value"]


# ── Fail-open: a broken tracer never changes run behaviour ──────────────────────

def test_telemetry_failure_never_breaks_run(sdk_factory, clean_db):
    from opentelemetry.sdk.trace import Tracer
    sdk = sdk_factory("otel-boom")
    graph = sdk.compile(_one_node_graph("gen"), guards={"gen": "policy"})
    with patch.object(Tracer, "start_span", side_effect=RuntimeError("otel down")):
        with patch(EVAL_TARGET, side_effect=_breaches):
            r = sdk.start(graph, {"a": "", "b": ""})
        assert r.status == "paused" and r.node == "gen"
        r2 = sdk.resume(graph, r.activity_id, "approved", resolved_by="t")
        assert r2.status == "completed"
    doc = clean_db["hitl_resolutions"].find_one({"activity_id": r.activity_id})
    assert doc["status"] == "resolved" and doc["decision"] == "approved"
