"""
Tests for the graph-runtime server (create_app): bearer-token auth, start/resolve
round-trips, decision validation, and 404 for unknown activities.
Requires MongoDB on localhost:27017 (isolated "unistack_test" database).
"""

from typing import TypedDict

from fastapi.testclient import TestClient
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pymongo import MongoClient

import pytest

from unistack import UniStack
from unistack.server import create_app

MONGO_URI = "mongodb://localhost:27017"
TEST_DB = "unistack_test"
TOKEN = "sekret-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


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


class S(TypedDict):
    a: str
    b: str


def _builder():
    def work(state):
        return {"a": "output", "b": ""}
    b = StateGraph(S)
    b.add_node("work", work)
    b.add_edge(START, "work")
    b.add_edge("work", END)
    return b


def _client(token=TOKEN) -> TestClient:
    sdk = UniStack.init(workflow="srv", mongo_uri=MONGO_URI, db_name=TEST_DB)
    graph = sdk.compile(_builder(), reviews=["work"])
    return TestClient(create_app(sdk, graph, token=token))


# ── Auth ────────────────────────────────────────────────────────────────────────

def test_start_requires_bearer_token():
    c = _client()
    assert c.post("/activities", json={"initial_state": {"a": "", "b": ""}}).status_code == 401
    wrong = {"Authorization": "Bearer nope"}
    assert c.post("/activities", json={"initial_state": {"a": "", "b": ""}},
                  headers=wrong).status_code == 401


def test_resolve_requires_bearer_token():
    c = _client()
    assert c.post("/activities/srv-x/resolve", json={"decision": "approve"}).status_code == 401


def test_health_is_open():
    c = _client()
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["workflow"] == "srv"


def test_no_token_app_is_open_but_works():
    c = _client(token=None)
    r = c.post("/activities", json={"initial_state": {"a": "", "b": ""}})
    assert r.status_code == 201


# ── Start / resolve round-trip ──────────────────────────────────────────────────

def test_start_resolve_roundtrip():
    c = _client()
    r = c.post("/activities", json={"initial_state": {"a": "", "b": ""}}, headers=AUTH)
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "paused" and body["node"] == "work"

    r2 = c.post(f"/activities/{body['activity_id']}/resolve",
                json={"decision": "approve", "resolved_by": "t@x.com"}, headers=AUTH)
    assert r2.status_code == 200
    assert r2.json()["status"] == "completed"

    # repeated resolve of a finalized activity: idempotent no-op, not a 404
    r3 = c.post(f"/activities/{body['activity_id']}/resolve",
                json={"decision": "approve"}, headers=AUTH)
    assert r3.status_code == 200
    assert r3.json()["status"] == "completed"


def test_resolve_reject_halts():
    c = _client()
    body = c.post("/activities", json={"initial_state": {"a": "", "b": ""}},
                  headers=AUTH).json()
    r = c.post(f"/activities/{body['activity_id']}/resolve",
               json={"decision": "reject"}, headers=AUTH)
    assert r.status_code == 200 and r.json()["status"] == "hitl_rejected"


# ── Validation & unknown ids ────────────────────────────────────────────────────

def test_resolve_invalid_decision_422():
    c = _client()
    r = c.post("/activities/srv-x/resolve", json={"decision": "maybe"}, headers=AUTH)
    assert r.status_code == 422


def test_resolve_unknown_activity_404():
    c = _client()
    r = c.post("/activities/srv-19990101T000000000000-dead/resolve",
               json={"decision": "approve"}, headers=AUTH)
    assert r.status_code == 404
