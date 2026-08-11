"""
Tests for the graph-runtime server (create_app): OIDC and static-token auth, scope
separation, unforgeable approver identity, start/resolve round-trips, decision validation,
and 404 for unknown activities.
Requires MongoDB on localhost:27017 (isolated "unistack_test" database).
"""

from typing import TypedDict

import pytest
from fastapi.testclient import TestClient
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pymongo import MongoClient

from tests.conftest import AUDIENCE, ISSUER
from unistack import UniStack
from unistack_auth import AuthConfig
from unistack.server import create_app

MONGO_URI = "mongodb://localhost:27017"
TEST_DB = "unistack_test"
TOKEN = "sekret-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
BOTH_SCOPES = ("activity.start", "activity.resolve")


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


def _app(auth: AuthConfig) -> TestClient:
    sdk = UniStack.init(workflow="srv", mongo_uri=MONGO_URI, db_name=TEST_DB)
    graph = sdk.compile(_builder(), reviews=["work"])
    return TestClient(create_app(sdk, graph, auth=auth))


def _client(scopes=BOTH_SCOPES) -> TestClient:
    """Static-token app — the default for tests that are not about OIDC itself."""
    return _app(AuthConfig.static_token(token=TOKEN, identity="dev@local", scopes=scopes))


def _oidc_client(jwks_url) -> TestClient:
    return _app(AuthConfig.oidc(jwks_url=jwks_url, issuer=ISSUER, audience=AUDIENCE))


def _start(c, headers=AUTH):
    return c.post("/activities", json={"initial_state": {"a": "", "b": ""}}, headers=headers)


# ── Auth is mandatory ───────────────────────────────────────────────────────────

def test_create_app_requires_auth():
    """There is no way to construct an unauthenticated runtime, not even by omission."""
    sdk = UniStack.init(workflow="srv", mongo_uri=MONGO_URI, db_name=TEST_DB)
    graph = sdk.compile(_builder(), reviews=["work"])
    with pytest.raises(TypeError):
        create_app(sdk, graph)


def test_start_requires_bearer_token():
    c = _client()
    assert _start(c, headers=None).status_code == 401
    assert _start(c, headers={"Authorization": "Bearer nope"}).status_code == 401


def test_resolve_requires_bearer_token():
    c = _client()
    assert c.post("/activities/srv-x/resolve", json={"decision": "approve"}).status_code == 401


def test_health_is_open():
    r = _client().get("/health")
    assert r.status_code == 200 and r.json()["workflow"] == "srv"


def test_401_carries_www_authenticate():
    r = _start(_client(), headers=None)
    assert r.status_code == 401 and r.headers["WWW-Authenticate"].startswith("Bearer")


# ── Scope separation — the acceptance criterion ─────────────────────────────────

def test_start_scope_only_gets_403_on_resolve():
    """Whoever can run an agent must not be able to sign off its own guardrail breach."""
    c = _client(scopes=("activity.start",))
    body = _start(c).json()
    r = c.post(f"/activities/{body['activity_id']}/resolve",
               json={"decision": "approve"}, headers=AUTH)
    assert r.status_code == 403
    assert 'scope="activity.resolve"' in r.headers["WWW-Authenticate"]
    assert "insufficient_scope" in r.headers["WWW-Authenticate"]


def test_resolve_scope_only_gets_403_on_start():
    r = _start(_client(scopes=("activity.resolve",)))
    assert r.status_code == 403
    assert 'scope="activity.start"' in r.headers["WWW-Authenticate"]


# ── The approver identity cannot be forged ──────────────────────────────────────

def test_resolved_by_from_body_is_rejected():
    c = _client()
    body = _start(c).json()
    r = c.post(f"/activities/{body['activity_id']}/resolve",
               json={"decision": "approve", "resolved_by": "ceo@corp.com"}, headers=AUTH)
    assert r.status_code == 422
    assert "resolved_by" in r.text


def test_resolved_by_is_taken_from_token_not_body(clean_db, jwks_url, make_token):
    """
    Asserted on the PERSISTED document, not the response — the audit record is what an
    auditor reads, and it is what must be impossible to forge from a request payload.
    """
    c = _oidc_client(jwks_url)
    token = make_token(BOTH_SCOPES, email="approver@example.com", sub="sub-42")
    headers = {"Authorization": f"Bearer {token}"}

    body = c.post("/activities", json={"initial_state": {"a": "", "b": ""}},
                  headers=headers).json()
    assert c.post(f"/activities/{body['activity_id']}/resolve",
                  json={"decision": "approve"}, headers=headers).status_code == 200

    doc = clean_db.hitl_resolutions.find_one({"activity_id": body["activity_id"]})
    assert doc["resolved_by"] == "approver@example.com"
    assert doc["resolved_by_subject"] == "sub-42"
    assert doc["resolved_by_issuer"] == ISSUER
    assert doc["resolved_auth_mode"] == "oidc"


def test_static_token_mode_attributes_to_dev_identity(clean_db):
    """Dev mode still cannot take identity from the caller — and says so in the record."""
    c = _client()
    body = _start(c).json()
    c.post(f"/activities/{body['activity_id']}/resolve",
           json={"decision": "approve"}, headers=AUTH)

    doc = clean_db.hitl_resolutions.find_one({"activity_id": body["activity_id"]})
    assert doc["resolved_by"] == "dev@local"
    assert doc["resolved_auth_mode"] == "token"


# ── Start / resolve round-trip ──────────────────────────────────────────────────

def test_start_resolve_roundtrip():
    c = _client()
    r = _start(c)
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "paused" and body["node"] == "work"

    r2 = c.post(f"/activities/{body['activity_id']}/resolve",
                json={"decision": "approve"}, headers=AUTH)
    assert r2.status_code == 200
    assert r2.json()["status"] == "completed"

    # repeated resolve of a finalized activity: idempotent no-op, not a 404
    r3 = c.post(f"/activities/{body['activity_id']}/resolve",
                json={"decision": "approve"}, headers=AUTH)
    assert r3.status_code == 200
    assert r3.json()["status"] == "completed"


def test_resolve_reject_halts():
    c = _client()
    body = _start(c).json()
    r = c.post(f"/activities/{body['activity_id']}/resolve",
               json={"decision": "reject"}, headers=AUTH)
    assert r.status_code == 200 and r.json()["status"] == "hitl_rejected"


def test_oidc_end_to_end_over_loopback_jwks(jwks_url, make_token):
    """The one test that exercises the real JWKS fetch across HTTP."""
    c = _oidc_client(jwks_url)
    headers = {"Authorization": f"Bearer {make_token(BOTH_SCOPES)}"}
    assert c.post("/activities", json={"initial_state": {"a": "", "b": ""}},
                  headers=headers).status_code == 201


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
