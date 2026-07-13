"""
Focused graph-runtime — the ONLY component that imports the graph + SDK.

Its whole job is to start an activity and resume it on a human decision. State is
durable (checkpointer), so start and resume can be different requests / processes.
Everything read-only (listing pending approvals, fetching a thread) is NOT here — it
comes from LangSmith directly.

Build it with `create_app(sdk, graph, token=...)` or run it with
`unistack serve module:builder --token ...`. When a token is set, both POST
endpoints require `Authorization: Bearer <token>`; without one the runtime is open —
anyone who can reach the port can start and approve activities.
"""

import secrets

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel


class StartRequest(BaseModel):
    initial_state: dict
    run_id: str | None = None


class ResolveRequest(BaseModel):
    decision: str                      # "approve" | "reject"
    resolved_by: str | None = None


def _result(r) -> dict:
    return {"activity_id": r.activity_id, "status": r.status, "node": r.node, "message": r.message}


def create_app(sdk, graph, token: str | None = None) -> FastAPI:
    """A thin FastAPI hosting one compiled graph: start + resolve, nothing else."""
    app = FastAPI(title="UniStack graph-runtime")

    def require_auth(authorization: str | None = Header(default=None)):
        if token is None:
            return
        scheme, _, supplied = (authorization or "").partition(" ")
        if scheme != "Bearer" or not secrets.compare_digest(supplied.strip(), token):
            raise HTTPException(401, "missing or invalid bearer token")

    @app.get("/health")
    def health():
        return {"status": "ok", "workflow": sdk._workflow}

    @app.post("/activities", status_code=201, dependencies=[Depends(require_auth)])
    def start_activity(body: StartRequest):
        return _result(sdk.start(graph, body.initial_state, body.run_id))

    @app.post("/activities/{activity_id}/resolve", dependencies=[Depends(require_auth)])
    def resolve_activity(activity_id: str, body: ResolveRequest):
        if body.decision not in ("approve", "reject"):
            raise HTTPException(422, "decision must be 'approve' or 'reject'")
        decision = "approved" if body.decision == "approve" else "rejected"
        r = sdk.resume(graph, activity_id, decision, resolved_by=body.resolved_by)
        if r.status == "not_found":
            raise HTTPException(404, r.message or f"no such activity: '{activity_id}'")
        return _result(r)

    return app


__all__ = ["create_app"]
