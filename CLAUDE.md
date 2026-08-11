# UniStack SDK

## What this is

The UniStack SDK adds **guardrails** and **durable human-in-the-loop (HITL)** to an existing
LangGraph agent — without the agent author changing a single node. You hand UniStack your
`StateGraph` builder plus a map of which nodes to guard/review; it compiles the graph with static
breakpoints **and a durable checkpointer**, then drives the pauses.

HITL is **durable and non-blocking**: graph state is persisted, so an activity can pause for a
human and be resumed later — in a different process, after a restart. There is no in-memory
blocking and no Mongo queue; a tiny `hitl_resolutions` collection acts as the per-pause
resolution **lock**, the **pending-approvals index**, and the **audit record** (exactly one
resolver wins a pause). Tracing is pure OpenTelemetry (OTLP) — wireable to self-hosted
Langfuse or any collector. Standalone pip-installable package; no dependency on other repos.

## The whole integration (unchanged for the author)

```python
from unistack import UniStack
from my_app.graph import builder          # your existing, untouched StateGraph

sdk   = UniStack.init(workflow="content", mongo_uri="mongodb://localhost:27017",
                      anthropic_api_key="sk-ant-...",
                      otel_endpoint="https://langfuse.internal/api/public/otel",
                      context="Brand voice: professional, no unverified claims.")
graph = sdk.compile(builder, guards={"generate": "No unverified claims."}, reviews=["publish"])
```

Then either run it locally, or serve it as a durable runtime:

```python
# Local dev (single process, blocks and asks for each decision on the terminal):
result = sdk.run(graph, {"topic": "..."})            # -> "completed" | "hitl_rejected"

# Production (durable, non-blocking) — a service starts and later resumes:
r = sdk.start(graph, {"topic": "..."})               # -> "paused" | "completed"
r = sdk.resume(graph, r.activity_id, "approved")     # continue; may pause again or complete
```

Your nodes are plain functions — UniStack never asks you to import it inside them. **All config
is passed explicitly** — the SDK reads no environment variables, writes none, and loads no `.env`.

## Concepts

- **Guard** — `guards={node: policy}`. After the node runs, an LLM judges its output against the
  policy (+ `context`). Passed → continue silently. Breached → a HITL pause.
  **Fail-closed:** if the judge itself fails (API error, malformed verdict), the output is
  treated as a breach → HITL pause with "judge unavailable". A degraded judge never silently
  passes output, and never crashes the activity.
- **Review** — `reviews=[node]`. Unconditional human sign-off after the node. Always pauses.

Both are LangGraph **static breakpoints** (`interrupt_after`). The author's graph topology is
never modified. In a parallel fan-out, **every** guarded node of the super-step is judged (a
breach in any of them pauses; the message lists all breaches). Dynamic `interrupt()` inside a
node is **not supported** — the SDK detects it and raises `UniStackError` instead of hanging.

## Public API

```python
sdk   = UniStack.init(...)                                  # see parameters below
graph = sdk.compile(builder, guards={"n": "policy"}, reviews=["m"])

r = sdk.start(graph, initial_state, run_id=None)           # non-blocking; run_id → unique ts+hex
r = sdk.resume(graph, activity_id, decision, resolved_by=None)   # "approved" | "rejected"
r = sdk.run(graph, initial_state, decide=None)             # local convenience over start/resume
sdk.evaluate("policy", output_str)                         # {"passed", "reason"} — raw guard check
sdk.close()                                                # also flushes buffered OTel spans
                                                           # or use `with UniStack.init(...) as sdk:`

# RunResult (dataclass): .activity_id  .state  .node  .message
#   .status: "completed" | "paused" | "hitl_rejected" | "not_found" | "failed"
```

`resolved_by` accepts a plain **string** (a bare label — what local and library callers pass)
or a **`Resolver`** carrying verified identity:

```python
Resolver(label="approver@x", subject="sub-1", issuer="https://idp", auth_mode="oidc")
```

The graph-runtime builds one from the caller's validated token. Strings coerce automatically,
so nothing that passed a string before needs to change.

### `UniStack.init()` parameters

| Param | Required | Purpose |
|---|---|---|
| `workflow` | Yes | Workflow name; prefix of the activity id `{workflow}-{run_id}` |
| `mongo_uri` | Yes | MongoDB — backs the **durable checkpointer** (graph state) + `hitl_resolutions` |
| `anthropic_api_key` | No | LLM guardrail judge; keyword-scan fallback when omitted. Judge failures fail **closed** → HITL pause |
| `otel_endpoint` | No | Enables OTLP/HTTP tracing — base URL of any OTLP backend (Langfuse: `…/api/public/otel`; `/v1/traces` is appended if missing) |
| `otel_headers` | No | OTLP headers, dict or the standard `k=v,k2=v2` string (Langfuse: `Authorization=Basic <base64(pk:sk)>`) |
| `otel_service_name` | No | OTel `service.name` resource (default `unistack-{workflow}`) |
| `context` | No | Business-domain text for the guardrail judge |
| `db_name` | No | Mongo database (default `unistack`) |
| `guardrail_model` | No | Judge model (default `claude-haiku-4-5-20251001`) |
| `checkpointer` | No | Override the default `MongoDBSaver` (e.g. a Postgres saver) |
| `tracer_provider` | No | Caller-owned OTel `TracerProvider` — overrides `otel_endpoint`; the test seam. Never installed globally |

## How start / resume work (durable, request-driven)

```
start(graph, initial_state):
  advance: stream segments; every reached GUARD in the segment is judged inline (pass → keep going).
  pause at a guard breach or a review node → record a pending resolution (status "pending",
  carrying the pause message + this leg's OTel trace ids) → return status "paused".
  reach END → return "completed".

resume(graph, activity_id, decision):     # triggered by the human's decision, in any process
  load the persisted checkpoint (thread_id = activity_id).
  CLAIM the pause's resolution atomically in hitl_resolutions — exactly one resolver wins;
  a concurrent/repeated resolve is a recorded no-op, never a second advance.
  emit the hitl_pause span retroactively into the pausing leg's trace (decision + wait duration).
  reject → "hitl_rejected".  approve → advance to the next pause or END.
  unknown activity_id → "not_found";  already-finalized activity → "completed" (no-op).
```

Resuming a static breakpoint re-streams the graph with `None` input (not `Command(resume=…)`,
which is for dynamic `interrupt()`). Because state is durable, `start` and `resume` can be
different requests, processes, or a process that restarted in between.

**Idempotency / terminal pauses:** a static breakpoint on a *terminal* node leaves
`get_state().next` empty — indistinguishable from a completed graph. So `resume` claims first,
checks the decision (reject always halts), and treats "approve with nothing left to run" as a
harmless completed no-op. The old double-approve race is closed by the claim: the loser gets a
"pause already resolved" no-op result and never advances the graph.

## Deployment — `unistack serve`

The focused **graph-runtime** is the only component that imports the graph + SDK. It exposes
`POST /activities` (start) and `POST /activities/{id}/resolve` (resume), nothing else:

```bash
unistack serve my_app.graph:builder --workflow content \
  --guard "generate=No unverified claims." --review publish --context "Brand voice: …"
# auth comes from UNISTACK_OIDC_* env vars — see the Auth section below
```

**`UNISTACK_CONFIG` — governance as data, collocated with the graph.** Passing policy text
(especially `context`, which can be long) as shell arguments on every deploy is awkward, and it
puts the policy somewhere the graph's author doesn't see it. Instead, the author's module can
declare a plain dict next to `builder`:

```python
# my_app/graph.py — still zero `unistack` import; this is just data.
UNISTACK_CONFIG = {
    "workflow": "content",
    "guards": {"generate": "No unverified claims."},
    "reviews": ["publish"],
    "context": "Brand voice: professional, no unverified claims.",
}
```

`unistack serve` auto-discovers a sibling `UNISTACK_CONFIG` in the same module as `builder` (by
name — absent is fine, fully backward compatible). With it present, the deploy command collapses
to `unistack serve my_app.graph:builder` — no flags. CLI flags still work and **merge on top**:
`--guard`/`--review` add to (CLI wins per-key on `--guard` collisions) the config's sets;
`--context`/`--workflow` override outright if passed. Useful for a one-off ops override without a
redeploy, without making the common case carry the whole policy on the command line.

Install with the server extra: `pip install "unistack[server]"` (fastapi + uvicorn + pyjwt).
Everything read-only (listing pending approvals, fetching pause history) is **not** here — it
reads the `hitl_resolutions` Mongo collection (see unistack-api). Self-host anywhere (Azure
Container Apps, etc.) with a managed Mongo. **Scaling:** state is durable and pause resolution
is claim-based, so multiple uvicorn workers / replicas behind a load balancer are safe.

## Auth

Token verification lives in the shared **`unistack-auth`** package — see
[its CLAUDE.md](../unistack-auth/CLAUDE.md) for the `Principal` shape, claim mapping, status-code
taxonomy and hard rules. **This repo owns only the wiring**, which is:

| Endpoint | Scope required |
|---|---|
| `POST /activities` | `activity.start` |
| `POST /activities/{id}/resolve` | `activity.resolve` |
| `GET /health` | none — liveness probes carry no token |

**Auth is mandatory and not omittable** — `auth` is a required keyword-only argument to
`create_app`, and `unistack serve` exits rather than degrading when config is incomplete.

```bash
# Production — env-driven, like MONGO_URI and OTEL_*
export UNISTACK_OIDC_JWKS_URL=https://login.microsoftonline.com/<tenant>/discovery/v2.0/keys
export UNISTACK_OIDC_ISSUER=https://login.microsoftonline.com/<tenant>/v2.0
export UNISTACK_OIDC_AUDIENCE=api://unistack-runtime
unistack serve my_app.graph:builder

# Local dev — no IdP needed; identity is CONFIGURED, never taken from the caller
unistack serve my_app.graph:builder --auth-mode token --token dev-secret
```

> **Grant the two scopes to DIFFERENT identities.** Separation is what stops an agent approving
> its own guardrail breaches, but only if an operator actually grants them apart — one service
> principal holding both still self-approves every run. A real `--deny-self-approval` check
> needs `started_by` on the activity record, which arrives with BUILD_PLAN item 3.

**The approver cannot be forged.** `resolved_by` is no longer accepted in the resolve body
(sending it is a **422**); the verified `Principal` becomes the `Resolver` written to the audit
doc as `resolved_by` (label, unchanged meaning) plus `resolved_by_subject`, `resolved_by_issuer`
and `resolved_auth_mode` — that last field is what distinguishes a verified approver from a
dev-mode attribution.

`tests/conftest.py` generates an RSA keypair and serves its JWKS over loopback, so the auth
tests need no identity provider.

> **The gotcha that costs an hour:** Keycloak puts `account` in `aud` by default, not your
> client. Add an **Audience** protocol mapper to the client scope or every token 401s on
> audience and it looks like a UniStack bug.

```bash
JWKS=http://localhost:8080/realms/unistack/protocol/openid-connect/certs
ISS=http://localhost:8080/realms/unistack
curl -d grant_type=client_credentials -d client_id=runner -d client_secret=… \
     -d scope=activity.start $ISS/protocol/openid-connect/token
```

## OpenTelemetry — the span model

Pass `otel_endpoint` (+ optional `otel_headers` / `otel_service_name`), or hand in your own
`tracer_provider`. Pure OTLP/HTTP — point it at self-hosted Langfuse
(`https://<langfuse>/api/public/otel`, `Authorization=Basic <base64(pk:sk)>`), an OTel
collector, or a hyperscaler tracing agent. Vendor-neutral attributes throughout (GenAI
semconv `gen_ai.*`, `session.id`, `unistack.*`); the only `langfuse.*` keys are additive
rendering hints, harmless elsewhere.

**One trace per leg.** `start()` and each `resume()` open their own root span
(`unistack.start` / `unistack.resume`); every leg of an activity carries
`session.id = activity_id` (Langfuse's Sessions view groups them), and a resume root **links**
to the leg it resolves. Under a leg root: the graph run (named by activity id) → node spans →
LLM spans (`chat {model}`, with `gen_ai.usage.*` tokens) via the SDK's own LangChain callback
handler; `guardrail_eval` spans carry policy/verdict plus the judge's Claude call as a child
generation.

**The `hitl_pause` span is emitted retroactively.** OTLP cannot export an open span, and a
pause can outlive the process — so at pause time the leg root's trace ids are persisted in the
pending doc, and at resolve time the claim winner emits the completed `hitl_pause` span into
the **pausing leg's** trace with `start_time = opened_at`. Its duration is the real human wait;
it carries the decision + resolver. Emitted exactly once (the claim guarantees it).

**Pending approvals live in Mongo, not the tracing backend:**
`hitl_resolutions.find({"status": "pending"})` — see unistack-api. Telemetry is fail-open:
every telemetry call is best-effort (warn + continue), state and resume never depend on the
tracing backend, and a telemetry failure can never change a verdict or a run's outcome.

## File structure

```
unistack/
  __init__.py      ← exports UniStack, RunResult, Resolver, UniStackError; NullHandler on the logger
  core.py          ← UniStack: init, compile, start, resume, run, guard eval, resolution
                     claims, retroactive hitl_pause emission; Resolver (audit identity)
  _telemetry.py    ← Telemetry (instance-scoped OTel provider/spans, fail-open) +
                     OTelCallbackHandler (LangChain events → OTel spans, GenAI semconv)
  _guardrail.py    ← evaluate_guardrail() via Claude tool-use (keyword-scan fallback; fail-closed)
  server.py        ← create_app(sdk, graph, *, auth): the focused graph-runtime (FastAPI)
  cli.py           ← `unistack serve module:builder …`; discovers a sibling UNISTACK_CONFIG
pyproject.toml  requirements.txt  README.md
tests/conftest.py (RSA keypair + loopback JWKS + make_token)  tests/test_auth.py
tests/test_guardrail.py  tests/test_telemetry.py  tests/test_server.py  tests/test_cli.py
```

## MongoDB — what this writes (and cleans up)

Database `unistack` (configurable). The SDK writes the **durable checkpointer** collections
(`checkpoints`, `checkpoint_writes`) — LangGraph's persisted graph state — plus
**`hitl_resolutions`**: one tiny doc per pause, unique on `(activity_id, checkpoint_id)`.
It is three things at once: the per-pause resolution **lock** (exactly one resolver wins;
duplicates become no-ops), the **pending-approvals index** (`status: "pending"`, with node,
message, workflow, opened_at — what unistack-api lists), and the **audit record** (decision,
`resolved_by` + `resolved_by_subject` + `resolved_by_issuer` + `resolved_auth_mode`,
resolved_at, plus the pausing leg's OTel `trace_id`/`span_id` for deep-linking into the tracing
backend). It is **not** a queue — nothing polls it; resolution is request-driven. Resolution
docs are kept after an activity terminates (bounded: one per pause).

**Retention.** The moment an activity reaches a terminal outcome (completed or rejected), its
thread is deleted via `MongoDBSaver.delete_thread(activity_id)` — nothing will ever resume from it
again. So at steady state Mongo holds working-state only for genuinely in-flight / paused
activities; nothing lingers after completion. Cleanup is best-effort (a Mongo hiccup logs and
leaves docs) and never affects the returned status/state; exported traces are never touched.

Notes:
- We do **not** prune intermediate (resumed-past) checkpoints mid-activity: `MongoDBSaver.prune()`
  is unimplemented (raises `NotImplementedError`), and hand-rolling schema-level deletes would be
  fragile and can silently corrupt graphs using the experimental `DeltaChannel`. Those intermediate
  docs are small and deleted anyway when the activity terminates.
- **Abandoned** activities (started, paused, never resolved) keep their checkpoint indefinitely —
  correctly, since it is still resumable. If you want to reap them, set a `MongoDBSaver` `ttl`
  **longer than your worst-case approval SLA** (too short would delete a live pause and break
  resume). Off by default so the default path can never delete a live pause.

## Environment variables

**None read, none written.** All config is constructor params; OTel tracing is instance-scoped
(the SDK builds — or is handed — a `TracerProvider` and injects its own callback handler into
the graph config; `trace.set_tracer_provider` is never called, no `OTEL_*` globals). The
`unistack serve` CLI, acting as the consuming app, reads `MONGO_URI` / `ANTHROPIC_API_KEY` /
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` (or `OTEL_EXPORTER_OTLP_ENDPOINT`) /
`OTEL_EXPORTER_OTLP_HEADERS` / `OTEL_SERVICE_NAME`, plus the auth vars — `UNISTACK_AUTH_MODE`,
`UNISTACK_OIDC_JWKS_URL`, `UNISTACK_OIDC_ISSUER`, `UNISTACK_OIDC_AUDIENCE`,
`UNISTACK_API_TOKEN`, `UNISTACK_DEV_IDENTITY`, `UNISTACK_TOKEN_SCOPES` — and passes them in.
`auth.py` itself reads no environment: it receives an `AuthConfig`.
Note `langsmith` remains installed *transitively* (langchain-core requires it) — never set
`LANGSMITH_TRACING=true` in a deployment env, or langchain's own global tracer re-activates.

## Logging

The SDK never prints — it logs through `logging.getLogger("unistack")` (with a `NullHandler`
by default, per library convention). Consuming apps opt in, e.g.
`logging.basicConfig(level=logging.INFO)`. The only exception is `run()`'s interactive
terminal prompt, which is deliberately stdin/stdout.

## Install & test

```bash
python3.13 -m venv venv
venv/bin/python -m pip install -e ".[server,dev]"   # server: fastapi/uvicorn/pyjwt · dev: pytest/httpx
PYTHONPATH=. venv/bin/python -m pytest tests/ -v    # needs MongoDB on localhost:27017
```

Auth tests need no IdP: `tests/conftest.py` generates an RSA keypair per session and serves its
JWKS from a loopback HTTP server, so the real `PyJWKClient` fetch/cache path is exercised with
no network and no test-only seam in shipped code.

## Hard constraints

1. Never modify the author's graph topology — guards/reviews are static breakpoints only.
   Dynamic `interrupt()` in a node is unsupported: detect it and raise `UniStackError`,
   never loop.
2. Activity IDs are human-readable: `{workflow}-{run_id}`, `run_id` defaulting to a UTC
   microsecond timestamp plus a 4-hex-char suffix (collision-proof across replicas).
   Never UUID.
3. A HITL pause is not an error — `status="paused"` is normal, never `failed`.
4. Guardrails use LLM evaluation — policy enforcement, not deterministic computation. The
   judge **fails closed**: an unavailable/unparseable judge is a breach (→ pause), never a
   silent pass, never a crash.
5. On rejection, the activity is abandoned (not resumed).
6. State lives in the durable checkpointer; telemetry is observability only. Resume must never
   depend on the tracing backend being reachable — every telemetry call is fail-open
   (best-effort, warn + continue) and can never change a verdict or a run's outcome.
7. A terminal activity's checkpoints are deleted (best-effort); read final state BEFORE deleting.
   Cleanup failures must never change the returned status/state.
8. A pause is resolved exactly once: `resume` must win the `hitl_resolutions` claim before
   advancing; losers return a no-op, unknown ids return `not_found`.
9. The SDK reads no environment variables and writes none; tracing is instance-scoped — the
   OTel provider is constructor-supplied or SDK-built, never installed globally, and no
   instrumentor may monkey-patch LangChain/LangGraph.
10. `langgraph` stays a version RANGE (`>=1.2,<2.0`), never a pin — the SDK must install
    alongside whatever LangGraph the consumer's agent already uses.
11. Auth is neither optional nor omittable: `create_app` takes `auth` as a required
    keyword-only argument and the CLI exits on incomplete config, so there is no code path
    that serves an open runtime. The audit identity comes from verified token claims, never
    from a request body. Starting and resolving are separate scopes. Only `RS256` is
    accepted and the algorithm is never read from the token.

## Roadmap

### Agreed build order (August 2026) — see `../BUILD_PLAN.md` for the full plan

The SDK items, in order. Do not start one before its predecessors.

1. ~~**Inbound identity & authorization.**~~ **DONE (v0.3.0)** — see the Auth section above.
   Validated OIDC/JWT with separate `activity.start` / `activity.resolve` scopes, `resolved_by`
   derived from verified claims (and rejected with 422 if sent in the body), static-token dev
   mode behind `--auth-mode token`, and the open mode removed entirely. Also fixed here: the
   already-resolved message no longer leaked the prior approver's identity to any caller.
   *Still open, deliberately:* one identity holding both scopes can self-approve. Closing that
   needs `started_by` on the activity record, which item 3 introduces.
2. **LiteLLM gateway.** `base_url` + model **alias** (`agent-primary` / `judge-fast`) plumbed
   through `UniStack.init()` into `_guardrail.py`, via the Anthropic passthrough route so
   forced tool-use stays exact. SDK sets guardrails per request (default-on is a LiteLLM
   enterprise feature). `activity_id`/`workflow`/`node` in request metadata. W3C `traceparent`
   propagated so gateway spans nest inside the node span. **Budget exhaustion opens a HITL
   pause, not a crash.** LiteLLM must never log to Langfuse.
3. **Activity record.** One small doc per activity in `unistack.activities` at start, updated
   at terminal, carrying outcome, `trace_id`, and `analysis_status: "pending"`. Today a run
   that never pauses leaves *no* durable record — checkpoints are deleted at terminal and
   `hitl_resolutions` only exists when something paused, so the only evidence is a fail-open
   trace. Same weight and role as `hitl_resolutions`: an index and audit record, not a queue.
4. **Trace enrichment.** Capture retrieval/context inputs as node-span attributes and tag
   traces with `workflow`. Langfuse evaluators can only judge what the spans contain — a
   groundedness judge that cannot see the retrieved context produces confident noise.
5. **Security layer** (distinct from guards — see below). Layer 1 always on and deterministic:
   tool allow-list, argument inspection (secrets, PII, destructive verbs), untrusted-source
   flagging. Layer 2 LLM security judge behind a trigger only (ambiguous flag, high-risk tool,
   retrieved content) — never on by default. Findings to `unistack.security_events` in Mongo
   (authoritative, telemetry-independent) plus a span event for trace correlation.

**The security check is not the business-policy guard.** A guard runs *after* a node, judges
its *output* against policy, and *pauses for a human*. A security check runs *before* a call
goes out, inspects the *proposed call and arguments*, and *blocks outright*. Different
question, different moment, different response — keep them separate. And do not LLM what a
list can do: allow-lists and pattern checks are cheaper, faster, non-flaky, and auditable.

### Planned (must-have, not yet built)

- **Knowledge-base-backed guards.** Today a guard is a plain policy *string* judged by the LLM.
  Add a first-class **knowledge base** resource so a guard can ground its judgment against
  retrieved documents (e.g. a client compliance manual), not just inline text — e.g.
  `guards={"generate": {"policy": "...", "knowledge_base": "compliance_docs"}}`, with a retrieval
  step feeding the judge's context before the compliance check. Bounded addition (a retrieval call
  + a KB registry), not a rewrite. **Highest-value next feature.**
- **Multiple deployment surfaces.** `unistack serve` exposes one REST API today. The graph engine
  is transport-agnostic — `start()`/`resume()`/status over an `activity_id` — so additional
  surfaces are **thin adapters over that same core, never forks of the engine**:
  - *Webhook* — a POST adapter mapping a third-party payload (Slack event, etc.) to `initial_state`
    → `start()`. Nearly identical to today's `POST /activities`; just payload translation.
  - *Schedule / cron* — prefer the **client's own scheduler** (Cloud Scheduler / EventBridge / cron)
    hitting the existing endpoint on a schedule (zero new engine code, survives restarts) over an
    embedded in-process scheduler.
  - *MCP server* — expose the graph as an MCP tool so the client's other agents/tools can call it.
    A paused HITL activity returns `{status: paused, activity_id}`; the caller resumes via the same
    resolve path. Genuinely forward-looking (makes the agent composable into client tooling).
  - *Chat* — a streaming (SSE/WebSocket) adapter; LangGraph streams natively. A HITL pause maps to
    "assistant is awaiting approval" in the chat UX.

### Direction noted (later, not urgent)

- **Managed credentials, resolved from the client's own secret manager — NOT a UniStack-owned
  vault.** Raw env vars are fine for infra secrets (`MONGO_URI`); per-integration credentials
  (Gmail/Slack tokens, client REST keys) should resolve by name from the client's existing
  AWS Secrets Manager / GCP Secret Manager / Azure Key Vault / Vault via a thin resolver interface.
  UniStack must not become a secrets custodian (owning encryption/rotation is a liability and
  contradicts "everything stays in the client's infrastructure").

### Deferred by design (do NOT re-propose without new reasons)

- **Runtime evaluators — rejected in the hot path.** Only things that must *gate* execution live
  mid-run — guards, reviews, and (new) security blocks. Measurement/scoring (compliance, revenue,
  quality, hallucination, groundedness, KRA/ROI) is a **pure function of the completed trace**,
  runs offline, and adds zero latency to the agent. Keeping the runtime lean is deliberate.
  (If a score must influence control flow, that is a guard/router, not an evaluator.)
  **Updated Aug 2026 — the destination changed, the principle did not:** scoring now runs as
  **Langfuse evaluators** on live observations (self-hosted Langfuse includes them free;
  sampling + async, judge model pointed at LiteLLM), with the **projector** in `unistack-api`
  assembling projections from traces + scores. `unistack-brain` is being retired.
- **A "Tables" data abstraction — rejected.** The LangGraph author can already read/write any
  Mongo/Postgres table directly from their own nodes (plain Python); wrapping that would be scope
  creep and would violate the thin-latch-on principle. The only structured store UniStack owns is
  the governance/HITL audit trail, which already exists (`hitl_resolutions` + the OTel traces).
