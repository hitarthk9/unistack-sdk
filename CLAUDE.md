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
UNISTACK_API_TOKEN=<secret> unistack serve my_app.graph:builder --workflow content \
  --guard "generate=No unverified claims." --review publish --context "Brand voice: …"
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

Install with the server extra: `pip install "unistack[server]"` (adds fastapi + uvicorn).
**Auth:** with `--token` / `UNISTACK_API_TOKEN` set, both POST endpoints require
`Authorization: Bearer <token>` (401 otherwise); serving without one prints a loud warning —
an open runtime lets anyone who can reach the port start and approve activities. `/health`
stays open. Everything read-only (listing pending approvals, fetching pause history) is **not**
here — it reads the `hitl_resolutions` Mongo collection (see unistack-api). Self-host anywhere
(Azure Container Apps, etc.) with a managed Mongo. **Scaling:** state is durable and pause
resolution is claim-based, so multiple uvicorn workers / replicas behind a load balancer are safe.

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
  __init__.py      ← exports UniStack, RunResult, UniStackError; NullHandler on the logger
  core.py          ← UniStack: init, compile, start, resume, run, guard eval, resolution
                     claims, retroactive hitl_pause emission
  _telemetry.py    ← Telemetry (instance-scoped OTel provider/spans, fail-open) +
                     OTelCallbackHandler (LangChain events → OTel spans, GenAI semconv)
  _guardrail.py    ← evaluate_guardrail() via Claude tool-use (keyword-scan fallback; fail-closed)
  server.py        ← create_app(sdk, graph, token): the focused graph-runtime (FastAPI, bearer auth)
  cli.py           ← `unistack serve module:builder …`; discovers a sibling UNISTACK_CONFIG
pyproject.toml  requirements.txt  README.md
tests/test_guardrail.py  tests/test_telemetry.py  tests/test_server.py  tests/test_cli.py
```

## MongoDB — what this writes (and cleans up)

Database `unistack` (configurable). The SDK writes the **durable checkpointer** collections
(`checkpoints`, `checkpoint_writes`) — LangGraph's persisted graph state — plus
**`hitl_resolutions`**: one tiny doc per pause, unique on `(activity_id, checkpoint_id)`.
It is three things at once: the per-pause resolution **lock** (exactly one resolver wins;
duplicates become no-ops), the **pending-approvals index** (`status: "pending"`, with node,
message, workflow, opened_at — what unistack-api lists), and the **audit record** (decision,
resolved_by, resolved_at, plus the pausing leg's OTel `trace_id`/`span_id` for deep-linking
into the tracing backend). It is **not** a queue — nothing polls it; resolution is
request-driven. Resolution docs are kept after an activity terminates (bounded: one per pause).

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
`OTEL_EXPORTER_OTLP_HEADERS` / `OTEL_SERVICE_NAME` / `UNISTACK_API_TOKEN` and passes them in.
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
venv/bin/python -m pip install -e ".[server]"
PYTHONPATH=. venv/bin/python -m pytest tests/ -v   # needs MongoDB on localhost:27017
```

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

## Roadmap

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
  mid-run (that is what guards/reviews are). Measurement/scoring (compliance, revenue, quality,
  KRA/ROI) is a **pure function of the completed trace** → belongs in **unistack-brain**, offline,
  fully customizable per client. Keeping the runtime lean is deliberate. (If a score must influence
  control flow, that is a guard/router, not an evaluator.)
- **A "Tables" data abstraction — rejected.** The LangGraph author can already read/write any
  Mongo/Postgres table directly from their own nodes (plain Python); wrapping that would be scope
  creep and would violate the thin-latch-on principle. The only structured store UniStack owns is
  the governance/HITL audit trail, which already exists (`hitl_resolutions` + the OTel traces).
