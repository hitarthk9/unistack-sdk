# UniStack SDK

## What this is

The UniStack SDK adds **guardrails** and **durable human-in-the-loop (HITL)** to an existing
LangGraph agent — without the agent author changing a single node. You hand UniStack your
`StateGraph` builder plus a map of which nodes to guard/review; it compiles the graph with static
breakpoints **and a durable checkpointer**, then drives the pauses.

HITL is **durable and non-blocking**: graph state is persisted, so an activity can pause for a
human and be resumed later — in a different process, after a restart. There is no in-memory
blocking and no Mongo queue; a tiny `hitl_resolutions` collection acts as the per-pause
resolution **lock + audit stub** (exactly one resolver wins a pause). Standalone
pip-installable package; no dependency on other repos.

## The whole integration (unchanged for the author)

```python
from unistack import UniStack
from my_app.graph import builder          # your existing, untouched StateGraph

sdk   = UniStack.init(workflow="content", mongo_uri="mongodb://localhost:27017",
                      anthropic_api_key="sk-ant-...", langsmith_api_key="ls-...",
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
sdk.close()                                                # or use `with UniStack.init(...) as sdk:`

# RunResult (dataclass): .activity_id  .state  .node  .message
#   .status: "completed" | "paused" | "hitl_rejected" | "not_found" | "failed"
```

### `UniStack.init()` parameters

| Param | Required | Purpose |
|---|---|---|
| `workflow` | Yes | Workflow name; prefix of the activity id `{workflow}-{run_id}`, and the LangSmith project default |
| `mongo_uri` | Yes | MongoDB — backs the **durable checkpointer** (graph state) + `hitl_resolutions` |
| `anthropic_api_key` | No | LLM guardrail judge; keyword-scan fallback when omitted. Judge failures fail **closed** → HITL pause |
| `langsmith_api_key` | No | Enables tracing **and the HITL pending-index / audit** (see below) |
| `langsmith_project` | No | LangSmith project; defaults to `workflow` |
| `context` | No | Business-domain text for the guardrail judge |
| `db_name` | No | Mongo database (default `unistack`) |
| `guardrail_model` | No | Judge model (default `claude-haiku-4-5-20251001`) |
| `checkpointer` | No | Override the default `MongoDBSaver` (e.g. a Postgres saver) |

## How start / resume work (durable, request-driven)

```
start(graph, initial_state):
  advance: stream segments; every reached GUARD in the segment is judged inline (pass → keep going).
  pause at a guard breach or a review node → record a pending resolution + open a hitl_pause span
  → return status "paused".
  reach END → return "completed".

resume(graph, activity_id, decision):     # triggered by the human's decision, in any process
  load the persisted checkpoint (thread_id = activity_id).
  CLAIM the pause's resolution atomically in hitl_resolutions — exactly one resolver wins;
  a concurrent/repeated resolve is a recorded no-op, never a second advance.
  close the hitl_pause span (records decision + duration).
  reject → "hitl_rejected".  approve → advance to the next pause or END.
  unknown activity_id → "not_found";  already-finalized activity → "completed" (no-op).
```

Resuming a static breakpoint uses `graph.invoke(None, config)` (not `Command(resume=…)`, which is
for dynamic `interrupt()`). Because state is durable, `start` and `resume` can be different
requests, processes, or a process that restarted in between.

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

Install with the server extra: `pip install "unistack[server]"` (adds fastapi + uvicorn).
**Auth:** with `--token` / `UNISTACK_API_TOKEN` set, both POST endpoints require
`Authorization: Bearer <token>` (401 otherwise); serving without one prints a loud warning —
an open runtime lets anyone who can reach the port start and approve activities. `/health`
stays open. Everything read-only (listing pending approvals, fetching a thread) is **not**
here — it comes from LangSmith (see below). Self-host anywhere (Azure Container Apps, etc.)
with a managed Mongo. **Scaling:** state is durable and pause resolution is claim-based, so
multiple uvicorn workers / replicas behind a load balancer are safe.

## LangSmith — tracing + the HITL index

Pass `langsmith_api_key` (+ optional `langsmith_project`). Two roles:

1. **Tracing.** Every activity is **one LangSmith thread** keyed by `activity_id` (the SDK stamps
   `metadata.thread_id` on the graph config, `guardrail_eval`, and `hitl_pause`). Uniform whether
   or not it pauses — a plain workflow is a one-trace thread; a HITL one has several.
2. **Pending index + audit.** A `hitl_pause` span spans the real human wait: `start`/`resume`
   **open** it (posted, un-ended) → **an open `hitl_pause` = a pending approval**; the next
   `resume` **closes** it with the decision → that's the audit, and its duration is the wait time.
   Spans are addressed by a deterministic id (`uuid5(activity_id, checkpoint_id)`) so open/close
   never race.

**Fetch pending approvals** (dashboard / API, no SDK):
`list_runs(project, filter='eq(name,"hitl_pause")')` → keep those with `end_time is None`.
**Fetch a thread:** `list_runs(project, filter='and(eq(metadata_key,"thread_id"),eq(metadata_value,"<activity_id>"))')`.

The pending *list* requires LangSmith enabled (it is the index). State and resume never depend on
it — only discovery does; if LangSmith blips, nothing is lost (the checkpointer owns state).

## File structure

```
unistack/
  __init__.py      ← exports UniStack, RunResult, UniStackError; NullHandler on the logger
  core.py          ← UniStack: init, compile, start, resume, run, guard eval, resolution
                     claims, hitl_pause spans
  _guardrail.py    ← evaluate_guardrail() via Claude tool-use (keyword-scan fallback; fail-closed)
  server.py        ← create_app(sdk, graph, token): the focused graph-runtime (FastAPI, bearer auth)
  cli.py           ← `unistack serve module:builder … --token …`
pyproject.toml  requirements.txt  README.md  tests/test_guardrail.py  tests/test_server.py
```

## MongoDB — what this writes (and cleans up)

Database `unistack` (configurable). The SDK writes the **durable checkpointer** collections
(`checkpoints`, `checkpoint_writes`) — LangGraph's persisted graph state — plus
**`hitl_resolutions`**: one tiny doc per pause, unique on `(activity_id, checkpoint_id)`.
It is the per-pause resolution **lock** (exactly one resolver wins; duplicates become no-ops)
and the disambiguation record (resolving an unknown id → `not_found`, a finalized one →
completed no-op). It is **not** a queue — the pending list + audit still live in LangSmith.
Resolution docs are kept after an activity terminates (bounded: one per pause).

**Retention.** The moment an activity reaches a terminal outcome (completed or rejected), its
thread is deleted via `MongoDBSaver.delete_thread(activity_id)` — nothing will ever resume from it
again. So at steady state Mongo holds working-state only for genuinely in-flight / paused
activities; nothing lingers after completion. Cleanup is best-effort (a Mongo hiccup logs and
leaves docs) and never affects the returned status/state; LangSmith history is never touched.

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

**None read, none written.** All config is constructor params; LangSmith tracing is
instance-scoped (an explicit `Client` + `LangChainTracer` in the graph config — no
`LANGSMITH_*` globals). The `unistack serve` CLI, acting as the consuming app, reads
`MONGO_URI` / `ANTHROPIC_API_KEY` / `LANGSMITH_*` / `UNISTACK_API_TOKEN` and passes them in.

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
6. State lives in the durable checkpointer; LangSmith is discovery/audit only. Resume must never
   depend on LangSmith being reachable.
7. A terminal activity's checkpoints are deleted (best-effort); read final state BEFORE deleting.
   Cleanup failures must never change the returned status/state.
8. A pause is resolved exactly once: `resume` must win the `hitl_resolutions` claim before
   advancing; losers return a no-op, unknown ids return `not_found`.
9. The SDK reads no environment variables and writes none; tracing is instance-scoped.
10. `langgraph` stays a version RANGE (`>=1.2,<2.0`), never a pin — the SDK must install
    alongside whatever LangGraph the consumer's agent already uses.
