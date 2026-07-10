# UniStack SDK

## What this is

The UniStack SDK adds **guardrails** and **durable human-in-the-loop (HITL)** to an existing
LangGraph agent — without the agent author changing a single node. You hand UniStack your
`StateGraph` builder plus a map of which nodes to guard/review; it compiles the graph with static
breakpoints **and a durable checkpointer**, then drives the pauses.

HITL is **durable and non-blocking**: graph state is persisted, so an activity can pause for a
human and be resumed later — in a different process, after a restart. There is no in-memory
blocking and no Mongo queue. Standalone pip-installable package; no dependency on other repos.

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
is passed explicitly** — the SDK reads no environment variables and loads no `.env`.

## Concepts

- **Guard** — `guards={node: policy}`. After the node runs, an LLM judges its output against the
  policy (+ `context`). Passed → continue silently. Breached → a HITL pause.
- **Review** — `reviews=[node]`. Unconditional human sign-off after the node. Always pauses.

Both are LangGraph **static breakpoints** (`interrupt_after`). The author's graph topology is
never modified.

## Public API

```python
sdk   = UniStack.init(...)                                  # see parameters below
graph = sdk.compile(builder, guards={"n": "policy"}, reviews=["m"])

r = sdk.start(graph, initial_state, run_id=None)           # non-blocking; run_id → unique ts
r = sdk.resume(graph, activity_id, decision, resolved_by=None)   # "approved" | "rejected"
r = sdk.run(graph, initial_state, decide=None)             # local convenience over start/resume
sdk.evaluate("policy", output_str)                         # {"passed", "reason"} — raw guard check

# RunResult: .activity_id  .status ("completed"|"paused"|"hitl_rejected"|"failed")  .node  .message
```

### `UniStack.init()` parameters

| Param | Required | Purpose |
|---|---|---|
| `workflow` | Yes | Workflow name; prefix of the activity id `{workflow}-{run_id}`, and the LangSmith project default |
| `mongo_uri` | Yes | MongoDB — backs the **durable checkpointer** (graph state) |
| `anthropic_api_key` | No | LLM guardrail judge; keyword-scan fallback when omitted |
| `langsmith_api_key` | No | Enables tracing **and the HITL pending-index / audit** (see below) |
| `langsmith_project` | No | LangSmith project; defaults to `workflow` |
| `context` | No | Business-domain text for the guardrail judge |
| `db_name` | No | Mongo database (default `unistack`) |
| `guardrail_model` | No | Judge model (default `claude-haiku-4-5-20251001`) |
| `checkpointer` | No | Override the default `MongoDBSaver` (e.g. a Postgres saver) |

## How start / resume work (durable, request-driven)

```
start(graph, initial_state):
  advance: stream segments; a reached GUARD is judged inline (pass → keep going).
  pause at a guard breach or a review node → open a hitl_pause span → return status "paused".
  reach END → return "completed".

resume(graph, activity_id, decision):     # triggered by the human's decision, in any process
  load the persisted checkpoint (thread_id = activity_id).
  close the hitl_pause span (records decision + duration).
  reject → "hitl_rejected".  approve → advance to the next pause or END.
```

Resuming a static breakpoint uses `graph.invoke(None, config)` (not `Command(resume=…)`, which is
for dynamic `interrupt()`). Because state is durable, `start` and `resume` can be different
requests, processes, or a process that restarted in between.

**Idempotency / terminal pauses:** a static breakpoint on a *terminal* node leaves
`get_state().next` empty — indistinguishable from a completed graph. So `resume` checks the
decision first (reject always halts), and treats "approve with nothing left to run" as a harmless
completed no-op. Concurrent double-*approve* of the *same* non-terminal pause within LangSmith's
ingestion window is a known small race (mitigate in the UI); a lock is a future hardening.

## Deployment — `unistack serve`

The focused **graph-runtime** is the only component that imports the graph + SDK. It exposes
`POST /activities` (start) and `POST /activities/{id}/resolve` (resume), nothing else:

```bash
unistack serve my_app.graph:builder --workflow content \
  --guard "generate=No unverified claims." --review publish --context "Brand voice: …"
```

Install with the server extra: `pip install "unistack[server]"` (adds fastapi + uvicorn).
Everything read-only (listing pending approvals, fetching a thread) is **not** here — it comes
from LangSmith (see below). Self-host anywhere (Azure Container Apps, etc.) with a managed Mongo.

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
  __init__.py      ← exports UniStack, RunResult
  core.py          ← UniStack: init, compile, start, resume, run, guard eval, hitl_pause spans
  _guardrail.py    ← evaluate_guardrail() via Claude (keyword-scan fallback)
  server.py        ← create_app(sdk, graph): the focused graph-runtime (FastAPI)
  cli.py           ← `unistack serve module:builder …`
pyproject.toml  requirements.txt  README.md  tests/test_guardrail.py
```

## MongoDB — what this writes

Database `unistack` (configurable). The SDK writes only the **durable checkpointer** collections
(`checkpoints`, `checkpoint_writes`) — LangGraph's persisted graph state. There is **no**
`hitl_queue`; the pending list + audit live in LangSmith.

## Environment variables

**None read.** All config is constructor params. The only env vars the SDK *writes* are the
`LANGSMITH_*` trio, and only when `langsmith_api_key` is passed. (The `unistack serve` CLI, acting
as the consuming app, reads `MONGO_URI` / `ANTHROPIC_API_KEY` / `LANGSMITH_*` and passes them in.)

## Install & test

```bash
python3.13 -m venv venv
venv/bin/python -m pip install -e ".[server]"
PYTHONPATH=. venv/bin/python -m pytest tests/ -v   # needs MongoDB on localhost:27017
```

## Hard constraints

1. Never modify the author's graph topology — guards/reviews are static breakpoints only.
2. Activity IDs are human-readable: `{workflow}-{run_id}`, `run_id` a UTC microsecond timestamp
   by default. Never UUID.
3. A HITL pause is not an error — `status="paused"` is normal, never `failed`.
4. Guardrails use LLM evaluation — policy enforcement, not deterministic computation.
5. On rejection, the activity is abandoned (not resumed).
6. State lives in the durable checkpointer; LangSmith is discovery/audit only. Resume must never
   depend on LangSmith being reachable.
