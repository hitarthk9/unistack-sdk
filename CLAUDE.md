# UniStack SDK

## What this is

The UniStack SDK adds **guardrails** and **human-in-the-loop (HITL)** to an existing LangGraph
agent — without the agent author changing a single node. You hand UniStack your `StateGraph`
builder plus a map of which nodes to guard/review; it compiles the graph with static
breakpoints and drives the pauses.

This is a standalone pip-installable package. It has no dependency on other UniStack repos.

## The whole integration (3 lines)

```python
from unistack import UniStack
from my_app.graph import builder          # your existing, untouched StateGraph

sdk   = UniStack.init(workflow="content",
                      mongo_uri="mongodb://localhost:27017",
                      anthropic_api_key="sk-ant-...",
                      langsmith_api_key="ls-...",   # optional: enables LangSmith tracing
                      context="Brand voice: professional, no unverified claims.")
graph = sdk.compile(builder,
                    guards={"generate": "No unverified medical or financial claims."},
                    reviews=["publish"])
result = sdk.run(graph, {"topic": "..."})   # run_id defaults to a unique timestamp
# result.status → "completed" | "hitl_rejected" | "failed"
```

Your nodes are plain functions. UniStack never asks you to import it inside them. **All config
is passed explicitly** — the SDK reads no environment variables and loads no `.env`. The
consuming app owns its environment and hands the values in.

## Concepts

- **Guard** — attached to a node via `guards={node: policy}`. After the node runs, an LLM acts
  as judge: it evaluates the node's output against the policy (plus the business `context`).
  - **Passed** → the graph resumes silently, no human involved.
  - **Breached** → a HITL pause opens with the breach reason; a human approves (resume) or
    rejects (halt).
- **Review** — attached via `reviews=[node]`. An **unconditional** human sign-off after the
  node, before continuing. No LLM, always pauses.

Both are just LangGraph **static breakpoints** (`interrupt_after`) under the hood. The graph
topology is never modified — no extra nodes, edges, reducers, or conditional routing.

## Public API

```python
sdk = UniStack.init(builder_config)   # see parameters below
graph = sdk.compile(builder, guards={"node": "policy"}, reviews=["node2"])
result = sdk.run(graph, initial_state)   # run_id optional; defaults to a unique timestamp

# Also available:
sdk.evaluate("policy text", output_str)   # {"passed": bool, "reason": str} — raw guard check
```

### `UniStack.init()` parameters

| Param | Required | Purpose |
|---|---|---|
| `workflow` | Yes | Workflow name; part of the activity id `{workflow}-{run_id}` and the LangSmith project default |
| `mongo_uri` | Yes | MongoDB connection string; the SDK reads/writes `hitl_queue` here directly |
| `anthropic_api_key` | No | Enables the LLM guardrail judge; falls back to a keyword scan when omitted |
| `langsmith_api_key` | No | Enables LangSmith tracing (see below) |
| `langsmith_project` | No | LangSmith project name; defaults to `workflow` |
| `context` | No | Business-domain text injected into the guardrail judge prompt |
| `db_name` | No | Mongo database name (default `unistack`) |
| `guardrail_model` | No | Judge model (default `claude-haiku-4-5-20251001`) |
| `hitl_poll_interval` | No | Seconds between HITL-queue polls while paused (default `2.0`) |

`sdk.run()` is **blocking** — it polls `unistack.hitl_queue` during pauses and resumes
automatically when a human decision is recorded. On rejection it does **not** resume; the graph
is abandoned.

## LangSmith tracing

Pass `langsmith_api_key` (and optionally `langsmith_project`) to `UniStack.init()` and every run
is traced in LangSmith. Under the hood the SDK sets
`LANGSMITH_TRACING`/`LANGSMITH_API_KEY`/`LANGSMITH_PROJECT` for the underlying LangChain tracer.
Omit the key and tracing stays off — LangSmith ships as a transitive LangGraph dependency but
stays dormant.

### One activity = one LangSmith thread

`sdk.run()` streams the graph once per breakpoint, so a HITL activity is several native traces
(one graph segment per pause, plus a `guardrail_eval` / `hitl_pause` span at each gate). Rather
than force these into one hand-stitched tree (which mixes LangChain-runtime and SDK-runtime runs
and renders badly), the SDK groups them into a single **LangSmith thread** keyed by `activity_id`
— it stamps `metadata.thread_id = activity_id` on every graph config, guardrail_eval, and
hitl_pause span (see `_thread_meta`). Each trace stays clean and native; the thread ties them
together.

This is **uniform across every workflow**: a plain workflow that never pauses is just a thread
with one trace; a HITL/guarded workflow is a thread with several. There is no "trace vs thread"
fork.

### Fetching an activity downstream (e.g. the brain)

Always fetch by thread — one query, HITL or not:

```python
from langsmith import Client
runs = Client().list_runs(
    project_name=<project>,
    filter=f'and(eq(metadata_key,"thread_id"),eq(metadata_value,"{activity_id}"))',
)
```

Filtering to `is_root=True` and sorting by `start_time` yields the activity's timeline of events
(graph segments, guard evaluations, HITL pauses); each root's children hold the detail.

## How run() works

```
compile(builder, guards, reviews) → interrupt_after=[guarded + reviewed nodes]
run():
  loop:
    stream the graph (stream_mode="updates") until the next breakpoint or END,
      capturing the last node + its output and whether a breakpoint fired
    if no breakpoint fired: done (ran to END)
    gate(last_node, last_output):
      guard  → evaluate output; passed = continue, breached = HITL pause
      review → HITL pause
    approved → resume with stream(None, config);  rejected → stop
    if the gated node was terminal (no successor): done
```

A static breakpoint surfaces as a `{"__interrupt__": ()}` chunk in the stream — this fires
even when the node's only successor is `END`, so guards/reviews on terminal nodes still pause
(unlike checking `get_state().next`, which is already empty at END). Resuming uses
`graph.stream(None, config)` — not `Command(resume=...)` (that is for dynamic `interrupt()`
calls inside nodes, which this SDK does not require).

## Guardrail context

The LLM judge works best knowing the business domain. Pass it once at init as a plain string:

```python
sdk = UniStack.init(..., context="B2B wholesale, India only. Reject sanctioned regions.")
```

## File structure

```
unistack/
  __init__.py      ← exports UniStack, RunResult
  core.py          ← the whole UniStack class (init, compile, run, gate, hitl helpers)
  _guardrail.py    ← evaluate_guardrail() via Claude (keyword-scan fallback)
pyproject.toml
requirements.txt
README.md
tests/test_guardrail.py
```

## MongoDB — what this writes

Database: `unistack` (configurable via `db_name`)

| Collection | Written by | Purpose |
|---|---|---|
| `unistack.hitl_queue` | `sdk.run()` | One doc per pause (guard breach or review) |

hitl_queue document:
```json
{
  "activity_id": "content-2026-07-09",
  "status": "pending | approved | rejected",
  "message": "Guardrail breach after 'generate': ...",
  "created_at": "ISODate",
  "resolved_at": "ISODate | null",
  "resolved_by": "string | null"
}
```

## Environment variables

**None.** The SDK reads no environment variables for configuration — everything is a
constructor parameter (see the parameters table above). The consuming app supplies the values
(often from its own `.env`). The only env vars the SDK *writes* are the `LANGSMITH_*` trio, and
only when you pass `langsmith_api_key`.

## Install & test

```bash
python3.13 -m venv venv
venv/bin/python -m pip install -e .
PYTHONPATH=. venv/bin/python -m pytest tests/ -v   # needs MongoDB on localhost:27017
```

## Hard constraints

1. Never modify the author's graph topology — guards/reviews are static breakpoints only.
2. Activity IDs are human-readable: `{workflow}-{run_id}`. Never UUID. `run_id` defaults to a
   UTC microsecond timestamp (unique + sortable) when the caller doesn't supply one.
3. A HITL pause is not an error — never mark the run failed for a pause.
4. Guardrails use LLM evaluation — policy enforcement, not deterministic computation.
5. On rejection, `run()` does NOT resume — the graph is abandoned.
