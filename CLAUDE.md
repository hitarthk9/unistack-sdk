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

sdk   = UniStack.init("mongodb://localhost:27017", workflow="content",
                      context="Brand voice: professional, no unverified claims.")
graph = sdk.compile(builder,
                    guards={"generate": "No unverified medical or financial claims."},
                    reviews=["publish"])
result = sdk.run(graph, {"topic": "..."}, run_id="2026-07-09")
# result.status → "completed" | "hitl_rejected" | "failed"
```

Your nodes are plain functions. UniStack never asks you to import it inside them.

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
sdk = UniStack.init(
    "mongodb://localhost:27017",
    workflow="my-workflow",
    context="Business domain text for guardrail evaluation.",  # optional
    context_file="context/my-workflow.yaml",                   # alternative to context=
)

graph = sdk.compile(builder, guards={"node": "policy"}, reviews=["node2"])
result = sdk.run(graph, initial_state, run_id="2026-07-09")

# Also available:
sdk.evaluate("policy text", output_str)   # {"passed": bool, "reason": str} — raw guard check
sdk.checkpointer                          # MongoDBSaver (used internally by compile)
```

`sdk.run()` is **blocking** — it polls `unistack.hitl_queue` during pauses and resumes
automatically when the HITL API records a human decision. On rejection it does **not** resume;
the graph is abandoned and its checkpoint deleted.

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

The LLM judge works best knowing the business domain. Pass it once at init:

```python
sdk = UniStack.init(..., context="B2B wholesale, India only. Reject sanctioned regions.")
sdk = UniStack.init(..., context_file="context/my-workflow.yaml")  # uses `guardrail_context:` key
```

## File structure

```
unistack/
  __init__.py      ← exports UniStack, RunResult
  core.py          ← the whole UniStack class (init, compile, run, gate, hitl helpers)
  _guardrail.py    ← evaluate_guardrail() via Claude Haiku (keyword-scan fallback)
  config.py        ← GUARDRAIL_MODEL
pyproject.toml
requirements.txt
tests/test_guardrail.py
```

## MongoDB — what this writes

Database: `unistack` (configurable via `db_name`)

| Collection | Written by | Purpose |
|---|---|---|
| `unistack.hitl_queue` | `sdk.run()` | One doc per pause (guard breach or review) |
| `unistack.checkpoints` | LangGraph MongoDBSaver | Transient — cleared at run start/end |
| `unistack.checkpoint_writes` | LangGraph MongoDBSaver | Transient — cleared at run start/end |

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

| Var | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | No | LLM guardrail judge via Claude Haiku; falls back to keyword scan |
| `UNISTACK_GUARDRAIL_MODEL` | No | Override the judge model (default `claude-haiku-4-5-20251001`) |

## Install & test

```bash
python3.13 -m venv venv
venv/bin/python -m pip install -e .
PYTHONPATH=. venv/bin/python -m pytest tests/ -v   # needs MongoDB on localhost:27017
```

## Hard constraints

1. Never modify the author's graph topology — guards/reviews are static breakpoints only.
2. Activity IDs are human-readable: `{workflow}-{run_id}`. Never UUID.
3. A HITL pause is not an error — never mark the run failed for a pause.
4. Guardrails use LLM evaluation — policy enforcement, not deterministic computation.
5. On rejection, `run()` does NOT resume — the graph is abandoned and its checkpoint deleted.
