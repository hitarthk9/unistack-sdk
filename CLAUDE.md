# UniStack SDK

## What this is

The UniStack SDK is a Python library that instruments LangGraph agents with business observability, guardrails, and human-in-the-loop (HITL) capabilities. Agent builders import this package and add three decorators — nothing else changes in their agent code.

This is a standalone pip-installable package. It has no dependency on other UniStack repos.

## Public API

```python
from unistack import UniStack

sdk = UniStack.init("mongodb://localhost:27017", workflow="my-workflow")

@sdk.node                                     # trace any LangGraph node
def my_node(state): ...

@sdk.node
@sdk.guardrail("Policy text")                 # LLM-evaluated policy check (apply BELOW @sdk.node)
def checked_node(state): ...

@sdk.node
def approval_node(state):
    sdk.hitl("Needs VP sign-off", data={...}) # trigger HITL pause
    return {}

graph = builder.compile(checkpointer=sdk.checkpointer)
result = sdk.run(graph, initial_state, run_id="2026-06-29")
# result.activity_id  → "my-workflow-2026-06-29"
# result.status       → "completed" | "hitl_rejected" | "guardrail_breached" | "failed"
```

`sdk.run()` is **blocking** — it polls `unistack.hitl_queue` during HITL pauses and resumes automatically when the HITL API records a human decision.

## File structure

```
unistack/
  __init__.py     ← UniStack class (entire public surface)
  _exporter.py    ← MongoDBSpanExporter — writes raw spans to MongoDB
  _guardrail.py   ← GuardrailBreached exception + evaluate_guardrail() via Claude Haiku
pyproject.toml    ← package metadata for pip install -e .
requirements.txt  ← same deps as pyproject.toml
```

## MongoDB — what this writes

Database: `unistack` (configurable via `db_name` param)

| Collection | Written by | Purpose |
|---|---|---|
| `unistack.spans` | `_exporter.py` | One doc per node execution — raw OTel span |
| `unistack.hitl_queue` | `sdk.run()` | One doc per HITL/guardrail pause — polled by SDK, resolved by HITL API |
| `unistack.checkpoints` | LangGraph MongoDBSaver | Transient graph state during HITL wait — auto-deleted after activity completes |
| `unistack.checkpoint_writes` | LangGraph MongoDBSaver | Transient — auto-deleted after activity completes |

Spans shape (written by `_exporter.py`):
```
trace_id, span_id, parent_span_id, name, activity_id, workflow,
start_time, end_time, duration_ms, status, attributes
```

Status values: `completed` | `failed` | `hitl_pending` | `guardrail_breached`

## Environment variables

| Var | Required | Default | Purpose |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | No | — | LLM guardrail evaluation via Claude Haiku; falls back to keyword scan without it |
| `MONGO_URI` | Yes | `mongodb://localhost:27017` | Passed explicitly to `UniStack.init()` |

## How to install

```bash
# Dev (editable — changes are live immediately):
pip install -e .

# From another repo that depends on this SDK:
pip install -e ../unistack-sdk

# Future (once published):
pip install git+https://github.com/org/unistack-sdk.git
```

## How to set up a dev environment

```bash
python3.13 -m venv venv
venv/bin/python -m pip install -e .
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

## Hard constraints

1. No LLM inference for deterministic computation — KRA arithmetic is pure Python.
2. Never replace an existing OTel TracerProvider — attach to it.
3. Activity IDs are human-readable: `{workflow}-{run_id}`. Never UUID.
4. A LangGraph `GraphInterrupt` must not mark a span as failed — HITL pause is not an error.
5. Guardrails use LLM evaluation — this is policy enforcement, not deterministic computation.

## Decorator order matters

`@sdk.node` must be the **outer** decorator; `@sdk.guardrail` must be **inner**:

```python
@sdk.node           # outer — catches GuardrailBreached from the inner wrapper
@sdk.guardrail("policy")   # inner — evaluates output, raises or interrupts on breach
def my_node(state): ...
```

## Downstream consumers of spans

The `unistack.spans` collection is read by two other repos:
- `unistack-assembly` — assembles spans into nested activity documents
- `unistack-api` — surfaces HITL queue items from `unistack.hitl_queue`

This SDK has **no dependency** on those repos.
