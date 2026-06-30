# UniStack SDK

## What this is

The UniStack SDK instruments AI agents with business observability, guardrails, and human-in-the-loop (HITL) capabilities. It is framework-agnostic: the default adapter targets LangGraph; a native adapter works with any Python pipeline (raw Anthropic/OpenAI calls, CrewAI, AutoGen, etc.).

Agent builders import this package and add three decorators — nothing else changes in their agent code.

This is a standalone pip-installable package. It has no dependency on other UniStack repos.

## Public API

```python
# Default — LangGraph adapter (backwards-compatible)
from unistack import UniStack

# Explicit adapters
from unistack.adapters.langgraph import UniStack        # same as above
from unistack.adapters.native import UniStack           # any framework, no LangGraph

sdk = UniStack.init(
    "mongodb://localhost:27017",
    workflow="my-workflow",
    context="Business domain text for guardrail evaluation.",  # optional
    context_file="context/my-workflow.yaml",                   # alternative to context=
)

@sdk.node                                     # trace any node/step
def my_node(state): ...

@sdk.node
@sdk.guardrail("Policy text")                 # LLM-evaluated policy check (apply BELOW @sdk.node)
def checked_node(state): ...

@sdk.node
def approval_node(state):
    sdk.hitl("Needs VP sign-off", data={...}) # trigger HITL pause
    return {}

# LangGraph adapter:
graph = builder.compile(checkpointer=sdk.checkpointer)
result = sdk.run(graph, initial_state, run_id="2026-06-29")

# Native adapter:
result = sdk.run(my_pipeline_fn, initial_input, run_id="2026-06-29")

# result.activity_id  → "my-workflow-2026-06-29"
# result.status       → "completed" | "hitl_rejected" | "guardrail_breached" | "failed"
```

`sdk.run()` is **blocking** — it polls `unistack.hitl_queue` during HITL pauses and resumes automatically when the HITL API records a human decision.

## Guardrail context

Guardrail evaluators work best when they know the business domain. Pass it at init time:

```python
# Option A — inline string (quick)
sdk = UniStack.init(..., context="B2B wholesale, India only. Reject sanctioned regions.")

# Option B — YAML file (structured, reusable)
sdk = UniStack.init(..., context_file="context/adani-retail.yaml")
```

YAML file schema (only `guardrail_context` is used by the SDK; other fields are for documentation):
```yaml
workflow: adani-retail
guardrail_context: |
  Business: B2B wholesale platform ...
  Compliance rules: ...
  Fraud indicators: ...
```

## File structure

```
unistack/
  __init__.py        ← re-exports UniStack from adapters.langgraph (backwards-compat)
  core.py            ← UniStackCore: all framework-agnostic logic
  adapters/
    langgraph.py     ← UniStack: LangGraph nodes, interrupt/resume, MongoDBSaver
    native.py        ← UniStack: blocking HITL, direct guardrail raise, any framework
  _exporter.py       ← MongoDBSpanExporter — writes raw spans to MongoDB
  _guardrail.py      ← GuardrailBreached + evaluate_guardrail() via Claude Haiku
pyproject.toml       ← package metadata for pip install -e .
requirements.txt     ← same deps as pyproject.toml
```

## MongoDB — what this writes

Database: `unistack` (configurable via `db_name` param)

| Collection | Written by | Purpose |
|---|---|---|
| `unistack.spans` | `_exporter.py` | One doc per node execution — raw OTel span |
| `unistack.hitl_queue` | `sdk.run()` / `sdk.hitl()` | One doc per HITL/guardrail pause |
| `unistack.checkpoints` | LangGraph MongoDBSaver | Transient — auto-deleted after activity |
| `unistack.checkpoint_writes` | LangGraph MongoDBSaver | Transient — auto-deleted after activity |

Spans shape:
```
trace_id, span_id, parent_span_id, name, activity_id, workflow,
start_time, end_time, duration_ms, status, attributes
```

Status values: `completed` | `failed` | `hitl_pending` | `guardrail_breached`

## Environment variables

| Var | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | No | LLM guardrail evaluation via Claude Haiku; falls back to keyword scan |
| `MONGO_URI` | No | Passed explicitly to `UniStack.init()` — default `mongodb://localhost:27017` |

## How to install

```bash
# Dev (editable — changes are live immediately):
pip install -e .

# From another repo that depends on this SDK:
pip install -e ../unistack-sdk
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

- `unistack-assembly` — assembles spans into nested activity documents
- `unistack-api` — surfaces HITL queue items from `unistack.hitl_queue`

This SDK has **no dependency** on those repos.
