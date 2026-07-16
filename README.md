# UniStack SDK

Add **guardrails** and **durable human-in-the-loop (HITL)** to an existing LangGraph agent —
without changing a single node. You hand UniStack your `StateGraph` builder plus a map of which
nodes to guard or review; it compiles the graph with static breakpoints **and a durable
checkpointer**, then drives the pauses. An activity can pause for a human and resume later, in a
different process, after a restart.

## Install

```bash
pip install -e ".[server]"          # from a clone (server extra = the `unistack serve` runtime)
# or: pip install "git+https://github.com/hitarthk9/unistack-sdk.git#egg=unistack[server]"
```

Requires Python ≥ 3.10 and a reachable MongoDB.

## Quickstart

All configuration is passed explicitly at init — the SDK never reads your environment. Your app
supplies the values (commonly from its own `.env`):

```python
import os
from dotenv import load_dotenv
load_dotenv()                              # your app loads its own env

from unistack import UniStack
from my_app.graph import builder           # your existing, untouched StateGraph

sdk = UniStack.init(
    workflow="content",
    mongo_uri=os.environ["MONGO_URI"],
    anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),      # LLM guardrail judge
    otel_endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"),  # optional: OTLP tracing
    otel_headers=os.environ.get("OTEL_EXPORTER_OTLP_HEADERS"),    # e.g. Langfuse Basic auth
    context="Brand voice: professional. No unverified medical or financial claims.",
)

graph = sdk.compile(
    builder,
    guards={"generate": "No unverified medical or financial claims."},  # LLM-judged after node
    reviews=["publish"],                                                 # unconditional sign-off
)

# Local dev — blocks and asks for each decision on the terminal:
result = sdk.run(graph, {"topic": "a new productivity app"})
print(result.status)   # "completed" | "hitl_rejected"
```

For production, don't block — **start** and later **resume** (state is durable, so these can be
different requests / processes):

```python
r = sdk.start(graph, {"topic": "..."})              # -> status "paused" | "completed"
r = sdk.resume(graph, r.activity_id, "approved")    # continue; may pause again or complete
```

…or just serve the graph as a durable runtime — no boilerplate:

```bash
UNISTACK_API_TOKEN=<secret> unistack serve my_app.graph:builder --workflow content \
  --guard "generate=No unverified claims." --review publish
# POST /activities  ·  POST /activities/{id}/resolve   (Authorization: Bearer <secret>)
```

Long policy text doesn't have to live on the command line — declare a plain `UNISTACK_CONFIG`
dict next to `builder` (still zero `unistack` import) and the deploy command collapses to
`unistack serve my_app.graph:builder`, no flags. See [CLAUDE.md](CLAUDE.md) for the convention.

- **Guard** — after the node runs, an LLM judges its output against the policy. Pass → continue;
  breach → a HITL pause. **Fail-closed**: if the judge itself errors, the output pauses for a
  human instead of slipping through.
- **Review** — an unconditional human sign-off after the node.
- **Durable** — graph state is persisted by a MongoDB checkpointer; a paused activity survives a
  restart and can be resumed by any process. Each pause is resolved **exactly once** (atomic
  claim), so concurrent approvals can't double-advance a graph — safe to run multiple replicas.
- **OpenTelemetry** — pure OTLP tracing, wireable to self-hosted Langfuse, a collector, or any
  hyperscaler backend. Each start/resume leg is one trace; every trace of an activity shares
  `session.id = activity_id` (Langfuse groups them as a session). The human wait is a
  `hitl_pause` span emitted at resolve time into the pausing leg's trace. Pending approvals are
  listed from MongoDB (`hitl_resolutions`), not from the tracing backend. Omit the endpoint and
  tracing stays off. Tracing is instance-scoped — the SDK never reads or writes environment
  variables.

See [CLAUDE.md](CLAUDE.md) for the full reference (start/resume, the runtime, the span model,
hard constraints).
