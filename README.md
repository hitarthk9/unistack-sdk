# UniStack SDK

Add **guardrails** and **human-in-the-loop (HITL)** to an existing LangGraph agent — without
changing a single node. You hand UniStack your `StateGraph` builder plus a map of which nodes to
guard or review; it compiles the graph with static breakpoints and drives the pauses.

## Install

```bash
pip install -e .          # from a clone
# or: pip install git+https://github.com/hitarthk9/unistack-sdk.git
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
    langsmith_api_key=os.environ.get("LANGSMITH_API_KEY"),      # optional: LangSmith tracing
    context="Brand voice: professional. No unverified medical or financial claims.",
)

graph = sdk.compile(
    builder,
    guards={"generate": "No unverified medical or financial claims."},  # LLM-judged after node
    reviews=["publish"],                                                 # unconditional sign-off
)

result = sdk.run(graph, {"topic": "a new productivity app"}, run_id="2026-07-09")
print(result.status)   # "completed" | "hitl_rejected" | "failed"
```

- **Guard** — after the node runs, an LLM judges its output against the policy. Pass → continue
  silently; breach → open a HITL pause.
- **Review** — an unconditional human sign-off after the node.
- **LangSmith** — pass `langsmith_api_key` and every run is traced. All of an activity's
  traces (graph segments, guard checks, HITL pauses) are grouped into one **LangSmith thread**
  keyed by `activity_id`, so you fetch a run's full history with a single query — whether or not
  it paused. Omit the key and tracing stays off.

`sdk.run()` is blocking: it polls `unistack.hitl_queue` while paused and resumes when a human
records a decision (approve → continue, reject → abandon).

See [CLAUDE.md](CLAUDE.md) for the full reference (parameters, run loop internals, MongoDB
schema, hard constraints).
