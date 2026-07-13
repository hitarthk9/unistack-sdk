"""
`unistack serve module:builder ...` — deploy a graph as a durable HITL runtime with no
hand-written init/compile boilerplate. Mirrors how `langgraph` serves a graph.

The CLI is the consuming app here: it reads its own environment (MONGO_URI,
ANTHROPIC_API_KEY, LANGSMITH_API_KEY, LANGSMITH_PROJECT, UNISTACK_API_TOKEN) and
passes the values explicitly into UniStack — the SDK itself still reads no environment.
"""

import argparse
import importlib
import os
import sys


def _load_builder(spec: str):
    """Import a StateGraph builder from 'package.module:attribute'."""
    if ":" not in spec:
        sys.exit("builder must be 'module:attribute', e.g. demo.agent:builder")
    module_path, attr = spec.split(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _serve(args) -> None:
    from unistack import UniStack
    from unistack.server import create_app
    import uvicorn

    builder = _load_builder(args.builder)
    guards = dict(g.split("=", 1) for g in (args.guard or []))
    reviews = list(args.review or [])

    sdk = UniStack.init(
        workflow=args.workflow,
        mongo_uri=os.environ.get("MONGO_URI", "mongodb://localhost:27017"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        langsmith_api_key=os.environ.get("LANGSMITH_API_KEY") or None,
        langsmith_project=os.environ.get("LANGSMITH_PROJECT") or None,
        context=args.context,
    )
    graph = sdk.compile(builder, guards=guards, reviews=reviews)
    token = args.token or os.environ.get("UNISTACK_API_TOKEN") or None
    if not token:
        print("[UniStack] WARNING: serving WITHOUT authentication — anyone who can reach "
              "this port can start and approve activities. Set --token or UNISTACK_API_TOKEN.")
    print(f"[UniStack] serving '{args.workflow}' from {args.builder} "
          f"(guards={list(guards)}, reviews={reviews}, "
          f"auth={'bearer token' if token else 'OFF'}) on {args.host}:{args.port}")
    uvicorn.run(create_app(sdk, graph, token=token), host=args.host, port=args.port)


def main() -> None:
    parser = argparse.ArgumentParser(prog="unistack")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Serve a compiled graph as a durable HITL runtime.")
    serve.add_argument("builder", help="StateGraph builder as 'module:attribute'")
    serve.add_argument("--workflow", required=True, help="Workflow name (project / activity prefix)")
    serve.add_argument("--guard", action="append", metavar="NODE=POLICY",
                       help="Guard a node (repeatable): --guard generate='No unverified claims'")
    serve.add_argument("--review", action="append", metavar="NODE",
                       help="Require human sign-off after a node (repeatable)")
    serve.add_argument("--context", default=None, help="Business context for the guardrail judge")
    serve.add_argument("--token", default=None,
                       help="Bearer token required on the POST endpoints "
                            "(default: UNISTACK_API_TOKEN env var; unset = no auth, with a warning)")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
