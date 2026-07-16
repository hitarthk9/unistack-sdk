"""
`unistack serve module:builder ...` — deploy a graph as a durable HITL runtime with no
hand-written init/compile boilerplate. Mirrors how `langgraph` serves a graph.

The CLI is the consuming app here: it reads its own environment (MONGO_URI,
ANTHROPIC_API_KEY, OTEL_EXPORTER_OTLP_ENDPOINT / OTEL_EXPORTER_OTLP_TRACES_ENDPOINT,
OTEL_EXPORTER_OTLP_HEADERS, OTEL_SERVICE_NAME, UNISTACK_API_TOKEN) and passes the
values explicitly into UniStack — the SDK itself still reads no environment.

Governance (workflow / guards / reviews / context) can also be declared as plain data next
to the builder — a module-level `UNISTACK_CONFIG` dict — so a deploy command doesn't need to
carry policy text (which can be long) as shell arguments. This is pure data: the module still
imports nothing from `unistack`, preserving "the author's graph is untouched." CLI flags merge
on top of `UNISTACK_CONFIG` for one-off overrides without a redeploy.
"""

import argparse
import importlib
import os
import sys


def _load_builder_and_config(spec: str):
    """
    Import 'package.module:attribute' for the StateGraph builder. Also reads a sibling
    module-level `UNISTACK_CONFIG` dict from the same module, if present — absent is fine,
    returned as {} (fully backward compatible with builder-only modules).
    """
    if ":" not in spec:
        sys.exit("builder must be 'module:attribute', e.g. demo.agent:builder")
    module_path, attr = spec.split(":", 1)
    module = importlib.import_module(module_path)
    builder = getattr(module, attr)
    config = getattr(module, "UNISTACK_CONFIG", {})
    return builder, config


def _serve(args) -> None:
    from unistack import UniStack
    from unistack.server import create_app
    import uvicorn

    builder, config = _load_builder_and_config(args.builder)

    workflow = args.workflow or config.get("workflow")
    if not workflow:
        sys.exit("workflow is required: pass --workflow, or set 'workflow' in the module's "
                 "UNISTACK_CONFIG")

    guards = dict(config.get("guards") or {})
    guards.update(g.split("=", 1) for g in (args.guard or []))   # CLI wins per-key
    reviews = sorted(set(config.get("reviews") or []) | set(args.review or []))
    context = args.context if args.context is not None else config.get("context")

    sdk = UniStack.init(
        workflow=workflow,
        mongo_uri=os.environ.get("MONGO_URI", "mongodb://localhost:27017"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        # Standard OTel env vars; the signal-specific TRACES endpoint wins when both set.
        otel_endpoint=(os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
                       or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or None),
        otel_headers=os.environ.get("OTEL_EXPORTER_OTLP_HEADERS") or None,
        otel_service_name=os.environ.get("OTEL_SERVICE_NAME") or f"unistack-{workflow}",
        context=context,
    )
    graph = sdk.compile(builder, guards=guards, reviews=reviews)
    token = args.token or os.environ.get("UNISTACK_API_TOKEN") or None
    if not token:
        print("[UniStack] WARNING: serving WITHOUT authentication — anyone who can reach "
              "this port can start and approve activities. Set --token or UNISTACK_API_TOKEN.")
    print(f"[UniStack] serving '{workflow}' from {args.builder} "
          f"(guards={list(guards)}, reviews={reviews}, "
          f"auth={'bearer token' if token else 'OFF'}) on {args.host}:{args.port}")
    try:
        uvicorn.run(create_app(sdk, graph, token=token), host=args.host, port=args.port)
    finally:
        sdk.close()      # flush buffered spans (BatchSpanProcessor) on shutdown


def main() -> None:
    parser = argparse.ArgumentParser(prog="unistack")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Serve a compiled graph as a durable HITL runtime.")
    serve.add_argument("builder", help="StateGraph builder as 'module:attribute'")
    serve.add_argument("--workflow", default=None,
                       help="Workflow name (project / activity prefix). "
                            "Falls back to the module's UNISTACK_CONFIG['workflow'] if omitted.")
    serve.add_argument("--guard", action="append", metavar="NODE=POLICY",
                       help="Guard a node (repeatable): --guard generate='No unverified claims'. "
                            "Merges with (and overrides per-key) UNISTACK_CONFIG['guards'].")
    serve.add_argument("--review", action="append", metavar="NODE",
                       help="Require human sign-off after a node (repeatable). Merges with "
                            "UNISTACK_CONFIG['reviews'].")
    serve.add_argument("--context", default=None,
                       help="Business context for the guardrail judge. Overrides "
                            "UNISTACK_CONFIG['context'] if given.")
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
