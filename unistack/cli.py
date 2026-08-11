"""
`unistack serve module:builder ...` — deploy a graph as a durable HITL runtime with no
hand-written init/compile boilerplate. Mirrors how `langgraph` serves a graph.

The CLI is the consuming app here: it reads its own environment (MONGO_URI,
ANTHROPIC_API_KEY, OTEL_EXPORTER_OTLP_ENDPOINT / OTEL_EXPORTER_OTLP_TRACES_ENDPOINT,
OTEL_EXPORTER_OTLP_HEADERS, OTEL_SERVICE_NAME, and the UNISTACK_AUTH_MODE /
UNISTACK_OIDC_* / UNISTACK_SCOPE_* / UNISTACK_API_TOKEN / UNISTACK_DEV_IDENTITY /
UNISTACK_TOKEN_SCOPES auth vars) and passes the values explicitly into UniStack — the SDK
itself still reads no environment.

Auth is required. In the default `oidc` mode the JWKS URL, issuer and audience must all be
supplied or the runtime refuses to boot; `--auth-mode token` is a local-dev fallback. There
is no way to serve without authentication.

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


def _build_auth(args, AuthConfig):
    """
    Auth settings come from the environment, like MONGO_URI and every OTEL_* var; only the
    two you actually type in local dev are flags. Misconfiguration EXITS rather than falling
    back to an open runtime — refusing to boot is what makes "no unauthenticated mode" real.
    """
    env = os.environ.get
    mode = (args.auth_mode or env("UNISTACK_AUTH_MODE") or "oidc").lower()
    try:
        if mode == "oidc":
            return AuthConfig.oidc(jwks_url=env("UNISTACK_OIDC_JWKS_URL", ""),
                                   issuer=env("UNISTACK_OIDC_ISSUER", ""),
                                   audience=env("UNISTACK_OIDC_AUDIENCE", ""))
        if mode == "token":
            return AuthConfig.static_token(token=args.token or env("UNISTACK_API_TOKEN", ""),
                                           identity=env("UNISTACK_DEV_IDENTITY", "dev@local"),
                                           scopes=env("UNISTACK_TOKEN_SCOPES") or None)
    except ValueError as exc:
        sys.exit(f"[UniStack] cannot start: {exc}.\n"
                 "  Set UNISTACK_OIDC_JWKS_URL / _ISSUER / _AUDIENCE,\n"
                 "  or run local dev with:  --auth-mode token --token <secret>")
    sys.exit(f"[UniStack] unknown --auth-mode {mode!r} (expected 'oidc' or 'token')")


def _serve(args) -> None:
    from unistack import UniStack
    from unistack.server import AuthConfig, create_app
    import uvicorn

    # Resolve auth first: a misconfiguration should exit before any Mongo/LLM setup work.
    auth = _build_auth(args, AuthConfig)
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

    # flush=True: stdout is block-buffered when piped to a container log, and an auth
    # warning nobody sees until shutdown is worthless.
    auth_line = (f"auth=oidc issuer={auth.issuer} aud={','.join(auth.audience)}"
                 if auth.mode == "oidc" else
                 f"auth=token (LOCAL DEV ONLY — identity is NOT verified; every resolution "
                 f"is attributed to '{auth.identity}') scopes={sorted(auth.token_scopes)}")
    print(f"[UniStack] serving '{workflow}' from {args.builder} (guards={list(guards)}, "
          f"reviews={reviews}) on {args.host}:{args.port}\n[UniStack] {auth_line}", flush=True)

    try:
        uvicorn.run(create_app(sdk, graph, auth=auth), host=args.host, port=args.port)
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
    # Auth settings are env-driven, like MONGO_URI and OTEL_*; only these two are flags,
    # because they are what you type by hand in local dev.
    serve.add_argument("--auth-mode", choices=("oidc", "token"), default=None,
                       help="'oidc' (default) validates JWTs against UNISTACK_OIDC_JWKS_URL / "
                            "_ISSUER / _AUDIENCE; 'token' is a LOCAL-DEV static bearer token. "
                            "Env: UNISTACK_AUTH_MODE. There is no unauthenticated mode.")
    serve.add_argument("--token", default=None,
                       help="Static bearer token for --auth-mode token. Env: UNISTACK_API_TOKEN "
                            "(see also UNISTACK_DEV_IDENTITY, UNISTACK_TOKEN_SCOPES)")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
