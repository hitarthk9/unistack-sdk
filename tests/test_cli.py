"""
Tests for `unistack serve`'s CLI: UNISTACK_CONFIG auto-discovery, CLI-flag merge/override
semantics, and backward compatibility with builder-only modules (no config present).
"""

import sys
import types

import pytest

from unistack.cli import _load_builder_and_config, main


def _install_module(name: str, **attrs) -> types.ModuleType:
    """Register an in-memory module so 'name:attr' import specs resolve without real files."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


@pytest.fixture(autouse=True)
def _cleanup_modules():
    yield
    for name in ("fake_agent_with_config", "fake_agent_no_config"):
        sys.modules.pop(name, None)


# ── _load_builder_and_config ────────────────────────────────────────────────────

def test_discovers_sibling_unistack_config():
    _install_module("fake_agent_with_config", builder="the-builder-object",
                    UNISTACK_CONFIG={"workflow": "content", "guards": {"gen": "policy"},
                                     "reviews": ["refine"], "context": "brand voice"})
    builder, config = _load_builder_and_config("fake_agent_with_config:builder")
    assert builder == "the-builder-object"
    assert config == {"workflow": "content", "guards": {"gen": "policy"},
                      "reviews": ["refine"], "context": "brand voice"}


def test_missing_config_returns_empty_dict_backward_compatible():
    _install_module("fake_agent_no_config", builder="the-builder-object")
    builder, config = _load_builder_and_config("fake_agent_no_config:builder")
    assert builder == "the-builder-object"
    assert config == {}


def test_bad_spec_without_colon_exits():
    with pytest.raises(SystemExit):
        _load_builder_and_config("no_colon_here")


# ── _serve merge semantics (via main(), mocking UniStack/uvicorn) ──────────────

def _run_serve_argv(argv, monkeypatch, patch_uvicorn):
    """Run `unistack serve <argv>`, capturing the sdk.compile() guards/reviews/context/workflow."""
    captured = {}

    class FakeSDK:
        def compile(self, builder, guards=None, reviews=None):
            captured["guards"] = guards
            captured["reviews"] = reviews
            return "compiled-graph"

        def close(self):
            captured["closed"] = True

    def fake_init(**kwargs):
        captured["workflow"] = kwargs["workflow"]
        captured["context"] = kwargs["context"]
        captured["init_kwargs"] = kwargs
        return FakeSDK()

    monkeypatch.setattr("unistack.UniStack.init", staticmethod(fake_init))
    monkeypatch.setattr("unistack.server.create_app", lambda sdk, graph, token=None: "app")
    monkeypatch.setattr(patch_uvicorn, "run", lambda app, host, port: None)
    monkeypatch.setattr(sys, "argv", ["unistack", "serve", *argv])
    main()
    return captured


def test_serve_uses_config_with_zero_flags(monkeypatch):
    import uvicorn
    _install_module("fake_agent_with_config", builder="b",
                    UNISTACK_CONFIG={"workflow": "content", "guards": {"gen": "policy"},
                                     "reviews": ["refine"], "context": "brand voice"})
    captured = _run_serve_argv(["fake_agent_with_config:builder"], monkeypatch, uvicorn)
    assert captured["workflow"] == "content"
    assert captured["guards"] == {"gen": "policy"}
    assert captured["reviews"] == ["refine"]
    assert captured["context"] == "brand voice"


def test_serve_cli_flags_override_and_merge_with_config(monkeypatch):
    import uvicorn
    _install_module("fake_agent_with_config", builder="b",
                    UNISTACK_CONFIG={"workflow": "content", "guards": {"gen": "old-policy"},
                                     "reviews": ["refine"], "context": "old context"})
    captured = _run_serve_argv([
        "fake_agent_with_config:builder",
        "--workflow", "override-workflow",
        "--guard", "gen=new-policy",       # overrides the config's 'gen' entry
        "--guard", "publish=extra-policy", # additive
        "--review", "publish",             # merges into reviews
        "--context", "new context",
    ], monkeypatch, uvicorn)
    assert captured["workflow"] == "override-workflow"
    assert captured["guards"] == {"gen": "new-policy", "publish": "extra-policy"}
    assert captured["reviews"] == ["publish", "refine"]
    assert captured["context"] == "new context"


def test_serve_no_config_pure_cli_flags_backward_compatible(monkeypatch):
    import uvicorn
    _install_module("fake_agent_no_config", builder="b")
    captured = _run_serve_argv([
        "fake_agent_no_config:builder", "--workflow", "content",
        "--guard", "gen=policy", "--review", "refine",
    ], monkeypatch, uvicorn)
    assert captured["workflow"] == "content"
    assert captured["guards"] == {"gen": "policy"}
    assert captured["reviews"] == ["refine"]
    assert captured["context"] is None


def test_serve_missing_workflow_everywhere_exits_clearly(monkeypatch):
    import uvicorn
    _install_module("fake_agent_no_config", builder="b")
    with pytest.raises(SystemExit, match="workflow is required"):
        _run_serve_argv(["fake_agent_no_config:builder"], monkeypatch, uvicorn)


# ── OTel env plumbing: CLI reads the standard vars and passes them in ───────────

def test_serve_passes_otel_env_into_init_and_closes_sdk(monkeypatch):
    import uvicorn
    _install_module("fake_agent_no_config", builder="b")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "Authorization=Basic abc")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "my-service")
    captured = _run_serve_argv(
        ["fake_agent_no_config:builder", "--workflow", "content"], monkeypatch, uvicorn)
    kwargs = captured["init_kwargs"]
    assert kwargs["otel_endpoint"] == "http://collector:4318"
    assert kwargs["otel_headers"] == "Authorization=Basic abc"
    assert kwargs["otel_service_name"] == "my-service"
    assert captured["closed"] is True            # spans flush on shutdown


def test_serve_traces_endpoint_wins_over_generic(monkeypatch):
    import uvicorn
    _install_module("fake_agent_no_config", builder="b")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://generic:4318")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://traces:4318/v1/traces")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    captured = _run_serve_argv(
        ["fake_agent_no_config:builder", "--workflow", "content"], monkeypatch, uvicorn)
    kwargs = captured["init_kwargs"]
    assert kwargs["otel_endpoint"] == "http://traces:4318/v1/traces"
    assert kwargs["otel_service_name"] == "unistack-content"   # default from workflow
