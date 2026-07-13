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

    def fake_init(**kwargs):
        captured["workflow"] = kwargs["workflow"]
        captured["context"] = kwargs["context"]
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
