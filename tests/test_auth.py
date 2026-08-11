"""
Unit tests for the shared `unistack_auth` package — no Mongo, no FastAPI, no graph.

This is the ONLY place token verification is tested, because it is the only place it is
written. unistack-api installs the same package and tests what is actually its own concern:
that the read scope is enforced on its routers.
"""

import base64
import hashlib
import hmac
import json

import pytest

from tests.conftest import AUDIENCE, ISSUER, KID
from unistack_auth import (
    AuthConfig,
    AuthError,
    Principal,
    _extract_scopes,
    make_verifier,
    require_scope,
)


@pytest.fixture
def verifier(jwks_url):
    return make_verifier(AuthConfig.oidc(jwks_url=jwks_url, issuer=ISSUER, audience=AUDIENCE))


def _b64(raw: bytes) -> bytes:
    return base64.urlsafe_b64encode(raw).rstrip(b"=")


def _handcrafted(header: dict, payload: dict, secret: str | None = None) -> str:
    """
    Build a JWT WITHOUT pyjwt, so its own encode-side guards can't stop us forging one.
    This is what an attacker sends; pyjwt refuses to HMAC-sign with an asymmetric PEM, so
    the alg-confusion attack cannot be reproduced through the normal encode() path.
    """
    signing_input = _b64(json.dumps(header).encode()) + b"." + _b64(json.dumps(payload).encode())
    sig = (hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
           if secret is not None else b"")
    return (signing_input + b"." + _b64(sig)).decode()


# ── config validation ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kwargs, match", [
    ({"jwks_url": "u", "issuer": "i", "audience": ""}, "audience"),
    ({"jwks_url": "", "issuer": "i", "audience": "a"}, "jwks_url"),
])
def test_oidc_config_refuses_incomplete_settings(kwargs, match):
    """Incomplete config must fail loudly — never degrade to skipping validation."""
    with pytest.raises(ValueError, match=match):
        AuthConfig.oidc(**kwargs)


def test_audience_accepts_comma_separated_list(jwks_url):
    cfg = AuthConfig.oidc(jwks_url=jwks_url, issuer=ISSUER, audience="a, b")
    assert cfg.audience == ("a", "b")


# ── scope extraction ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("claims, expected", [
    ({"scp": "activity.start activity.resolve"}, {"activity.start", "activity.resolve"}),
    ({"scp": ["activity.start"]},                {"activity.start"}),
    ({"scope": "activity.read"},                 {"activity.read"}),
    ({"roles": ["Activity.Start"]},              {"activity.start"}),
    ({"scp": "a", "scope": "b", "roles": ["C"]}, {"a", "b", "c"}),
    ({},                                          set()),
])
def test_extract_scopes(claims, expected):
    assert _extract_scopes(claims) == expected


def test_scope_match_is_case_insensitive():
    p = Principal(subject="s", issuer="i", label="l", scopes=frozenset({"activity.start"}))
    assert p.has("Activity.Start")


# ── the two attacks that must never work ─────────────────────────────────────────────────

def test_hs256_token_signed_with_public_key_rejected(verifier, rsa_key):
    """Algorithm confusion: sign HS256 using the PUBLIC key as the HMAC secret."""
    token = _handcrafted(
        {"alg": "HS256", "typ": "JWT", "kid": KID},
        {"iss": ISSUER, "aud": AUDIENCE, "sub": "attacker", "iat": 0, "exp": 9999999999},
        secret=rsa_key["public_pem"],
    )
    with pytest.raises(AuthError) as exc:
        verifier.verify(f"Bearer {token}")
    assert exc.value.status == 401


def test_alg_none_rejected(verifier):
    token = _handcrafted(
        {"alg": "none", "typ": "JWT", "kid": KID},
        {"iss": ISSUER, "aud": AUDIENCE, "sub": "attacker", "iat": 0, "exp": 9999999999},
    )
    with pytest.raises(AuthError) as exc:
        verifier.verify(f"Bearer {token}")
    assert exc.value.status == 401


# ── token validation ─────────────────────────────────────────────────────────────────────

def test_valid_token_yields_principal(verifier, make_token):
    p = verifier.verify(f"Bearer {make_token()}")
    assert p.subject == "user-sub-1"
    assert p.issuer == ISSUER
    assert p.label == "approver@example.com"
    assert p.auth_mode == "oidc"
    assert p.has("activity.resolve")


def test_bearer_scheme_is_case_insensitive(verifier, make_token):
    assert verifier.verify(f"bearer {make_token()}").subject == "user-sub-1"


@pytest.mark.parametrize("header", [None, "", "Basic abc", "Bearer", "Bearer   "])
def test_missing_or_malformed_header_401(verifier, header):
    with pytest.raises(AuthError) as exc:
        verifier.verify(header)
    assert exc.value.status == 401
    assert exc.value.www_authenticate


@pytest.mark.parametrize("overrides", [
    {"exp": 1},                                  # expired
    {"nbf": 9999999999},                         # not yet valid
    {"iss": "https://evil.test"},                # wrong issuer
    {"aud": "some-other-app"},                   # wrong audience
    {"exp": None},                               # required claim missing
    {"aud": None},
])
def test_invalid_claims_401(verifier, make_token, overrides):
    with pytest.raises(AuthError) as exc:
        verifier.verify(f"Bearer {make_token(**overrides)}")
    assert exc.value.status == 401


@pytest.mark.parametrize("token_kwargs, why", [
    ({"kid": "no-such-key"}, "unknown signing key"),
    ({"sub": None},          "no subject — an unattributable audit record"),
])
def test_unusable_tokens_401(verifier, make_token, token_kwargs, why):
    with pytest.raises(AuthError) as exc:
        verifier.verify(f"Bearer {make_token(**token_kwargs)}")
    assert exc.value.status == 401, why


def test_garbage_token_401(verifier):
    with pytest.raises(AuthError) as exc:
        verifier.verify("Bearer not-a-jwt")
    assert exc.value.status == 401


def test_error_detail_does_not_leak_internals(verifier, make_token):
    with pytest.raises(AuthError) as exc:
        verifier.verify(f"Bearer {make_token(exp=1)}")
    assert exc.value.detail == "token expired"          # coarse category, not a stack trace


# ── identity mapping ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("overrides, expected", [
    ({"email": None, "preferred_username": "pu@x"},   "pu@x"),   # falls through email
    ({"email": None},                                 "user-sub-1"),  # ...all the way to sub
])
def test_label_claim_precedence(verifier, make_token, overrides, expected):
    assert verifier.verify(f"Bearer {make_token(**overrides)}").label == expected


def test_subject_falls_back_to_oid(verifier, make_token):
    """Entra's `sub` is pairwise per-application; `oid` is the tenant-stable id."""
    assert verifier.verify(f"Bearer {make_token(sub=None, oid='oid-1')}").subject == "oid-1"


def test_scopes_read_from_roles_claim(verifier, make_token):
    p = verifier.verify(f"Bearer {make_token(('activity.resolve',), claim='roles')}")
    assert p.has("activity.resolve") and not p.has("activity.start")


# ── scope enforcement ────────────────────────────────────────────────────────────────────

def test_require_scope_raises_403_with_insufficient_scope():
    p = Principal(subject="s", issuer="i", label="l", scopes=frozenset({"activity.start"}))
    with pytest.raises(AuthError) as exc:
        require_scope(p, "activity.resolve")
    assert exc.value.status == 403
    assert "insufficient_scope" in exc.value.www_authenticate
    assert 'scope="activity.resolve"' in exc.value.www_authenticate


# ── infrastructure failure ───────────────────────────────────────────────────────────────

def test_jwks_unreachable_is_503_not_401(make_token):
    """401 tells the caller to get a new token, which is useless when the fault is ours."""
    v = make_verifier(AuthConfig.oidc(jwks_url="http://127.0.0.1:1/jwks.json",
                                      issuer=ISSUER, audience=AUDIENCE))
    with pytest.raises(AuthError) as exc:
        v.verify(f"Bearer {make_token()}")
    assert exc.value.status == 503


# ── static-token mode ────────────────────────────────────────────────────────────────────

def test_static_token_accepts_and_attributes_to_configured_identity():
    v = make_verifier(AuthConfig.static_token(token="s3cret", identity="dev@local"))
    p = v.verify("Bearer s3cret")
    assert p.label == "dev@local" and p.subject == "dev@local"
    assert p.auth_mode == "token"        # the field that keeps a dev record distinguishable


def test_static_token_rejects_wrong_token():
    v = make_verifier(AuthConfig.static_token(token="s3cret"))
    with pytest.raises(AuthError) as exc:
        v.verify("Bearer wrong")
    assert exc.value.status == 401
