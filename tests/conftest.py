"""
Shared fixtures for the UniStack SDK suite.

The auth tests need a real OIDC issuer to verify against. Rather than monkeypatch
`PyJWKClient`'s internals, we generate an RSA keypair at session start and serve its JWKS
from a loopback HTTP server on an ephemeral port. That exercises the production code path
end to end — URL config → HTTP fetch → `kid` lookup → cache — with no external network and
no test-only seam in shipped code that could drift from the real path.

The private key is GENERATED per session, never committed: a checked-in PEM trips secret
scanners and reads as a real leak in every future audit.
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

KID = "test-key-1"
ISSUER = "https://issuer.test/realms/unistack"
AUDIENCE = "unistack-runtime"


@pytest.fixture(scope="session")
def rsa_key():
    """A generated 2048-bit RSA keypair: the PEM to sign with, the object to publish."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return {
        "kid": KID,
        "private_pem": key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode(),
        "public_pem": key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode(),
        "public_key": key.public_key(),
    }


@pytest.fixture(scope="session")
def jwks(rsa_key):
    jwk = jwt.algorithms.RSAAlgorithm.to_jwk(rsa_key["public_key"], as_dict=True)
    jwk.update(kid=rsa_key["kid"], use="sig", alg="RS256")
    return {"keys": [jwk]}


@pytest.fixture(scope="session")
def jwks_url(jwks):
    """Serve the JWKS over loopback on an ephemeral port, for the session."""
    payload = json.dumps(jwks).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):          # keep pytest output clean
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/jwks.json"
    server.shutdown()
    server.server_close()


@pytest.fixture
def make_token(rsa_key):
    """
    Mint a signed JWT. `scopes` land in the claim named by `claim` (`roles` gets a list,
    everything else a space-delimited string). Any claim can be overridden by keyword, and
    passing None for one DROPS it — which is how the "no subject" / "no aud" cases are built.
    """
    def _make(scopes=("activity.start", "activity.resolve"), *, claim="scp",
              key=None, algorithm="RS256", kid=KID, **overrides):
        now = int(time.time())
        payload = {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "sub": "user-sub-1",
            "email": "approver@example.com",
            "iat": now,
            "exp": now + 300,
        }
        if scopes is not None:
            payload[claim] = list(scopes) if claim == "roles" else " ".join(scopes)
        payload.update(overrides)
        payload = {k: v for k, v in payload.items() if v is not None}
        return jwt.encode(payload, key or rsa_key["private_pem"],
                          algorithm=algorithm, headers={"kid": kid})
    return _make


@pytest.fixture(autouse=True)
def _clean_unistack_env(monkeypatch):
    """A developer's exported UNISTACK_*/OTEL_* vars must not change test results."""
    for name in [k for k in os.environ if k.startswith(("UNISTACK_", "OTEL_"))]:
        monkeypatch.delenv(name, raising=False)
