"""
SDK-wide configuration constants. All can be overridden via environment variables.
"""

import os

GUARDRAIL_MODEL: str = os.environ.get("UNISTACK_GUARDRAIL_MODEL", "claude-haiku-4-5-20251001")
MONGO_URI: str       = os.environ.get("MONGO_URI",          "mongodb://localhost:27017")
API_URL: str         = os.environ.get("UNISTACK_API_URL",   "http://localhost:8000")
