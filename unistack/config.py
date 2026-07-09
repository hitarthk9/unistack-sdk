"""
SDK-wide configuration constants.

Override the guardrail model at runtime with the UNISTACK_GUARDRAIL_MODEL env var.
"""

import os

# Model used by the SDK's own guardrail evaluator (LLM-as-judge).
GUARDRAIL_MODEL: str = os.environ.get("UNISTACK_GUARDRAIL_MODEL", "claude-haiku-4-5-20251001")
