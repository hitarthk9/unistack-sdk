"""
SDK-wide configuration constants.

Model names and pricing are defined here — the single source of truth.
Override the guardrail model at runtime with the UNISTACK_GUARDRAIL_MODEL env var.
"""

import os

# Model used by the SDK's own guardrail evaluator
GUARDRAIL_MODEL: str = os.environ.get("UNISTACK_GUARDRAIL_MODEL", "claude-haiku-4-5-20251001")

# (input $/token, output $/token) — used by _instrument_anthropic() to compute span cost
LLM_PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (0.0000008, 0.000004),   # $0.80 / $4.00 per MTok
    "claude-haiku-4-5":          (0.0000008, 0.000004),
    "claude-sonnet-4-6":         (0.000003,  0.000015),
    "claude-opus-4-8":           (0.000015,  0.000075),
}
DEFAULT_INPUT_PRICE:  float = 0.000001   # fallback for unknown models
DEFAULT_OUTPUT_PRICE: float = 0.000005
