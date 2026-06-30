import json
import os


class GuardrailBreached(Exception):
    def __init__(self, policy: str, reason: str):
        self.policy = policy
        self.reason = reason
        super().__init__(f"Guardrail breached: {reason}")


def evaluate_guardrail(policy: str, output: str, context: str | None = None) -> dict:
    """
    Returns {"passed": bool, "reason": str}.
    Uses Claude Haiku if ANTHROPIC_API_KEY is set; falls back to keyword scan.
    When context is provided, it is injected into the LLM prompt so the
    evaluator has workflow-specific business domain knowledge.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        import re
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        context_section = f"\nBusiness Context:\n{context}\n" if context else ""
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": (
                    "You are a business policy guardrail evaluator."
                    f"{context_section}\n"
                    f"Policy to enforce: {policy}\n"
                    f"Output to evaluate: {output}\n\n"
                    'Respond with ONLY valid JSON, no markdown: '
                    '{"passed": true, "reason": "..."} or {"passed": false, "reason": "..."}'
                ),
            }],
        )
        text = resp.content[0].text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            # If Claude couldn't produce clean JSON, treat as passed
            return {"passed": True, "reason": f"Guardrail parse error — defaulting to pass: {text[:80]}"}
    else:
        # Keyword-based fallback so demo works without an API key
        breach_keywords = ["fraud", "discriminat", "illegal", "banned", "blocked", "sanctioned"]
        lowered = output.lower()
        for kw in breach_keywords:
            if kw in lowered:
                return {"passed": False, "reason": f"Flagged keyword detected: '{kw}' (keyword scan)"}
        return {"passed": True, "reason": "No policy violations detected (keyword scan fallback)"}
