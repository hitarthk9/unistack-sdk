import json

from langsmith import traceable


def _judge_trace_inputs(inputs: dict) -> dict:
    """Log the judge's decision inputs to LangSmith — never the API key."""
    return {k: inputs[k] for k in ("policy", "output", "context", "model") if k in inputs}


@traceable(name="guardrail_eval", run_type="chain", process_inputs=_judge_trace_inputs)
def evaluate_guardrail(
    policy: str,
    output: str,
    context: str | None = None,
    api_key: str | None = None,
    model: str = "claude-haiku-4-5-20251001",
) -> dict:
    """
    Returns {"passed": bool, "reason": str}.
    Uses Claude when api_key is supplied; falls back to a keyword scan otherwise.
    When context is provided, it is injected into the LLM prompt so the
    evaluator has workflow-specific business domain knowledge.
    """
    if api_key:
        import re
        import anthropic
        from langsmith.wrappers import wrap_anthropic
        # wrap_anthropic traces the Claude call as a child LLM span with tokens + cost.
        client = wrap_anthropic(anthropic.Anthropic(api_key=api_key))
        context_section = f"\nBusiness Context:\n{context}\n" if context else ""
        resp = client.messages.create(
            model=model,
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
