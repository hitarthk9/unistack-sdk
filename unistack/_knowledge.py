"""
Knowledge bases for guards — a governed list of rules instead of a policy string.

A guard's policy is normally one sentence written inline in application code. That does not
survive contact with a real client compliance manual: the policy ends up scattered across
`guards={...}`, it has no version, and a breach verdict cannot say *which* rule was broken.

A knowledge base is plain structured data — validated here, rendered into the judge's prompt,
and cited back by rule id in the verdict:

    {"knowledge_base": "brand-policy",
     "version": 1,
     "rules": [{"id": "BP-001",
                "rule": "Do not imply a health outcome without naming a source.",
                "examples": {"violating": "...", "compliant": "..."}}]}

**Every rule is sent on every evaluation. There is deliberately no retrieval.** A guard is a
safety control, and retrieval introduces a failure mode it does not currently have: a miss means
the judge never sees the relevant rule and silently passes output it should have caught. Sending
the whole list keeps the guarantee that the judge always sees the entire policy — and because
the rendered block is then byte-identical on every call, it is prompt-cacheable, which retrieval
(different chunks per call) could never be. See `MAX_RULE_TOKENS` for where this stops scaling.

This module is pure: no IO, no network, no Mongo. The SDK reads no files (hard constraint #9) —
`unistack-runtime` loads the YAML and hands the parsed dict to `UniStack.init()`.
"""

import logging

logger = logging.getLogger("unistack")

#: Rough chars-per-token for English prose. Only used for the size warning, so an estimate is
#: fine — we never need to be exact, only loud before the corpus gets unreasonable.
_CHARS_PER_TOKEN = 4

#: The point at which to warn. NOT a context limit — Haiku 4.5 holds 200k tokens, so context is
#: the *least* binding constraint here. The real limit is attention: a judge weighing hundreds of
#: rules attends less carefully to any one of them than it would to ten. That degrades quietly,
#: which is exactly why this warns.
MAX_RULE_TOKENS = 50_000


class KnowledgeBaseError(ValueError):
    """A knowledge base is malformed. Raised at compile time, never mid-run."""


def validate(kb) -> dict:
    """
    Check a knowledge base's shape and return it.

    Fails loudly at compile time rather than producing a judge prompt with silently missing
    rules — a guard that quietly drops half its policy is worse than one that refuses to start.
    """
    if not isinstance(kb, dict):
        raise KnowledgeBaseError(f"knowledge base must be a mapping, got {type(kb).__name__}")

    name = kb.get("knowledge_base")
    if not isinstance(name, str) or not name.strip():
        raise KnowledgeBaseError("knowledge base needs a non-empty 'knowledge_base' name")

    rules = kb.get("rules")
    if not isinstance(rules, list) or not rules:
        raise KnowledgeBaseError(f"knowledge base {name!r} has no 'rules'")

    seen = set()
    for i, rule in enumerate(rules):
        where = f"{name!r} rule #{i + 1}"
        if not isinstance(rule, dict):
            raise KnowledgeBaseError(f"{where} must be a mapping, got {type(rule).__name__}")
        rid, text = rule.get("id"), rule.get("rule")
        if not isinstance(rid, str) or not rid.strip():
            raise KnowledgeBaseError(f"{where} needs a non-empty 'id' — the verdict cites it")
        if not isinstance(text, str) or not text.strip():
            raise KnowledgeBaseError(f"{where} ({rid}) needs a non-empty 'rule'")
        if rid in seen:
            # Duplicate ids would make a citation ambiguous, which defeats the point of having
            # ids at all.
            raise KnowledgeBaseError(f"{name!r} has duplicate rule id {rid!r}")
        seen.add(rid)
    return kb


def render(kb: dict) -> str:
    """
    Render a validated knowledge base into the block the judge reads.

    Stable output for stable input: the same knowledge base always renders byte-identically, so
    the prompt prefix can be cached. Do not add timestamps, ordering by dict iteration, or
    anything else non-deterministic here — it would silently cost a ~10x price increase by
    invalidating the cache on every call.
    """
    lines = [f"Policy: {kb['knowledge_base']} (version {kb.get('version', 'unversioned')})",
             "Judge the output against EVERY rule below. Cite the id of each rule it breaches.",
             ""]
    for rule in kb["rules"]:
        lines.append(f"[{rule['id']}] {rule['rule'].strip()}")
        examples = rule.get("examples") or {}
        # Examples are what make a rule judgeable rather than merely stated — "no unverified
        # claims" is far more consistently applied when the model has seen one of each.
        #
        # EVERY label is rendered, in the order the author wrote them — not a fixed
        # violating/compliant pair. A rule often needs a second example to mark a boundary the
        # judge over-reaches on (see BP-001 in the demo), and a renderer that quietly dropped
        # the extra label would leave the author editing a file that has no effect. Order is
        # the YAML's own, so the block stays byte-stable and therefore cacheable.
        if isinstance(examples, dict):
            for label, text in examples.items():
                if text and not isinstance(text, (dict, list)):
                    lines.append(f"    {label}: {str(text).strip()}")
        lines.append("")
    return "\n".join(lines).rstrip()


def resolve(guard, knowledge_bases: dict | None) -> tuple[str, str | None]:
    """
    Turn a guard value into (policy_text, knowledge_base_name).

    Accepts both shapes, so existing graphs are untouched:
        "no unverified claims"                              -> prose, no KB
        {"policy": "...", "knowledge_base": "brand-policy"} -> either or both
    """
    if isinstance(guard, str):
        return guard, None
    if not isinstance(guard, dict):
        raise KnowledgeBaseError(
            f"a guard must be a policy string or a mapping, got {type(guard).__name__}")

    policy = guard.get("policy") or ""
    name = guard.get("knowledge_base")
    if name is None:
        if not policy:
            raise KnowledgeBaseError("a guard mapping needs 'policy', 'knowledge_base', or both")
        return policy, None
    if name not in (knowledge_bases or {}):
        # Naming a KB that was never supplied means the judge would run with an empty or
        # partial policy. Refuse at compile time instead.
        known = ", ".join(sorted(knowledge_bases or {})) or "none"
        raise KnowledgeBaseError(
            f"guard names unknown knowledge base {name!r} (loaded: {known})")
    return policy, name


def policy_block(policy: str, kb: dict | None) -> str:
    """The full trusted policy text for one guard: node-specific prose plus the shared rules."""
    parts = []
    if policy:
        parts.append(policy.strip())
    if kb:
        parts.append(render(kb))
    return "\n\n".join(parts)


def check_size(name: str, rendered: str) -> int:
    """Warn — loudly, once, at load time — when a knowledge base outgrows this approach."""
    tokens = len(rendered) // _CHARS_PER_TOKEN
    if tokens > MAX_RULE_TOKENS:
        logger.warning(
            "knowledge base %r renders to ~%d tokens, over the %d advised for a single judge "
            "prompt. Context is not the problem — attention is: a judge weighing this many "
            "rules applies each one less carefully. Split it per node, or move to retrieval.",
            name, tokens, MAX_RULE_TOKENS)
    return tokens


__all__ = ["KnowledgeBaseError", "MAX_RULE_TOKENS",
           "validate", "render", "resolve", "policy_block", "check_size"]
