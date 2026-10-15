"""Input, tool and output guardrails.

Heuristic and deterministic on purpose: fast, explainable, testable offline. In a real system
these would sit alongside a classifier model (for example a small prompt injection model or
Llama Guard), not be replaced by it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Prompt injection / jailbreak signals with weights. A score >= threshold blocks the request.
INJECTION_PATTERNS: list[tuple[str, float]] = [
    (r"\bignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instructions|rules|prompts?)", 1.0),
    (r"\bdisregard\s+(the\s+|all\s+|your\s+)?(system|previous|above|prior)\b", 1.0),
    (r"\b(reveal|print|show|repeat|leak)\s+(me\s+)?(your|the)\s+(system\s+prompt|instructions|hidden\s+prompt)", 1.0),
    (r"\byou\s+are\s+now\s+(dan|in\s+developer\s+mode|unrestricted|jailbroken)", 1.0),
    (r"\b(developer|god|admin|sudo)\s+mode\b", 0.6),
    (r"\bpretend\s+(you\s+are|to\s+be)\b.*\b(no|without)\s+(rules|restrictions|limits)", 0.9),
    (r"<\s*/?\s*(system|assistant)\s*>", 0.8),
    (r"\bnew\s+instructions?\s*:", 0.7),
    (r"\b(close|delete|resolve)\s+(all|every)\s+(tickets?|issues?|incidents?)", 0.9),
    (r"\bbase64\b.*\bdecode\b|\bdecode\b.*\bbase64\b", 0.9),
    (r"[A-Za-z0-9+/]{20,}={0,2}", 0.4),  # long base64-like payload
]
INJECTION_THRESHOLD = 0.8


@dataclass
class PiiMatch:
    kind: str
    value: str


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for d in reversed(digits):
        n = int(d)
        if alt:
            n *= 2
            if n > 9:
                n -= 9
        total += n
        alt = not alt
    return total % 10 == 0


PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "IBAN": re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,4})?\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "IE_PPSN": re.compile(r"\b\d{7}[A-W][A-IW]?\b"),
    "PHONE": re.compile(r"(?<!\w)\+?\d{1,3}[\s-]?\(?\d{2,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}(?!\w)"),
}


def find_pii(text: str) -> list[PiiMatch]:
    found: list[PiiMatch] = []
    taken: list[tuple[int, int]] = []
    for kind, pat in PII_PATTERNS.items():  # order matters: specific patterns first
        for m in pat.finditer(text):
            span = m.span()
            if any(s < span[1] and span[0] < e for s, e in taken):
                continue
            value = m.group(0)
            if kind == "CREDIT_CARD":
                digits = re.sub(r"\D", "", value)
                if not (13 <= len(digits) <= 19 and _luhn_ok(digits)):
                    continue
            found.append(PiiMatch(kind, value))
            taken.append(span)
    return found


def redact(text: str) -> tuple[str, list[str]]:
    kinds: list[str] = []
    for m in find_pii(text):
        text = text.replace(m.value, f"[{m.kind}]")
        kinds.append(m.kind)
    return text, kinds


@dataclass
class GuardResult:
    allowed: bool
    text: str
    reasons: list[str] = field(default_factory=list)
    injection_score: float = 0.0
    pii_kinds: list[str] = field(default_factory=list)


def injection_score(text: str) -> tuple[float, list[str]]:
    lowered = text.lower()
    score, hits = 0.0, []
    for pat, w in INJECTION_PATTERNS:
        if re.search(pat, lowered):
            score += w
            hits.append(pat)
    return min(score, 2.0), hits


def check_input(text: str, max_len: int = 4000) -> GuardResult:
    if len(text) > max_len:
        return GuardResult(False, "", ["input_too_long"])
    score, hits = injection_score(text)
    if score >= INJECTION_THRESHOLD:
        return GuardResult(False, "", ["prompt_injection"], injection_score=score)
    # PII is redacted, not blocked: the user can still get help without the data reaching tools or logs.
    clean, kinds = redact(text)
    return GuardResult(True, clean, ["pii_redacted"] if kinds else [], injection_score=score, pii_kinds=kinds)


def sanitize_tool_output(text: str) -> tuple[str, bool]:
    """Indirect injection defence: tool data is quoted as data and instructions inside it are neutralised."""
    score, _ = injection_score(text)
    clean, _ = redact(text)
    if score >= INJECTION_THRESHOLD:
        return "[content withheld: tool output contained instructions]", True
    return clean, False


@dataclass
class ToolPolicy:
    allowed_tools: set[str]
    max_tool_calls: int = 4
    write_tools_need_explicit_intent: bool = True


def check_output(text: str, system_prompt: str) -> GuardResult:
    reasons = []
    # Block verbatim system prompt leakage (any 8 word window).
    words = system_prompt.split()
    for i in range(max(0, len(words) - 7)):
        if " ".join(words[i : i + 8]) in text:
            return GuardResult(False, "I can't share that.", ["system_prompt_leak"])
    clean, kinds = redact(text)
    if kinds:
        reasons.append("pii_redacted")
    return GuardResult(True, clean, reasons, pii_kinds=kinds)
