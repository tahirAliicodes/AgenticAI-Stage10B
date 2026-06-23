# pii_redactor.py
#
# STANDALONE — no imports from the rest of the project.
#
# ELI5: this file finds and blacks out personal information (PII) before
# it goes anywhere — your SSN, email, phone number, credit card, etc.
#
# Two layers, same idea as the injection detector:
#   1. REGEX layer — fast, exact. PII like SSNs/emails/credit cards has a
#      fixed SHAPE (xxx-xx-xxxx, xxx@xxx.com), so regex catches these almost
#      perfectly and redacts them immediately.
#   2. LLM JUDGE layer — regex can't catch PII that doesn't follow a fixed
#      shape: a full name, a home address written in a sentence, an SSN
#      spelled out in words ("five five five..."). The LLM reads the
#      ALREADY-REDACTED text and flags anything regex missed.

import re
import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional


# ─────────────────────────────────────────
# RESULT TYPES
# ─────────────────────────────────────────

class SecurityLevel(str, Enum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


SEVERITY_SCORE = {
    SecurityLevel.SAFE: 0,
    SecurityLevel.LOW: 1,
    SecurityLevel.MEDIUM: 2,
    SecurityLevel.HIGH: 3,
    SecurityLevel.CRITICAL: 4,
}
SCORE_TO_LEVEL = {v: k for k, v in SEVERITY_SCORE.items()}


@dataclass
class PIIMatch:
    pii_type: str
    severity: SecurityLevel
    matched_text: str          # original value BEFORE redaction (for the report only)
    caught_by: str             # "regex" or "llm_judge"


@dataclass
class PIIResult:
    is_safe: bool                       # False if HIGH/CRITICAL PII found
    severity: SecurityLevel
    redacted_text: str
    pii_found: bool
    matches: List[PIIMatch] = field(default_factory=list)
    llm_reasoning: Optional[str] = None

    def summary(self) -> str:
        if not self.pii_found:
            msg = "✅ CLEAN — no PII detected"
            if self.llm_reasoning:
                msg += f"\n   🤖 LLM judge agreed: {self.llm_reasoning}"
            return msg
        verdict = "🚫 BLOCKED" if not self.is_safe else "⚠️  FLAGGED (allowed, redacted)"
        lines = [f"{verdict} — severity: {self.severity.value}"]
        for m in self.matches:
            lines.append(f"   - [{m.pii_type}] caught by {m.caught_by}: \"{m.matched_text}\"")
        lines.append(f"   redacted text: \"{self.redacted_text}\"")
        if self.llm_reasoning:
            lines.append(f"   🤖 LLM judge: {self.llm_reasoning}")
        return "\n".join(lines)


# ─────────────────────────────────────────
# LAYER 1 — REGEX PATTERNS (structured PII, fixed shapes)
# ─────────────────────────────────────────
# Each entry: (pii_type, regex, severity, redaction_placeholder)

PII_PATTERNS: List[Tuple[str, str, SecurityLevel, str]] = [

    ("ssn",
     r"\b\d{3}-\d{2}-\d{4}\b",
     SecurityLevel.CRITICAL,
     "[REDACTED-SSN]"),

    ("credit_card",
     r"\b(?:\d[ -]*?){13,16}\b",
     SecurityLevel.CRITICAL,
     "[REDACTED-CARD]"),

    ("email",
     r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
     SecurityLevel.HIGH,
     "[REDACTED-EMAIL]"),

    ("phone",
     r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
     SecurityLevel.HIGH,
     "[REDACTED-PHONE]"),

    ("ip_address",
     r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
     SecurityLevel.MEDIUM,
     "[REDACTED-IP]"),

    ("date_of_birth",
     r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b",
     SecurityLevel.HIGH,
     "[REDACTED-DOB]"),

    ("passport",
     r"\b(?:passport\s*(?:#|no\.?|number)?\s*[:=]?\s*)([A-Z0-9]{6,9})\b",
     SecurityLevel.CRITICAL,
     "[REDACTED-PASSPORT]"),
]


# ─────────────────────────────────────────
# REGEX REDACTOR
# ─────────────────────────────────────────

class RegexPIIRedactor:
    def __init__(self):
        self._compiled = [
            (name, re.compile(pattern, re.IGNORECASE), severity, placeholder)
            for name, pattern, severity, placeholder in PII_PATTERNS
        ]

    def redact(self, text: str) -> Tuple[str, List[PIIMatch]]:
        redacted = text
        matches: List[PIIMatch] = []

        for name, regex, severity, placeholder in self._compiled:
            for found in regex.finditer(text):
                matches.append(PIIMatch(
                    pii_type=name,
                    severity=severity,
                    matched_text=found.group(0),
                    caught_by="regex",
                ))
            redacted = regex.sub(placeholder, redacted)

        return redacted, matches


# ─────────────────────────────────────────
# LAYER 2 — LLM JUDGE (catches unstructured PII regex can't shape-match)
# ─────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"

JUDGE_PROMPT = """You are a PII (personally identifiable information) detector. \
The text below has ALREADY had structured PII (emails, SSNs, phone numbers, \
credit cards) redacted. Your job is to catch anything REMAINING that exposes \
a PRIVATE individual's personal identifying info: full names, home addresses, \
employer names tied to a private person, spelled-out numbers (e.g. "five \
five five..."), or other personal identifiers written in plain language.

IMPORTANT — do NOT flag these, they are NOT PII:
- Questions about public figures, celebrities, politicians, historical \
figures, or fictional characters (e.g. "what is Johnny Depp's wife's name",
"who is the president", "tell me about Einstein"). Public figures' publicly \
known info is general knowledge, not private PII.
- General knowledge questions that merely contain a name as the subject \
being asked about, rather than disclosing someone's private contact/identity \
info.

ONLY flag text that discloses or exposes a PRIVATE person's identifying \
details — e.g. "my neighbor John Smith lives at 12 Oak St" or "my coworker's \
number is..." — not text that merely asks a factual question about a known \
public figure.

Respond with EXACTLY this format, nothing else:
VERDICT: YES or NO
CATEGORIES: comma-separated list of PII types found, or NONE
REASON: one short sentence

Text to check:
\"\"\"{text}\"\"\"
"""


class LLMPIIJudge:
    def __init__(self, model: str = OLLAMA_MODEL, url: str = OLLAMA_URL, timeout: int = 15):
        self.model = model
        self.url = url
        self.timeout = timeout

    def judge(self, redacted_text: str) -> Tuple[Optional[bool], str, str]:
        """
        Returns (has_pii, categories, reason).
        has_pii is None if Ollama couldn't be reached (fail open).
        """
        payload = json.dumps({
            "model": self.model,
            "prompt": JUDGE_PROMPT.format(text=redacted_text),
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            self.url, data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ConnectionRefusedError, OSError) as e:
            return None, "", f"LLM judge unavailable ({e})"

        raw = body.get("response", "").strip()

        # Use the LAST match, not the first — some models echo the prompt's
        # format instructions ("VERDICT: YES or NO") before giving their
        # real answer. A naive first-match search would wrongly match the
        # echoed instruction line itself.
        verdict_matches = re.findall(r"VERDICT:\s*(YES|NO)", raw, re.IGNORECASE)
        has_pii = bool(verdict_matches) and verdict_matches[-1].upper() == "YES"

        cat_matches = re.findall(r"CATEGORIES:\s*(.+)", raw, re.IGNORECASE)
        categories = cat_matches[-1].strip() if cat_matches else "unknown"

        reason_matches = re.findall(r"REASON:\s*(.+)", raw, re.IGNORECASE)
        reason = reason_matches[-1].strip() if reason_matches else raw[:150]

        return has_pii, categories, reason


# ─────────────────────────────────────────
# COMBINED DETECTOR — regex (fast, exact) + LLM judge (catches the rest)
# ─────────────────────────────────────────

class SmartPIIRedactor:
    """
    1. Regex redacts everything with a fixed shape immediately (SSN, email,
       phone, credit card, IP, DOB, passport).
    2. LLM judge reads the ALREADY-REDACTED text and flags anything left
       that still looks like PII (names, addresses, spelled-out numbers).
       We don't ask the LLM to redact — LLMs aren't reliable at precise
       text editing — we just have it flag, same as a human reviewer would.
    """

    def __init__(self, use_llm: bool = True):
        self.regex_redactor = RegexPIIRedactor()
        self.llm_judge = LLMPIIJudge() if use_llm else None

    def redact(self, text: str) -> PIIResult:
        # ── Layer 1: regex ──
        redacted_text, regex_matches = self.regex_redactor.redact(text)

        max_score = max((SEVERITY_SCORE[m.severity] for m in regex_matches), default=0)

        if self.llm_judge is None:
            severity = SCORE_TO_LEVEL[max_score]
            return PIIResult(
                is_safe=max_score < SEVERITY_SCORE[SecurityLevel.HIGH],
                severity=severity,
                redacted_text=redacted_text,
                pii_found=bool(regex_matches),
                matches=regex_matches,
            )

        # ── Layer 2: LLM judge on the already-redacted text ──
        has_pii, categories, reason = self.llm_judge.judge(redacted_text)

        if has_pii is None:
            # Ollama unreachable — fail open to regex-only verdict
            severity = SCORE_TO_LEVEL[max_score]
            return PIIResult(
                is_safe=max_score < SEVERITY_SCORE[SecurityLevel.HIGH],
                severity=severity,
                redacted_text=redacted_text,
                pii_found=bool(regex_matches),
                matches=regex_matches,
                llm_reasoning=f"(skipped) {reason}",
            )

        if has_pii:
            llm_match = PIIMatch(
                pii_type=categories,
                severity=SecurityLevel.HIGH,
                matched_text="(see categories — LLM doesn't extract exact spans)",
                caught_by="llm_judge",
            )
            max_score = max(max_score, SEVERITY_SCORE[SecurityLevel.HIGH])
            return PIIResult(
                is_safe=False,
                severity=SCORE_TO_LEVEL[max_score],
                redacted_text=redacted_text,
                pii_found=True,
                matches=regex_matches + [llm_match],
                llm_reasoning=reason,
            )

        # LLM agrees nothing else is left
        severity = SCORE_TO_LEVEL[max_score]
        return PIIResult(
            is_safe=max_score < SEVERITY_SCORE[SecurityLevel.HIGH],
            severity=severity,
            redacted_text=redacted_text,
            pii_found=bool(regex_matches),
            matches=regex_matches,
            llm_reasoning=reason,
        )


# ─────────────────────────────────────────
# SINGLETONS
# ─────────────────────────────────────────

pii_redactor = RegexPIIRedactor()              # regex-only, fast
smart_pii_redactor = SmartPIIRedactor()        # regex + LLM judge


# ─────────────────────────────────────────
# DEMO — run this file directly:
#   python pii_redactor.py
# ─────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("PII REDACTOR — REGEX + LLM JUDGE (Ollama)")
    print("=" * 60)
    print("Type a message and press Enter to scan it.")
    print("Type 'quit' or 'exit' to stop.\n")

    detector = smart_pii_redactor

    while True:
        text = input("YOU> ").strip()

        if text.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if not text:
            continue

        start = time.time()
        result = detector.redact(text)
        elapsed = round((time.time() - start) * 1000)

        print(result.summary())
        print(f"   ⏱  {elapsed}ms")
        print()