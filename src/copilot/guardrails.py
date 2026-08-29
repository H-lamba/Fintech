"""
Checking what the model actually said, not what it was told to say.

A system prompt that instructs a model not to invent numbers is a request, not
a control. These checks are the control: they run on every response, before it
reaches a reviewer, and they are what make "grounded" a property of the output
rather than a claim about the prompt.

Three checks, in order of how badly a failure would matter:

1. **Prediction language.** The challenge disqualifies a solution that uses an
   LLM to classify. A note that says "this loan will default" has crossed from
   summarising the model's output into producing its own, and no amount of
   surrounding disclaimer fixes that.
2. **Decision language.** The output is a recommendation for a human. A note
   that says "deny the modification" has made the decision the governance
   framework reserves for a person.
3. **Numeric grounding.** Every number in the note must appear in the context
   the model was given. This is the hallucination check, and it is the one that
   catches the failure mode a reader is least able to spot: a confident,
   well-formatted note containing a balance nobody supplied.

A check that fires does not silently rewrite the output. It is recorded on the
response, surfaced in the report, and the note is held back from the reviewer
queue -- because a governance layer that quietly patches its model's mistakes
has destroyed the evidence that it makes them.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Phrases that assert an outcome the LLM is not permitted to produce. Matched
# case-insensitively on word boundaries.
PREDICTION_PATTERNS = [
    r"\bwill (?:default|prepay|become delinquent|cure)\b",
    r"\bis (?:likely|unlikely) to (?:default|prepay)\b",
    r"\b(?:i|we) (?:predict|forecast|estimate|expect) (?:a|an|the|this)\b",
    r"\bmy (?:estimate|prediction|forecast)\b",
    r"\bprobability of default is approximately\b",
    r"\bshould be classified as\b",
]

# Phrases that take the decision rather than inform it.
DECISION_PATTERNS = [
    # Gerunds matter: "I recommend denying" is the same act as "I recommend deny".
    r"\b(?:you should|we should|i recommend|recommend) (?:deny|denying|approve|approving|"
    r"reject|rejecting|foreclose|foreclosing|writ(?:e|ing) off)\b",
    r"\b(?:deny|denying|approve|approving|reject|rejecting) (?:the|this) "
    r"(?:loan|modification|application|claim)\b",
    r"\bno further review (?:is )?(?:needed|required)\b",
    r"\bthis (?:loan|record) (?:is|should be) (?:cleared|closed) without\b",
    r"\btake no action\b",
]

# Hedging that reads as authority without carrying any: the "vague" failure.
VAGUE_PATTERNS = [
    r"\bit (?:seems|appears) (?:likely|probable|reasonable)\b",
    r"\bgenerally speaking\b",
    r"\bin most cases\b",
    r"\btypically,? (?:loans|borrowers)\b",
]

# The lookbehind stops a range being read as a negative: in "3-5 sentences"
# the hyphen is a dash, not a minus, and reporting "-5" as a hallucinated
# figure is exactly the kind of false positive that gets a grounding check
# switched off by the people it is meant to protect.
_NUMBER = re.compile(r"(?<![\d.\-])-?\d[\d,]*\.?\d*")

# A date is one fact, not three numbers. Extracted and matched whole before the
# number scan runs, because a model writing "2023-06-01" for a date it was
# given must not be accused of inventing "06" and "01".
_DATE = re.compile(r"\d{4}-\d{2}(?:-\d{2})?")

# Models render hyphens with any of these. Left unnormalised, a Unicode
# non-breaking hyphen in "2023‑06‑01" defeats the date pattern and the
# fragments surface as hallucinated numbers -- which is what a false-positive
# grounding check looks like from the inside, and why users switch them off.
#
# NFKC alone does NOT fix this, which is worth stating because it is the
# obvious first reach and it fails silently. Measured on this Python:
#
#     U+2011 NON-BREAKING HYPHEN  --NFKC-->  U+2010 HYPHEN   (still not ASCII)
#     U+2013 EN DASH              --NFKC-->  U+2013          (unchanged)
#     U+2014 EM DASH              --NFKC-->  U+2014          (unchanged)
#     U+2212 MINUS SIGN           --NFKC-->  U+2212          (unchanged)
#
# En and em dashes are distinct semantic characters, not compatibility variants
# of hyphen-minus, so Unicode has no mandate to fold them. The explicit table
# below is what actually does the work; NFKC is applied first because it *does*
# fold the whitespace forms (NBSP, narrow NBSP) and other compatibility
# characters the table does not enumerate.
_DASHES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
                         "\u2014": "-", "\u2015": "-", "\u2212": "-", "\ufe58": "-",
                         "\ufe63": "-", "\uff0d": "-", "\u00a0": " ", "\u202f": " ",
                         "\u2009": " ", "\u200a": " ", "\u2007": " "})

# Numbers a note may use without them being in the context: ordinals and small
# counts it derives from the context itself ("three rules fired"), zero-padded
# or not, plus plausible calendar years.
_ALWAYS_ALLOWED = (
    {str(n) for n in range(0, 13)}
    | {f"{n:02d}" for n in range(0, 32)}
    | {str(y) for y in range(2015, 2031)}
)


def normalise(text: str) -> str:
    """
    Fold typographic characters so the patterns match what a model wrote.

    Two passes, and both are needed. NFKC handles compatibility forms --
    non-breaking and narrow spaces, full-width digits, ligatures -- which is a
    wider net than any hand-written table. It does not touch en dashes, em
    dashes or the minus sign, so the explicit table runs after it.
    """
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text).translate(_DASHES)


@dataclass
class GuardrailVerdict:
    """The result of checking one response. ``passed`` gates the reviewer queue."""

    passed: bool = True
    prediction_language: list = field(default_factory=list)
    decision_language: list = field(default_factory=list)
    vague_language: list = field(default_factory=list)
    ungrounded_numbers: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "prediction_language": self.prediction_language,
            "decision_language": self.decision_language,
            "vague_language": self.vague_language,
            "ungrounded_numbers": self.ungrounded_numbers,
            "notes": self.notes,
        }

    def summary(self) -> str:
        if self.passed:
            return "passed"
        reasons = []
        if self.prediction_language:
            reasons.append(f"prediction language ({len(self.prediction_language)})")
        if self.decision_language:
            reasons.append(f"decision language ({len(self.decision_language)})")
        if self.ungrounded_numbers:
            reasons.append(f"ungrounded numbers: {', '.join(self.ungrounded_numbers[:4])}")
        if self.vague_language:
            reasons.append(f"vague language ({len(self.vague_language)})")
        return "; ".join(reasons) or "failed"


def _matches(text: str, patterns: list) -> list:
    found = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            found.append(match.group(0))
    return found


def ungrounded_numbers(text: str, grounded: set) -> list:
    """
    Numbers in the output that do not appear in the supplied context.

    Matched against every reasonable rendering of each context value -- 250000,
    250,000, 250000.00 -- so a note that formats a figure differently is not
    accused of inventing it. What survives is a figure with no source.
    """
    text = normalise(text)

    # Match dates whole, then remove them so their parts are not rescanned.
    for date in _DATE.findall(text):
        if date not in grounded and date[:7] not in grounded:
            # A date with no source is still a hallucination -- report it as one
            # rather than as its fragments.
            return sorted({date} | set(_scan_numbers(text.replace(date, " "), grounded)))
        text = text.replace(date, " ")

    return _scan_numbers(text, grounded)


def _scan_numbers(text: str, grounded: set) -> list:
    unmatched = []
    for token in _NUMBER.findall(text):
        cleaned = token.replace(",", "").rstrip(".")
        if not cleaned or cleaned in _ALWAYS_ALLOWED:
            continue
        candidates = {cleaned, cleaned.lstrip("-")}
        try:
            as_float = float(cleaned)
        except ValueError:
            continue
        candidates.update({f"{as_float:g}", f"{as_float:.0f}", f"{as_float:.1f}", f"{as_float:.2f}"})
        candidates = {c.rstrip(".") for c in candidates}
        if not candidates & grounded:
            unmatched.append(token)
    return sorted(set(unmatched))


def check(text: str, grounded: set | None = None) -> GuardrailVerdict:
    """Run every guardrail against one response."""
    verdict = GuardrailVerdict()
    if not text or not text.strip():
        verdict.passed = False
        verdict.notes.append("empty response")
        return verdict

    text = normalise(text)
    verdict.prediction_language = _matches(text, PREDICTION_PATTERNS)
    verdict.decision_language = _matches(text, DECISION_PATTERNS)
    verdict.vague_language = _matches(text, VAGUE_PATTERNS)
    if grounded is not None:
        verdict.ungrounded_numbers = ungrounded_numbers(text, grounded)

    verdict.passed = not (
        verdict.prediction_language or verdict.decision_language or verdict.ungrounded_numbers
    )
    # Vague language is recorded but does not fail the note: it makes a note
    # less useful, not unsafe, and failing on it would suppress output for a
    # stylistic reason.
    if verdict.vague_language and verdict.passed:
        verdict.notes.append("hedging detected; note is weak but not unsafe")
    return verdict


DISCLAIMER = (
    "RECOMMENDATION, NOT A DECISION. Generated by an LLM from model output and "
    "loan data supplied to it. It contains no independent prediction and no "
    "credit decision. A human reviewer is responsible for any action taken."
)


def wrap(text: str, verdict: GuardrailVerdict) -> str:
    """
    Attach the mandatory disclaimer, and say so when the note failed a check.

    The disclaimer goes on both ends deliberately. A reviewer scanning a queue
    reads the first line; a reviewer pasting a note into a case file carries
    the last one with it.
    """
    header = f"[{DISCLAIMER}]"
    if not verdict.passed:
        header += f"\n[GUARDRAIL FAILED -- {verdict.summary()}. Withheld from the reviewer queue.]"
    return f"{header}\n\n{text.strip()}\n\n[{DISCLAIMER}]"
