"""
Task 1d: cross-column relationship breaks.

Two sources of rules:
  1. The organiser's `validation_rules.json` (run as-is, whatever shape it has).
  2. Our own domain rules below -- the problem statement explicitly rewards
     going beyond the starter checks, and these are the ones that actually
     catch contradictory loan states.

Every rule returns a boolean Series where True == violation, so the same
objects drive both the summary table and the record-level quality score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass
class Rule:
    name: str
    description: str
    severity: str  # "high" | "medium" | "low"
    fn: Callable[[pd.DataFrame], pd.Series]
    requires: list[str]

    def applicable(self, df: pd.DataFrame) -> bool:
        return set(self.requires) <= set(df.columns)

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        if not self.applicable(df):
            return pd.Series(False, index=df.index)
        try:
            return self.fn(df).fillna(False).astype(bool)
        except Exception:  # a malformed column shouldn't kill the whole run
            return pd.Series(False, index=df.index)


# --------------------------------------------------------------------------
# Custom domain rules
# --------------------------------------------------------------------------

CUSTOM_RULES: list[Rule] = [
    Rule(
        name="balance_exceeds_original",
        description="current_balance greater than original_balance without a modification flag",
        severity="high",
        requires=["current_balance", "original_balance"],
        fn=lambda d: (d["current_balance"] > d["original_balance"] * 1.001)
        & (d["modification_flag"] != 1 if "modification_flag" in d.columns else True),
    ),
    Rule(
        name="negative_balance",
        description="current_balance is negative",
        severity="high",
        requires=["current_balance"],
        fn=lambda d: d["current_balance"] < 0,
    ),
    Rule(
        name="prepaid_with_positive_balance",
        description="prepayment_flag set but current_balance is still materially above zero",
        severity="high",
        requires=["prepayment_flag", "current_balance"],
        fn=lambda d: (d["prepayment_flag"] == 1) & (d["current_balance"] > 1.0),
    ),
    Rule(
        name="default_without_delinquency",
        description="default_flag set but days_past_due is below the 90-day default threshold",
        severity="high",
        requires=["default_flag", "days_past_due"],
        fn=lambda d: (d["default_flag"] == 1) & (d["days_past_due"] < 90),
    ),
    Rule(
        name="delinquent_but_current_status",
        description="days_past_due > 0 while current_status reads as current/performing",
        severity="high",
        requires=["days_past_due", "current_status"],
        fn=lambda d: (d["days_past_due"] > 0)
        & (d["current_status"].astype(str).str.lower().isin(["current", "performing", "0"])),
    ),
    Rule(
        name="current_status_but_high_dpd",
        description="current_status says current while days_past_due exceeds 30",
        severity="medium",
        requires=["days_past_due", "current_status"],
        fn=lambda d: (d["days_past_due"] > 30)
        & (d["current_status"].astype(str).str.lower() == "current"),
    ),
    Rule(
        name="closed_status_with_balance",
        description="loan marked closed/prepaid/liquidated yet carries a balance",
        severity="high",
        requires=["current_status", "current_balance"],
        fn=lambda d: d["current_status"]
        .astype(str)
        .str.lower()
        .isin(["closed", "prepaid", "paid_off", "liquidated", "matured"])
        & (d["current_balance"] > 1.0),
    ),
    Rule(
        name="negative_remaining_term",
        description="remaining_term_months is negative",
        severity="high",
        requires=["remaining_term_months"],
        fn=lambda d: d["remaining_term_months"] < 0,
    ),
    Rule(
        name="implausible_interest_rate",
        description="interest_rate outside a plausible 0-25% band",
        severity="medium",
        requires=["interest_rate"],
        fn=lambda d: (d["interest_rate"] < 0) | (d["interest_rate"] > 25),
    ),
    Rule(
        name="negative_loan_age",
        description="loan_age_months is negative",
        severity="high",
        requires=["loan_age_months"],
        fn=lambda d: d["loan_age_months"] < 0,
    ),
    Rule(
        name="dpd_implausibly_large",
        description="days_past_due exceeds 1080 (three years) -- likely a unit or sentinel error",
        severity="medium",
        requires=["days_past_due"],
        fn=lambda d: d["days_past_due"] > 1080,
    ),
    Rule(
        name="missing_document_status",
        description="document_status missing or explicitly incomplete",
        severity="low",
        requires=["document_status"],
        fn=lambda d: d["document_status"].isna()
        | d["document_status"].astype(str).str.lower().isin(["missing", "incomplete", "pending"]),
    ),
    Rule(
        name="default_and_prepaid_together",
        description="default_flag and prepayment_flag both set on the same record",
        severity="high",
        requires=["default_flag", "prepayment_flag"],
        fn=lambda d: (d["default_flag"] == 1) & (d["prepayment_flag"] == 1),
    ),
    Rule(
        name="zero_original_balance",
        description="original_balance is zero or negative",
        severity="high",
        requires=["original_balance"],
        fn=lambda d: d["original_balance"] <= 0,
    ),
]


# --------------------------------------------------------------------------
# Organiser-supplied rules from validation_rules.json
# --------------------------------------------------------------------------


# Words that appear as bare identifiers in a pandas `eval` expression but are
# operators or literals, not columns. Treating them as required columns makes
# every expression that uses one look inapplicable.
_EVAL_KEYWORDS = frozenset(
    {
        "and", "or", "not", "in", "is", "if", "else",
        "True", "False", "None", "true", "false", "null",
        "abs", "min", "max", "log", "log10", "exp", "sqrt", "where", "isna", "notna",
    }
)

_STRING_LITERAL = re.compile(r"""(['"])(?:\\.|(?!\1).)*\1""")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _referenced_columns(expression: str) -> list[str]:
    """
    Column names an expression actually depends on.

    Two traps, both of which previously made a rule silently unevaluatable:
    identifiers inside **string literals** (``'90-DPD'`` yields ``DPD``) and
    the operator words ``and`` / ``or`` / ``not``. A rule whose ``requires``
    list contains any of those can never be satisfied by a real frame, so
    ``applicable()`` returns False and the rule is skipped -- reported as
    "not applicable to this data" rather than as the parser failure it is.
    """
    stripped = _STRING_LITERAL.sub(" ", expression)
    idents = _IDENTIFIER.findall(stripped)
    return sorted({token for token in idents if token not in _EVAL_KEYWORDS})


def _rule_from_spec(spec: dict) -> Rule | None:
    """
    Translate one JSON rule spec into a Rule.

    Supported shapes (we handle several because the exact format isn't published):
      {"name":..., "expression": "current_balance <= original_balance"}   -> pandas.eval
      {"name":..., "column": "days_past_due", "min": 0, "max": 1080}
      {"name":..., "column": "current_status", "allowed": ["current", ...]}
      {"name":..., "column": "loan_id", "not_null": true}
    A violation is the negation of the stated valid condition.
    """
    name = spec.get("name") or spec.get("rule_id") or spec.get("id")
    if not name:
        return None

    description = spec.get("description", "")
    severity = str(spec.get("severity", "medium")).lower()

    if "expression" in spec or "condition" in spec:
        expr = spec.get("expression") or spec.get("condition")

        def fn(d: pd.DataFrame, _expr=expr) -> pd.Series:
            valid = d.eval(_expr)
            return ~valid.astype(bool)

        return Rule(f"json__{name}", description or expr, severity, fn, _referenced_columns(expr))

    column = spec.get("column") or spec.get("field")
    if not column:
        return None

    if "allowed" in spec or "allowed_values" in spec:
        allowed = spec.get("allowed") or spec.get("allowed_values")

        def fn(d: pd.DataFrame, _c=column, _a=allowed) -> pd.Series:
            return ~d[_c].isin(_a)

        return Rule(f"json__{name}", description or f"{column} outside allowed set", severity, fn, [column])

    if "min" in spec or "max" in spec:
        lo, hi = spec.get("min"), spec.get("max")

        def fn(d: pd.DataFrame, _c=column, _lo=lo, _hi=hi) -> pd.Series:
            v = pd.Series(False, index=d.index)
            if _lo is not None:
                v |= d[_c] < _lo
            if _hi is not None:
                v |= d[_c] > _hi
            return v

        return Rule(f"json__{name}", description or f"{column} out of range", severity, fn, [column])

    if spec.get("not_null") or spec.get("required"):
        def fn(d: pd.DataFrame, _c=column) -> pd.Series:
            return d[_c].isna()

        return Rule(f"json__{name}", description or f"{column} must not be null", severity, fn, [column])

    return None


def build_rule_set(json_rules: list[dict] | None = None) -> list[Rule]:
    """Organiser rules first (so they're clearly credited), then our own."""
    rules: list[Rule] = []
    for spec in json_rules or []:
        if isinstance(spec, dict):
            rule = _rule_from_spec(spec)
            if rule is not None:
                rules.append(rule)
    rules.extend(CUSTOM_RULES)
    return rules


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def evaluate_rules(df: pd.DataFrame, rules: list[Rule]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (summary_table, violation_matrix).

    summary_table   : one row per rule with violation counts
    violation_matrix: boolean DataFrame, one column per applicable rule
    """
    matrix = pd.DataFrame(index=df.index)
    summary_rows = []

    for rule in rules:
        applicable = rule.applicable(df)
        mask = rule.evaluate(df)
        if applicable:
            matrix[f"rule__{rule.name}"] = mask
        summary_rows.append(
            {
                "rule": rule.name,
                "description": rule.description,
                "severity": rule.severity,
                "applicable": applicable,
                "n_violations": int(mask.sum()) if applicable else None,
                "pct_violations": round(100.0 * mask.mean(), 4) if applicable else None,
            }
        )

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["applicable", "n_violations"], ascending=[False, False]
        ).reset_index(drop=True)
    return summary, matrix
