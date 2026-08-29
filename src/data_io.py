"""
Loading and light normalisation of the data pack.

Deliberately tolerant: the organiser's real files may differ slightly from the
schema sketched in the problem statement, so missing columns are reported
rather than raised, and every loader returns whatever actually arrived.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config


def _coerce_dates(df: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    for col in candidates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_csv(path: Path, parse_dates: bool = True) -> pd.DataFrame:
    """Load one CSV, coercing known date columns. Returns empty frame if absent."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if parse_dates:
        df = _coerce_dates(df, config.EXPECTED_DATES)
    return df


def load_train() -> pd.DataFrame:
    return load_csv(config.TRAIN_PATH)


def load_test() -> pd.DataFrame:
    return load_csv(config.TEST_PATH)


def load_static() -> pd.DataFrame:
    return load_csv(config.STATIC_PATH)


def load_servicer_updates() -> pd.DataFrame:
    return load_csv(config.SERVICER_PATH)


def load_validation_rules() -> list[dict]:
    """
    Read validation_rules.json. Accepts either a bare list of rules or a dict
    with a 'rules' key, since the organiser's exact shape isn't published yet.
    """
    path = config.VALIDATION_RULES_PATH
    if not path.exists():
        return []
    with open(path) as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        return payload.get("rules", [])
    return payload if isinstance(payload, list) else []


def load_data_dictionary() -> dict[str, str]:
    """
    Parse data_dictionary.md into {field_name: definition}.

    Handles the two common layouts: a markdown table, or '- `field`: definition'
    bullet lines. Used for LLM grounding in Task 7, so it belongs in the
    profiling phase where we first read it.
    """
    path = config.DATA_DICTIONARY_PATH
    if not path.exists():
        return {}

    text = path.read_text()
    definitions: dict[str, str] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # markdown table row: | field | description | ...
        if line.startswith("|") and line.count("|") >= 3:
            cells = [c.strip().strip("`") for c in line.strip("|").split("|")]
            if len(cells) >= 2 and cells[0] and not set(cells[0]) <= set("-: "):
                if cells[0].lower() not in {"field", "column", "name"}:
                    definitions[cells[0]] = cells[1]
            continue

        # bullet line: - `field`: description
        if line.startswith(("-", "*")) and ":" in line:
            body = line.lstrip("-* ").strip()
            field, _, desc = body.partition(":")
            field = field.strip().strip("`")
            if field and desc.strip():
                definitions[field] = desc.strip()

    return definitions


def schema_report(df: pd.DataFrame, name: str) -> dict:
    """
    Compare what arrived against what the problem statement led us to expect.
    Surfaces schema drift early instead of at model-training time.
    """
    expected = set(
        config.EXPECTED_NUMERIC
        + config.EXPECTED_CATEGORICAL
        + config.EXPECTED_FLAGS
        + config.EXPECTED_DATES
        + [config.ID_COL]
    )
    actual = set(df.columns)
    return {
        "dataset": name,
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "expected_but_missing": sorted(expected - actual),
        "present_but_unexpected": sorted(actual - expected - set(config.TARGET_COLS)),
        "targets_present": sorted(actual & set(config.TARGET_COLS)),
    }
