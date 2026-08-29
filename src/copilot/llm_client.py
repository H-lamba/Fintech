"""
Thin, swappable LLM client for the Reviewer Copilot (Task 7).

Uses xAI's Grok API through its OpenAI-compatible endpoint (https://api.x.ai/v1),
so the same code works against any OpenAI-compatible provider by changing
LLM_BASE_URL / LLM_MODEL in .env.

Governance requirements from the challenge (Task 7) that are baked in here:
  * every call logs prompt, model, timestamp, and output
  * output is always returned wrapped as a RECOMMENDATION, never a decision
  * the caller is responsible for passing GROUNDED context in the prompt --
    the LLM must never be asked to invent numbers it wasn't given
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

LLM_API_KEY = os.getenv("XAI_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.x.ai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "grok-4")

LOG_PATH = Path(os.getenv("LLM_LOG_PATH", "reports/llm_prompt_log.jsonl"))

SYSTEM_PROMPT = (
    "You are a loan-portfolio review assistant. You must ground every statement "
    "strictly in the context provided in the user message. "
    "Never invent loan IDs, balances, rates, dates, probabilities, or field "
    "definitions that are not present in the context. "
    "If the context does not contain enough information to answer, say so "
    "explicitly rather than guessing. "
    "Your output is a RECOMMENDATION for a human reviewer, never a final decision."
)


@dataclass
class LLMResponse:
    """A single copilot response plus the audit metadata judges want to see."""

    call_id: str
    model: str
    timestamp: float
    prompt: str
    output: str
    status: str = "ok"
    label: str = "RECOMMENDATION - requires human review, not an automated decision"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _log_call(response: LLMResponse) -> None:
    """Append-only audit trail. Required by Task 7 (prompt/model/timestamp/output)."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as fh:
        fh.write(json.dumps(response.to_dict()) + "\n")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _invoke(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content or ""


def call_llm(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> LLMResponse:
    """
    Make one grounded LLM call and log it.

    `prompt` should already contain the retrieved context: the loan record,
    the relevant data-dictionary entries, the triggered validation rules, and
    the model's own predictions. Do not rely on the LLM's parametric knowledge
    of loan data -- that is exactly the failure mode the challenge penalises.
    """
    model_name = model or LLM_MODEL
    response = LLMResponse(
        call_id=str(uuid.uuid4()),
        model=model_name,
        timestamp=time.time(),
        prompt=prompt,
        output="",
    )

    if not LLM_API_KEY:
        response.status = "error"
        response.error = "XAI_API_KEY not set. Copy .env.example to .env and add your key."
        _log_call(response)
        raise RuntimeError(response.error)

    try:
        response.output = _invoke(prompt, model_name, temperature, max_tokens)
    except Exception as exc:  # noqa: BLE001 - we want the audit trail either way
        response.status = "error"
        response.error = f"{type(exc).__name__}: {exc}"
        _log_call(response)
        raise

    _log_call(response)
    return response


def load_prompt_log(path: Path | str = LOG_PATH) -> list[dict[str, Any]]:
    """Read the audit trail back, e.g. to render it in the LLM copilot report."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]
