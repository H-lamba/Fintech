"""
The LLM client, and the audit trail it is required to leave.

Governance requirements baked in, not bolted on:

* **Every call is logged** -- system prompt, user prompt, model, timestamp,
  latency, token usage, output, and the guardrail verdict -- to
  ``reports/llm_prompt_log.jsonl``. Logging happens on failure too: a call that
  errored or was refused is exactly the one a reviewer will ask about.
* **The system prompt forbids prediction.** The challenge disqualifies a
  solution that uses an LLM to classify, so the model is told it is a
  summariser and the :mod:`src.copilot.guardrails` layer checks that it behaved
  like one. The prompt is a request; the check is the control.
* **Offline mode** runs the whole pipeline with no API key and no network call,
  emitting deterministic responses that are *marked as such* in the log
  (``provider: "offline-stub"``). It exists so CI and a reviewer without a key
  can exercise the assembly, guardrails and logging, and it is never
  represented as a real model response.

Failure handling
----------------
Rate limits and transient server errors are retried with exponential backoff;
authentication and malformed-request errors are not, because retrying them
just burns quota to reach the same answer. A response that is not valid JSON
where JSON was requested is caught and recorded rather than raised, so one
malformed reply cannot end a batch.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()

# Any OpenAI-compatible provider works; the key itself says which one. This
# beats a hardcoded default because the failure it prevents is silent and
# confusing: a Groq key pointed at xAI's endpoint returns "Incorrect API key
# provided", which reads as a bad key rather than a wrong endpoint.
PROVIDERS = {
    "gsk_": ("https://api.groq.com/openai/v1", "openai/gpt-oss-120b", "groq"),
    "xai-": ("https://api.x.ai/v1", "grok-4", "xai"),
    "sk-": ("https://api.openai.com/v1", "gpt-4o-mini", "openai"),
}

# Read from .env only -- nothing here is ever hardcoded. XAI_API_KEY is kept as
# an alias because that is the name the project's existing .env uses; the key it
# holds is not necessarily an xAI key, which is exactly why the provider is
# detected from the key rather than from the variable's name.
ENV_KEY_NAMES = ("LLM_API_KEY", "XAI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY")

LLM_API_KEY = next((os.getenv(n) for n in ENV_KEY_NAMES if os.getenv(n)), None)
ENV_KEY_USED = next((n for n in ENV_KEY_NAMES if os.getenv(n)), None)


def _detect_provider(key: str | None) -> tuple[str, str, str]:
    """Base URL, default model and provider name inferred from the key prefix."""
    for prefix, settings in PROVIDERS.items():
        if key and key.startswith(prefix):
            return settings
    return ("https://api.x.ai/v1", "grok-4", "unknown")


_BASE_URL, _MODEL, PROVIDER_NAME = _detect_provider(LLM_API_KEY)

# An explicit setting in .env always wins over detection.
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or _BASE_URL
LLM_MODEL = os.getenv("LLM_MODEL") or _MODEL
LOG_PATH = Path(os.getenv("LLM_LOG_PATH") or "reports/llm_prompt_log.jsonl")

SYSTEM_PROMPT = (
    "You are a loan-portfolio review assistant. Your only job is to SUMMARISE, "
    "for a human reviewer, information that has already been computed and is "
    "supplied to you in the user message.\n\n"
    "Rules you must follow:\n"
    "1. Ground every statement strictly in the supplied context. Never invent or "
    "estimate a loan ID, balance, rate, date, probability, score or field "
    "definition that is not written in the context.\n"
    "2. DO NOT classify, predict, score or forecast anything. The predictions "
    "were produced by separate statistical models and are given to you. You may "
    "restate them. You may not produce your own.\n"
    "3. DO NOT state a decision or a recommended outcome for the loan. You may "
    "describe what a reviewer should verify. You may not say what they should "
    "decide.\n"
    "4. If the context does not contain something you are asked about, say so "
    "explicitly. Refusing to answer is always preferable to guessing.\n"
    "5. Be specific and concise. Quote the figures you were given rather than "
    "characterising them vaguely."
)


class RetryableLLMError(RuntimeError):
    """A transient failure worth retrying: rate limit, timeout, 5xx."""


class FatalLLMError(RuntimeError):
    """A failure retrying cannot fix: bad key, malformed request."""


@dataclass
class LLMResponse:
    """One copilot response plus the audit metadata Task 7 requires."""

    call_id: str
    timestamp: str
    model: str
    provider: str
    system_prompt: str
    prompt: str
    output: str = ""
    status: str = "ok"
    latency_seconds: float | None = None
    usage: dict = field(default_factory=dict)
    error: str | None = None
    purpose: str = "reviewer_note"
    context_json: str | None = None
    guardrails: dict = field(default_factory=dict)
    human_review: dict = field(default_factory=dict)
    disclaimer: str = "RECOMMENDATION, NOT A DECISION."
    # "call" for the record written the moment the call returns, "review" for
    # the follow-up carrying the guardrail verdict. See log_verdict below.
    record_type: str = "call"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def log_call(response: LLMResponse, path: Path | str = LOG_PATH) -> None:
    """
    Append one call to the audit trail.

    Append-only and one JSON object per line: a log a reviewer can `grep`, and
    one that a crashed run cannot truncate retroactively.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(response.to_dict(), default=str) + "\n")


def log_verdict(response: LLMResponse, path: Path | str = LOG_PATH) -> None:
    """
    Append the guardrail verdict for a call that has already been logged.

    Two records rather than one mutated record, because the log is append-only
    and that is the property that makes it evidence. The first is written the
    instant the call returns, so a crash between the call and the check cannot
    erase the fact that the call happened; the second carries the verdict and
    the human decision. A reader joins them on ``call_id``.
    """
    review = LLMResponse(
        call_id=response.call_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        model=response.model,
        provider=response.provider,
        system_prompt="",
        prompt="",
        output="",
        status=response.status,
        purpose=response.purpose,
        guardrails=response.guardrails,
        human_review=response.human_review,
        record_type="review",
    )
    log_call(review, path)


def load_prompt_log(path: Path | str = LOG_PATH) -> list[dict[str, Any]]:
    """
    Read the audit trail back.

    A malformed line is skipped rather than raising: the log is evidence, and a
    reader that refuses to open it because one line is truncated is worse than
    one that reports 99 of 100 calls.
    """
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def has_credentials() -> bool:
    return bool(LLM_API_KEY and LLM_API_KEY != "your_xai_api_key_here")


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------
@retry(
    retry=retry_if_exception_type(RetryableLLMError),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
def _invoke_live(
    system_prompt: str, prompt: str, model: str, temperature: float, max_tokens: int
) -> tuple[str, dict]:
    """
    One live call, with backoff on the failures that backoff can fix.

    Rate limits and 5xx are retried; a bad key or a malformed request is raised
    immediately, because retrying it burns quota to reach the same answer.
    """
    from openai import (
        APIConnectionError,
        APIStatusError,
        AuthenticationError,
        BadRequestError,
        OpenAI,
        RateLimitError,
    )

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=60.0)
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except (RateLimitError, APIConnectionError) as exc:
        raise RetryableLLMError(f"{type(exc).__name__}: {exc}") from exc
    except (AuthenticationError, BadRequestError) as exc:
        raise FatalLLMError(f"{type(exc).__name__}: {exc}") from exc
    except APIStatusError as exc:
        if 500 <= getattr(exc, "status_code", 0) < 600:
            raise RetryableLLMError(f"{type(exc).__name__}: {exc}") from exc
        raise FatalLLMError(f"{type(exc).__name__}: {exc}") from exc

    usage = {}
    if getattr(completion, "usage", None):
        usage = {
            "prompt_tokens": getattr(completion.usage, "prompt_tokens", None),
            "completion_tokens": getattr(completion.usage, "completion_tokens", None),
            "total_tokens": getattr(completion.usage, "total_tokens", None),
        }
    return (completion.choices[0].message.content or ""), usage


def _invoke_offline(prompt: str, purpose: str) -> tuple[str, dict]:
    """
    A deterministic stand-in, used when no key is configured.

    It restates only what the prompt already contains, so it exercises the
    assembly, guardrail and logging paths without pretending to be a model
    response. Every offline call is recorded with ``provider: offline-stub``;
    nothing in the report presents one as a real generation.
    """
    loan = "unknown"
    month = "unknown"
    for line in prompt.splitlines():
        if line.startswith("Loan ID:"):
            loan = line.split(":", 1)[1].strip()
        if line.startswith("Reporting month:"):
            month = line.split(":", 1)[1].strip()

    rules_fired = "None." not in prompt.split("## Validation rules that fired")[-1][:40]

    body = (
        f"[OFFLINE STUB -- not a model response] Loan {loan}, reporting month {month}. "
        + (
            "One or more validation rules fired on this record; see the rule table above "
            "for the specific checks and their severities. "
            if rules_fired
            else "No validation rules fired on this record. "
        )
        + "The model outputs supplied above are restated without alteration. A reviewer "
        "should verify the flagged fields against the servicer's source file."
    )
    return body, {"prompt_tokens": len(prompt) // 4, "completion_tokens": len(body) // 4}


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def call_llm(
    prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 700,
    purpose: str = "reviewer_note",
    context_json: str | None = None,
    offline: bool | None = None,
    log_path: Path | str = LOG_PATH,
) -> LLMResponse:
    """
    Make one grounded call and log it, whatever happens.

    ``prompt`` must already carry the retrieved context. Nothing here relies on
    the model's parametric knowledge of loan data -- that is precisely the
    failure mode the challenge penalises.

    ``offline`` defaults to "whenever there is no usable key", so a machine
    without credentials runs the pipeline instead of failing it.
    """
    use_offline = (not has_credentials()) if offline is None else offline
    model_name = "offline-stub" if use_offline else (model or LLM_MODEL)

    response = LLMResponse(
        call_id=str(uuid.uuid4()),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        model=model_name,
        provider="offline-stub" if use_offline else LLM_BASE_URL,
        system_prompt=system_prompt,
        prompt=prompt,
        purpose=purpose,
        context_json=context_json,
    )

    started = time.perf_counter()
    try:
        if use_offline:
            response.output, response.usage = _invoke_offline(prompt, purpose)
            response.status = "offline_stub"
        else:
            response.output, response.usage = _invoke_live(
                system_prompt, prompt, model_name, temperature, max_tokens
            )
    except Exception as exc:  # noqa: BLE001 -- the audit trail matters more than the traceback
        response.status = "error"
        response.error = f"{type(exc).__name__}: {exc}"
    finally:
        response.latency_seconds = round(time.perf_counter() - started, 3)
        log_call(response, log_path)

    return response


def parse_json_output(response: LLMResponse) -> tuple[dict | None, str | None]:
    """
    Parse a response expected to be JSON, without letting it end a batch.

    Models wrap JSON in prose and fences often enough that a bare
    ``json.loads`` is not a real parser. Returns ``(payload, error)``; the
    caller decides what a failure means, and the raw text is still in the log
    either way.
    """
    text = (response.output or "").strip()
    if not text:
        return None, "empty response"

    if "```" in text:
        blocks = [b for b in text.split("```") if b.strip()]
        for block in blocks:
            candidate = block[4:] if block.lower().startswith("json") else block
            try:
                return json.loads(candidate.strip()), None
            except json.JSONDecodeError:
                continue

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1]), None
        except json.JSONDecodeError as exc:
            return None, f"JSONDecodeError: {exc}"

    return None, "no JSON object found in the response"
