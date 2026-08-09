"""Tests for GeminiAgentClient: structured-output parsing and retry-then-degrade.

No network: the genai SDK object inside the client is replaced with a scripted
stand-in that replays responses/exceptions and records every call. Retry tests
strip tenacity's exponential-jitter waits so the suite never sleeps.
"""

from types import SimpleNamespace

import httpx
import pytest
import tenacity
from google.genai import errors

import src.agent.llm as llm_module
from src.agent.llm import GeminiAgentClient
from src.agent.schemas import GroundedAnswer, RouteDecision
from src.exceptions import LLMUnavailableError


def api_error(cls: type[errors.APIError], code: int, message: str = "boom") -> errors.APIError:
    return cls(code, {"error": {"message": message, "status": "STATUS"}})


def response(parsed: object = None, text: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(parsed=parsed, text=text)


class ScriptedModels:
    """Stand-in for genai's ``client.aio.models``: replays scripted outcomes."""

    def __init__(self, outcomes: list) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict] = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        assert self._outcomes, "generate_content called more times than scripted"
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def make_client(outcomes: list) -> tuple[GeminiAgentClient, ScriptedModels]:
    client = GeminiAgentClient(api_key="test-key", model="test-model")
    models = ScriptedModels(outcomes)
    client._client = SimpleNamespace(aio=SimpleNamespace(models=models))
    return client, models


@pytest.fixture
def no_backoff(monkeypatch):
    """Keep the real retry loop but strip the exponential-jitter sleeps."""
    real_retrying = llm_module.AsyncRetrying

    def waitless(**kwargs):
        kwargs["wait"] = tenacity.wait_none()
        return real_retrying(**kwargs)

    monkeypatch.setattr(llm_module, "AsyncRetrying", waitless)


# -- parsing -------------------------------------------------------------------


async def test_returns_parsed_schema_instance_as_is_and_sends_the_right_call():
    decision = RouteDecision(route="knowledge_base", confidence=0.9)
    client, models = make_client([response(parsed=decision, text=decision.model_dump_json())])

    result = await client.generate_structured(system="sys", user="usr", schema=RouteDecision)

    assert result is decision
    assert len(models.calls) == 1
    call = models.calls[0]
    assert call["model"] == "test-model"
    assert call["contents"] == "usr"
    config = call["config"]
    assert config.system_instruction == "sys"
    assert config.response_mime_type == "application/json"
    assert config.response_schema is RouteDecision
    assert config.temperature == 0.0


@pytest.mark.parametrize(
    "parsed",
    [None, {"answer": "ок"}],
    ids=["parsed-none", "parsed-not-a-model"],
)
async def test_falls_back_to_validating_text_when_parsed_is_unusable(parsed):
    answer = GroundedAnswer(
        answer="ок", sources=["leave_policy.md"], is_grounded=True, confidence=0.8
    )
    client, _ = make_client([response(parsed=parsed, text=answer.model_dump_json())])

    result = await client.generate_structured(system="s", user="u", schema=GroundedAnswer)

    assert isinstance(result, GroundedAnswer)
    assert result == answer


@pytest.mark.parametrize(
    "bad_text",
    ["это не JSON", "", None, '{"route": "nope", "confidence": 5}'],
    ids=["garbage", "empty", "missing", "valid-json-wrong-schema"],
)
async def test_unusable_text_raises_llm_unavailable(bad_text):
    client, models = make_client([response(parsed=None, text=bad_text)])

    with pytest.raises(LLMUnavailableError, match="unusable answer") as exc_info:
        await client.generate_structured(system="s", user="u", schema=RouteDecision)

    assert exc_info.value.status_code == 503
    assert exc_info.value.code == "llm_unavailable"
    assert len(models.calls) == 1  # a bad payload is not a reason to retry


# -- retries -------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_error",
    [
        lambda: api_error(errors.ClientError, 400, "bad request"),
        lambda: RuntimeError("unexpected SDK failure"),
    ],
    ids=["client-error-400", "unexpected-exception"],
)
async def test_non_retryable_error_fails_fast_after_one_call(no_backoff, make_error):
    error = make_error()
    client, models = make_client([error])

    with pytest.raises(LLMUnavailableError, match="Gemini API is unavailable") as exc_info:
        await client.generate_structured(system="s", user="u", schema=RouteDecision)

    assert len(models.calls) == 1
    assert exc_info.value.__cause__ is error


@pytest.mark.parametrize(
    "make_error",
    [
        lambda: api_error(errors.ServerError, 503, "unavailable"),
        lambda: api_error(errors.ClientError, 429, "rate limited"),
        lambda: httpx.ConnectError("connection refused"),
        lambda: httpx.ReadTimeout("read timed out"),
    ],
    ids=["server-503", "rate-limit-429", "connect-error", "read-timeout"],
)
async def test_retryable_error_is_retried_three_times_then_degrades(no_backoff, make_error):
    outcomes = [make_error() for _ in range(3)]
    client, models = make_client(outcomes)

    with pytest.raises(LLMUnavailableError, match="Gemini API is unavailable") as exc_info:
        await client.generate_structured(system="s", user="u", schema=RouteDecision)

    assert len(models.calls) == 3  # stop_after_attempt(3) exhausted
    assert exc_info.value.__cause__ is outcomes[-1]  # reraise keeps the last original


async def test_recovers_when_a_retry_succeeds(no_backoff):
    decision = RouteDecision(route="escalate", escalation_reason="причина", confidence=1.0)
    client, models = make_client(
        [
            api_error(errors.ServerError, 503),
            response(parsed=decision, text=decision.model_dump_json()),
        ]
    )

    result = await client.generate_structured(system="s", user="u", schema=RouteDecision)

    assert result is decision
    assert len(models.calls) == 2
