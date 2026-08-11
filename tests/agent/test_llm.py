"""
The degradation contract of the single seam to the LLM provider.

Everything above ``src/agent/llm.py`` is allowed to know exactly one failure
mode — :class:`LLMUnavailableError` — so this file pins down *when* that error is
raised, how many provider calls it costs, and that the original cause survives.

No test here touches a network: the SDK handle (``client._client``) is replaced
by a scripted stand-in, and tenacity's backoff is swapped for ``wait_none`` so
the real retry loop runs at full speed.
"""

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import tenacity
from google.genai import errors
from pydantic import ValidationError

from src.agent import llm as llm_module
from src.agent.llm import GeminiAgentClient, NullLLMClient
from src.agent.schemas import GroundedDraft, TicketClassification
from src.exceptions import LLMUnavailableError


#: What a healthy provider call returns.
DRAFT = GroundedDraft(
    answer="Карту можно заменить в разделе «Способы оплаты».",
    sources=["payment_methods.md"],
    is_grounded=True,
    confidence=0.9,
)
DRAFT_JSON = DRAFT.model_dump_json()


def api_error(code: int) -> errors.APIError:
    """The error the google-genai SDK raises for an HTTP status."""
    payload = {"error": {"code": code, "message": f"provider said {code}", "status": "ERROR"}}
    error_class = errors.ClientError if code < 500 else errors.ServerError
    return error_class(code, payload)


def response(*, parsed: Any = None, text: str | None = None) -> SimpleNamespace:
    """A stand-in for ``GenerateContentResponse``: only two fields are read."""
    return SimpleNamespace(parsed=parsed, text=text)


class ScriptedModels:
    """Stands in for ``client.aio.models`` — replays a script, records the calls.

    The script is index-clamped: once exhausted the last entry repeats, so
    ``ScriptedModels(error)`` fails every attempt while
    ``ScriptedModels(error, ok)`` fails once and then succeeds.
    """

    def __init__(self, *script: Any) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self._script[min(len(self.calls) - 1, len(self._script) - 1)]
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the real retry loop, drop the sleeping between attempts."""
    monkeypatch.setattr(
        llm_module, "wait_exponential_jitter", lambda **kwargs: tenacity.wait_none()
    )


@pytest.fixture
def client() -> GeminiAgentClient:
    """A real client object; its SDK handle is replaced before every call."""
    return GeminiAgentClient(api_key="test-key", model="gemini-test", timeout_ms=1_000)


def attach(client: GeminiAgentClient, models: ScriptedModels) -> ScriptedModels:
    """Replace the SDK handle so nothing can reach the network."""
    client._client = SimpleNamespace(aio=SimpleNamespace(models=models))
    return models


# --- happy path ----------------------------------------------------------


async def test_returns_the_parsed_object_on_the_first_attempt(
    client: GeminiAgentClient,
) -> None:
    models = attach(client, ScriptedModels(response(parsed=DRAFT, text=DRAFT_JSON)))

    result = await client.generate_structured(system="s", user="u", schema=GroundedDraft)

    assert result == DRAFT
    assert len(models.calls) == 1


async def test_sends_the_system_prompt_user_prompt_and_schema_to_the_provider(
    client: GeminiAgentClient,
) -> None:
    models = attach(client, ScriptedModels(response(parsed=DRAFT)))

    await client.generate_structured(
        system="ты помощник", user="как поменять карту?", schema=GroundedDraft
    )

    call = models.calls[0]
    config = call["config"]
    assert call["model"] == "gemini-test"
    assert call["contents"] == "как поменять карту?"
    assert config.system_instruction == "ты помощник"
    assert config.response_schema is GroundedDraft
    assert config.response_mime_type == "application/json"
    assert config.temperature == 0.0


# --- retryable failures --------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        api_error(503),
        api_error(429),
        httpx.ConnectError("connection refused"),
        httpx.TimeoutException("read timed out"),
    ],
    ids=["api_503", "api_429", "connect_error", "timeout"],
)
async def test_retryable_error_is_attempted_three_times_then_reported_unavailable(
    client: GeminiAgentClient, error: Exception
) -> None:
    models = attach(client, ScriptedModels(error))

    with pytest.raises(LLMUnavailableError, match="Gemini API is unavailable") as caught:
        await client.generate_structured(system="s", user="u", schema=GroundedDraft)

    assert len(models.calls) == 3
    assert caught.value.__cause__ is error


async def test_a_retryable_error_that_clears_is_not_reported_as_a_failure(
    client: GeminiAgentClient,
) -> None:
    models = attach(
        client, ScriptedModels(api_error(503), response(parsed=DRAFT, text=DRAFT_JSON))
    )

    result = await client.generate_structured(system="s", user="u", schema=GroundedDraft)

    assert result == DRAFT
    assert len(models.calls) == 2


# --- non-retryable failures ----------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        api_error(400),
        api_error(404),
        RuntimeError("something nobody predicted"),
    ],
    ids=["api_400", "api_404", "unexpected_exception"],
)
async def test_non_retryable_error_fails_on_the_first_attempt(
    client: GeminiAgentClient, error: Exception
) -> None:
    models = attach(client, ScriptedModels(error))

    with pytest.raises(LLMUnavailableError, match="Gemini API is unavailable") as caught:
        await client.generate_structured(system="s", user="u", schema=GroundedDraft)

    assert len(models.calls) == 1
    assert caught.value.__cause__ is error


async def test_the_failure_carries_the_status_and_code_callers_degrade_on(
    client: GeminiAgentClient,
) -> None:
    attach(client, ScriptedModels(api_error(400)))

    with pytest.raises(LLMUnavailableError) as caught:
        await client.generate_structured(system="s", user="u", schema=GroundedDraft)

    assert caught.value.status_code == 503
    assert caught.value.code == "llm_unavailable"


# --- unusable answers ----------------------------------------------------


@pytest.mark.parametrize(
    "parsed",
    [
        None,
        {"answer": "x", "sources": [], "is_grounded": True, "confidence": 0.5},
        TicketClassification(category="other", risk="low", confidence=0.5),
    ],
    ids=["parsed_is_none", "parsed_is_a_dict", "parsed_is_another_schema"],
)
async def test_falls_back_to_parsing_the_raw_text_when_parsed_is_unusable(
    client: GeminiAgentClient, parsed: Any
) -> None:
    models = attach(client, ScriptedModels(response(parsed=parsed, text=DRAFT_JSON)))

    result = await client.generate_structured(system="s", user="u", schema=GroundedDraft)

    assert result == DRAFT
    assert len(models.calls) == 1


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "   ",
        "I am afraid I cannot answer that.",
        "```json\n" + DRAFT_JSON + "\n```",
        '{"answer": "ok"}',
        '{"answer": "ok", "sources": [], "is_grounded": true, "confidence": 1.5}',
    ],
    ids=[
        "no_text",
        "empty_text",
        "blank_text",
        "prose_instead_of_json",
        "fenced_json",
        "missing_required_fields",
        "confidence_out_of_range",
    ],
)
async def test_unparseable_answer_is_reported_as_unavailable_without_retrying(
    client: GeminiAgentClient, text: str | None
) -> None:
    models = attach(client, ScriptedModels(response(parsed=None, text=text)))

    with pytest.raises(LLMUnavailableError, match="unusable answer") as caught:
        await client.generate_structured(system="s", user="u", schema=GroundedDraft)

    assert len(models.calls) == 1
    assert isinstance(caught.value.__cause__, ValidationError)


# --- the unconfigured deployment -----------------------------------------


@pytest.mark.parametrize(
    "schema", [GroundedDraft, TicketClassification], ids=["draft", "classification"]
)
async def test_null_client_always_reports_the_llm_as_unavailable(
    schema: type[GroundedDraft] | type[TicketClassification],
) -> None:
    null_client = NullLLMClient()

    with pytest.raises(LLMUnavailableError, match="GEMINI_API_KEY is not configured") as caught:
        await null_client.generate_structured(system="s", user="u", schema=schema)

    assert caught.value.status_code == 503
    assert caught.value.code == "llm_unavailable"
