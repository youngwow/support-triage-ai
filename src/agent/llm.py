from typing import Protocol, TypeVar

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.exceptions import LLMUnavailableError
from src.utils import get_logger


logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_RETRYABLE_STATUS_CODES = (408, 429, 500, 502, 503, 504)
_MAX_ATTEMPTS = 3


class SupportsStructuredGeneration(Protocol):
    """
    What the agent nodes need from an LLM — lets tests script the answers.
    """

    async def generate_structured(self, *, system: str, user: str, schema: type[T]) -> T: ...


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, errors.APIError):
        return exc.code in _RETRYABLE_STATUS_CODES
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


class GeminiAgentClient:
    def __init__(self, *, api_key: str, model: str) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def generate_structured(self, *, system: str, user: str, schema: type[T]) -> T:
        """
        One LLM call constrained to ``schema``; retried, never half-answered.

        Raises :class:`LLMUnavailableError` when the provider stays down after
        retries or returns something that does not validate — callers decide
        whether that means degrade or escalate.
        """
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(_MAX_ATTEMPTS),
                wait=wait_exponential_jitter(initial=1, max=8),
                retry=retry_if_exception(_is_retryable),
                reraise=True,
            ):
                with attempt:
                    response = await self._client.aio.models.generate_content(
                        model=self._model,
                        contents=user,
                        config=types.GenerateContentConfig(
                            system_instruction=system,
                            response_mime_type="application/json",
                            response_schema=schema,
                            temperature=0.0,
                        ),
                    )
        except Exception as exc:
            logger.warning(f"Gemini call failed after retries: {exc}")
            raise LLMUnavailableError("Gemini API is unavailable") from exc

        parsed = response.parsed
        if isinstance(parsed, schema):
            return parsed
        try:
            return schema.model_validate_json(response.text or "")
        except ValidationError as exc:
            logger.warning(f"Gemini returned unparseable structured output: {exc}")
            raise LLMUnavailableError("Gemini returned an unusable answer") from exc
