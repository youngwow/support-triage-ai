from typing import Optional


class AppError(Exception):
    """Base class for every expected failure in the application."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, detail: Optional[str] = None) -> None:
        self.detail = detail or self.__class__.__doc__ or "Unexpected error"
        super().__init__(self.detail)


class EntityNotFoundError(AppError):
    """The requested entity does not exist."""

    status_code = 404
    code = "not_found"


class InvalidRequestError(AppError):
    """The request is well-formed but violates a business rule."""

    status_code = 422
    code = "invalid_request"


class RepositoryUnavailableError(AppError):
    """The backing store cannot be reached."""

    status_code = 503
    code = "repository_unavailable"


class LLMUnavailableError(AppError):
    """The LLM provider did not answer after all retries."""

    status_code = 503
    code = "llm_unavailable"


class WebhookForbiddenError(AppError):
    """The webhook secret token does not match."""

    status_code = 403
    code = "webhook_forbidden"


class BotNotConfiguredError(AppError):
    """The Telegram bot token is not configured."""

    status_code = 503
    code = "telegram_not_configured"
