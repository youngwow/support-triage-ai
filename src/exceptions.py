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
    """The external LLM API is unreachable or returned an unusable answer.

    Never surfaced to a user as a failure: the triage path catches it and falls
    back to the rule classifier, and the draft path catches it and routes the
    ticket to an operator.
    """

    status_code = 503
    code = "llm_unavailable"


class KnowledgeBaseUnavailableError(AppError):
    """The vector index is not loaded, so retrieval cannot run."""

    status_code = 503
    code = "knowledge_base_unavailable"
