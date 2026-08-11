"""
Every wiring decision in the application lives here and nowhere else.

Two tiers:

* ``@lru_cache`` providers are process-wide singletons — anything holding state
  (stores, the queue, the loaded index, the provider client). **Each one must be
  registered in ``_CACHED_PROVIDERS`` in ``tests/conftest.py``**, or state leaks
  between tests and the suite becomes order-dependent.
* Plain providers are rebuilt per request. Services are cheap value objects over
  the singletons, so they belong here.

Routes declare ``service: TriageServiceDep`` and never write ``Depends(...)``
inline or construct a collaborator themselves.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from langgraph.graph.state import CompiledStateGraph

from src.agent.graph import build_draft_graph
from src.agent.llm import GeminiAgentClient, NullLLMClient, SupportsStructuredGeneration
from src.config import Settings, get_settings
from src.ml.classifier import LlmTopicClassifier, RuleTopicClassifier, TopicClassifier
from src.rag.embedder import Embedder, GigaEmbedder
from src.repositories.audit_log import AbstractAuditLog, InMemoryAuditLog
from src.repositories.draft_queue import DraftQueue
from src.repositories.knowledge_base import AbstractKnowledgeBase, FaissKnowledgeBase
from src.repositories.ticket_repository import (
    AbstractTicketRepository,
    InMemoryTicketRepository,
)
from src.services.draft_service import DraftService
from src.services.health import HealthService
from src.services.metrics_service import MetricsService
from src.services.triage_service import TriageService
from src.utils import get_logger


logger = get_logger(__name__)


# --- singletons ----------------------------------------------------------


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    return GigaEmbedder(
        settings.embedding_model_name,
        device=settings.embedding_device,
        hf_token=settings.hf_token,
        batch_size=settings.embedding_batch_size,
        max_length=settings.embedding_max_length,
    )


@lru_cache
def get_knowledge_base() -> AbstractKnowledgeBase:
    return FaissKnowledgeBase(get_embedder(), get_settings().knowledge_base_dir)


@lru_cache
def get_ticket_repository() -> AbstractTicketRepository:
    return InMemoryTicketRepository()


@lru_cache
def get_audit_log() -> AbstractAuditLog:
    return InMemoryAuditLog()


@lru_cache
def get_llm_client() -> SupportsStructuredGeneration:
    """The real client, or a null object when no key is configured.

    ``genai.Client(api_key="")`` raises, so an unconfigured deployment cannot
    hold a real client at all. Returning :class:`NullLLMClient` instead means
    the service still starts and every caller takes its normal
    ``LLMUnavailableError`` path — the PoC runs end to end without credentials.
    """
    settings = get_settings()
    if not settings.gemini_api_key:
        logger.warning("GEMINI_API_KEY is not set — running with the LLM permanently down")
        return NullLLMClient()
    return GeminiAgentClient(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        timeout_ms=settings.llm_timeout_ms,
    )


@lru_cache
def get_topic_classifier() -> TopicClassifier:
    """Swap this line to move from LLM classification to a distilled model."""
    return LlmTopicClassifier(get_llm_client())


@lru_cache
def get_draft_queue() -> DraftQueue:
    return DraftQueue(get_settings().draft_queue_maxsize)


@lru_cache
def get_draft_graph() -> CompiledStateGraph:
    return build_draft_graph(
        knowledge_base=get_knowledge_base(),
        llm=get_llm_client(),
        settings=get_settings(),
    )


def get_fallback_classifier() -> TopicClassifier:
    """Stateless, so it is rebuilt rather than cached."""
    return RuleTopicClassifier()


# --- typed aliases -------------------------------------------------------

SettingsDep = Annotated[Settings, Depends(get_settings)]
TicketRepositoryDep = Annotated[AbstractTicketRepository, Depends(get_ticket_repository)]
AuditLogDep = Annotated[AbstractAuditLog, Depends(get_audit_log)]
KnowledgeBaseDep = Annotated[AbstractKnowledgeBase, Depends(get_knowledge_base)]
TopicClassifierDep = Annotated[TopicClassifier, Depends(get_topic_classifier)]
DraftQueueDep = Annotated[DraftQueue, Depends(get_draft_queue)]
DraftGraphDep = Annotated[CompiledStateGraph, Depends(get_draft_graph)]


# --- per-request services ------------------------------------------------


def get_triage_service(
    settings: SettingsDep,
    tickets: TicketRepositoryDep,
    audit: AuditLogDep,
    classifier: TopicClassifierDep,
    queue: DraftQueueDep,
) -> TriageService:
    return TriageService(
        tickets=tickets,
        audit=audit,
        classifier=classifier,
        fallback_classifier=get_fallback_classifier(),
        queue=queue,
        settings=settings,
    )


def get_draft_service(
    settings: SettingsDep,
    tickets: TicketRepositoryDep,
    audit: AuditLogDep,
    knowledge_base: KnowledgeBaseDep,
    queue: DraftQueueDep,
    graph: DraftGraphDep,
) -> DraftService:
    return DraftService(
        tickets=tickets,
        audit=audit,
        knowledge_base=knowledge_base,
        queue=queue,
        graph=graph,
        settings=settings,
    )


def build_draft_service() -> DraftService:
    """The same wiring, for callers outside the request cycle (the worker)."""
    return get_draft_service(
        get_settings(),
        get_ticket_repository(),
        get_audit_log(),
        get_knowledge_base(),
        get_draft_queue(),
        get_draft_graph(),
    )


def get_metrics_service(settings: SettingsDep, audit: AuditLogDep) -> MetricsService:
    return MetricsService(audit=audit, settings=settings)


def get_health_service(
    settings: SettingsDep,
    tickets: TicketRepositoryDep,
    audit: AuditLogDep,
    knowledge_base: KnowledgeBaseDep,
) -> HealthService:
    return HealthService(
        settings,
        probes={
            "tickets": tickets,
            "audit": audit,
            "knowledge_base": knowledge_base,
        },
    )


TriageServiceDep = Annotated[TriageService, Depends(get_triage_service)]
DraftServiceDep = Annotated[DraftService, Depends(get_draft_service)]
MetricsServiceDep = Annotated[MetricsService, Depends(get_metrics_service)]
HealthServiceDep = Annotated[HealthService, Depends(get_health_service)]
