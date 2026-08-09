from functools import lru_cache
from typing import Annotated, Optional

from aiogram import Bot, Dispatcher
from fastapi import Depends
from langgraph.graph.state import CompiledStateGraph

from src.agent.graph import build_agent_graph
from src.agent.llm import GeminiAgentClient
from src.config import Settings, get_settings
from src.rag.embedder import Embedder, GigaEmbedder
from src.repositories.dialog_memory import AbstractDialogMemory, InMemoryDialogMemory
from src.repositories.hr_system import AbstractHRSystem, StubHRSystem
from src.repositories.knowledge_base import AbstractKnowledgeBase, FaissKnowledgeBase
from src.services.assistant_service import AssistantService
from src.services.health import HealthService
from src.telegram.dedup import UpdateDeduplicator
from src.telegram.handlers import create_dispatcher


SettingsDep = Annotated[Settings, Depends(get_settings)]


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    return GigaEmbedder(
        settings.embedding_model_name,
        device=settings.embedding_device,
        hf_token=settings.hf_token,
        batch_size=settings.embedding_batch_size,
    )


@lru_cache
def get_knowledge_base() -> AbstractKnowledgeBase:
    embedder = get_embedder()
    settings = get_settings()
    return FaissKnowledgeBase(
        embedder=embedder, 
        data_dir=settings.data_dir
    )


@lru_cache
def get_hr_system() -> AbstractHRSystem:
    return StubHRSystem()


@lru_cache
def get_dialog_memory() -> AbstractDialogMemory:
    settings = get_settings()
    return InMemoryDialogMemory(
        max_messages=settings.history_max_turns
    )


@lru_cache
def get_llm_client() -> GeminiAgentClient:
    settings = get_settings()
    return GeminiAgentClient(
        api_key=settings.gemini_api_key, 
        model=settings.gemini_model
    )


@lru_cache
def get_agent_graph() -> CompiledStateGraph:
    knowledge_base = get_knowledge_base()
    hr_system = get_hr_system()
    llm = get_llm_client()
    settings = get_settings()
    return build_agent_graph(
        knowledge_base=knowledge_base,
        hr_system=hr_system,
        llm=llm,
        settings=settings,
    )


def get_assistant_service() -> AssistantService:
    graph = get_agent_graph()
    dialog_memory = get_dialog_memory()
    knowledge_base = get_knowledge_base()
    settings = get_settings()
    return AssistantService(
        graph=graph,
        dialog_memory=dialog_memory,
        knowledge_base=knowledge_base,
        settings=settings,
    )


@lru_cache
def get_bot() -> Optional[Bot]:
    settings = get_settings()
    if not settings.telegram_bot_api_key:
        return None
    return Bot(
        token=settings.telegram_bot_api_key
    )


@lru_cache
def get_dispatcher() -> Optional[Dispatcher]:
    if get_bot() is None:
        return None
    assistant_service = get_assistant_service()
    return create_dispatcher(assistant_service)


@lru_cache
def get_update_deduplicator() -> UpdateDeduplicator:
    return UpdateDeduplicator()


KnowledgeBaseDep = Annotated[AbstractKnowledgeBase, Depends(get_knowledge_base)]
HRSystemDep = Annotated[AbstractHRSystem, Depends(get_hr_system)]
DialogMemoryDep = Annotated[AbstractDialogMemory, Depends(get_dialog_memory)]
AssistantServiceDep = Annotated[AssistantService, Depends(get_assistant_service)]
BotDep = Annotated[Optional[Bot], Depends(get_bot)]
DispatcherDep = Annotated[Optional[Dispatcher], Depends(get_dispatcher)]
UpdateDeduplicatorDep = Annotated[UpdateDeduplicator, Depends(get_update_deduplicator)]


def get_health_service(
    settings: SettingsDep,
    knowledge_base: KnowledgeBaseDep,
    hr_system: HRSystemDep,
    dialog_memory: DialogMemoryDep,
) -> HealthService:
    return HealthService(
        settings,
        probes={
            "knowledge_base": knowledge_base,
            "hr_system": hr_system,
            "dialog_memory": dialog_memory,
        },
    )


HealthServiceDep = Annotated[HealthService, Depends(get_health_service)]
