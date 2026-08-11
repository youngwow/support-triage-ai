"""
The draft graph — the slow path, in three nodes.

    START -> retrieve --(relevant fragments)--> generate --(grounded)--> END
                 |                                  |
                 +--(nothing relevant)--> escalate <+--(ungrounded or unsure)
                                              |
                                              v
                                             END

Deliberately small. There is no separate verification node: the generator is
asked for ``is_grounded`` and ``confidence`` in the same structured call that
produces the answer, so a ticket costs at most one generation round trip. A
second self-checking call would double the bill for a judgement the model has
already made.

The graph is compiled without a checkpointer — a draft run is a single pass with
no human interrupt inside it, and durability lives in the queue, not here.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent.llm import SupportsStructuredGeneration
from src.agent.nodes import DraftNodes
from src.agent.schemas import DraftState
from src.config import Settings
from src.repositories.knowledge_base import AbstractKnowledgeBase


def build_draft_graph(
    *,
    knowledge_base: AbstractKnowledgeBase,
    llm: SupportsStructuredGeneration,
    settings: Settings,
) -> CompiledStateGraph:
    nodes = DraftNodes(knowledge_base=knowledge_base, llm=llm, settings=settings)

    graph = StateGraph(DraftState)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("generate", nodes.generate)
    graph.add_node("escalate", nodes.escalate)

    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        nodes.route_after_retrieve,
        {"generate": "generate", "escalate": "escalate"},
    )
    graph.add_conditional_edges(
        "generate",
        nodes.route_after_generate,
        {"end": END, "escalate": "escalate"},
    )
    graph.add_edge("escalate", END)

    return graph.compile()
