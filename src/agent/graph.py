"""
Wiring of the LangGraph workflow.

    START -> classify --(escalate | low confidence)--> escalate
             classify --(needs balance/grade)--------> call_tools -> retrieve
             classify --(otherwise)------------------> retrieve
    retrieve --(no/weak chunks)--> escalate
    retrieve --(else)-----------> generate
    generate --(grounded & confident & clean sources)--> END
    generate --(else)-----------> escalate
    escalate -> END
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent.llm import SupportsStructuredGeneration
from src.agent.nodes import AgentNodes
from src.agent.state import AgentState
from src.agent.tools import AgentTools
from src.config import Settings
from src.repositories.hr_system import AbstractHRSystem
from src.repositories.knowledge_base import AbstractKnowledgeBase


def build_agent_graph(
    *,
    knowledge_base: AbstractKnowledgeBase,
    hr_system: AbstractHRSystem,
    llm: SupportsStructuredGeneration,
    settings: Settings,
) -> CompiledStateGraph:
    nodes = AgentNodes(
        knowledge_base=knowledge_base,
        tools=AgentTools(hr_system),
        llm=llm,
        settings=settings,
    )

    graph = StateGraph(AgentState)
    graph.add_node("classify", nodes.classify)
    graph.add_node("call_tools", nodes.call_tools)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("generate", nodes.generate)
    graph.add_node("escalate", nodes.escalate)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        nodes.route_after_classify,
        {"escalate": "escalate", "tools": "call_tools", "retrieve": "retrieve"},
    )
    graph.add_edge("call_tools", "retrieve")
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
