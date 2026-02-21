"""
LangGraph graph assembly for the tutoring agent.

Wires all nodes together with conditional routing based on
intent classification and tool call detection.

Flow: [START] → [router] → [content_node] → [tools?] → [loop/END]
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from backend.agent.nodes import (
    clarify_node,
    encourage_node,
    explain_node,
    quiz_node,
    router_node,
    summarize_node,
    tool_executor_node,
)
from backend.agent.state import TutorState
from backend.monitoring.logger import get_logger

logger = get_logger(__name__)


def route_from_router(state: TutorState) -> Literal[
    "explain", "quiz", "summarize", "encourage", "clarify", "__end__"
]:
    """Route from the router node to the appropriate content node."""
    phase = state.get("current_phase", "teaching")

    route_map = {
        "greeting": "explain",
        "topic_selection": "explain",
        "teaching": "explain",
        "quiz": "quiz",
        "review": "summarize",
        "wrap_up": "__end__",
    }

    destination = route_map.get(phase, "explain")

    # Check if encouragement is needed
    if state.get("encouragement_due", False):
        return "encourage"

    return destination


def should_use_tools(state: TutorState) -> Literal["tools", "__end__"]:
    """Check if the last AI message contains tool calls."""
    messages = state.get("messages", [])
    if not messages:
        return "__end__"

    last = messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    return "__end__"


def route_after_tools(state: TutorState) -> Literal[
    "explain", "quiz", "clarify", "__end__"
]:
    """After tool execution, return to the content node that invoked tools."""
    phase = state.get("current_phase", "teaching")

    if phase == "quiz":
        return "quiz"
    elif phase in ("teaching", "topic_selection", "greeting"):
        return "explain"
    else:
        return "explain"


def compile_graph(checkpointer=None) -> Any:
    """
    Build and compile the LangGraph tutoring agent.

    Returns a compiled graph that can be invoked with TutorState.
    """
    builder = StateGraph(TutorState)

    # Add all nodes with retry policies on LLM-calling nodes
    retry = RetryPolicy(max_attempts=3)

    builder.add_node("router", router_node, retry=retry)
    builder.add_node("explain", explain_node, retry=retry)
    builder.add_node("quiz", quiz_node, retry=retry)
    builder.add_node("summarize", summarize_node, retry=retry)
    builder.add_node("encourage", encourage_node, retry=retry)
    builder.add_node("clarify", clarify_node, retry=retry)
    builder.add_node("tools", tool_executor_node)

    # Entry edge: always start with router
    builder.add_edge(START, "router")

    # Router → content node (conditional)
    builder.add_conditional_edges("router", route_from_router)

    # Content nodes → check for tools or end
    for node in ["explain", "quiz", "clarify"]:
        builder.add_conditional_edges(node, should_use_tools)

    # Summarize and encourage always end the turn
    builder.add_edge("summarize", END)
    builder.add_edge("encourage", END)

    # Tools → back to content node
    builder.add_conditional_edges("tools", route_after_tools)

    # Compile
    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = builder.compile(checkpointer=checkpointer)

    logger.info("graph_compiled", node_count=7)
    return graph


def create_graph_with_persistence(db_uri: str | None = None) -> Any:
    """
    Create a graph with PostgreSQL-backed persistence.

    Falls back to in-memory checkpointer if db_uri is not provided.
    """
    if db_uri:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            checkpointer = AsyncPostgresSaver.from_conn_string(db_uri)
            logger.info("graph_with_postgres_persistence")
            return compile_graph(checkpointer)
        except Exception as e:
            logger.warning(
                "postgres_persistence_failed",
                error=str(e),
                fallback="memory",
            )

    return compile_graph()
