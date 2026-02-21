"""
LangGraph agent state schema for the tutoring system.

Defines TutorState as a TypedDict with all fields needed to track
the tutoring conversation, student progress, and agent decisions.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict


class TutorState(TypedDict):
    """
    State schema for the LangGraph tutoring agent.

    Fields are updated by nodes and persisted between turns
    via the LangGraph checkpointer.
    """

    # Conversation messages (append-only via operator.add)
    messages: Annotated[list[AnyMessage], operator.add]

    # Session identification
    session_id: str
    upload_id: str

    # State machine phase
    current_phase: Literal[
        "greeting", "topic_selection", "teaching",
        "quiz", "review", "wrap_up",
    ]

    # Current context
    current_slide: int | None
    current_topic: str | None
    retrieval_context: list[dict[str, Any]]

    # Student tracking
    topics_covered: list[str]
    quiz_score: dict[str, Any]  # {"correct": int, "total": int, "by_topic": {...}}
    student_profile: dict[str, Any]  # preferences, confidence, etc.

    # Teaching strategy
    explanation_mode: Literal[
        "standard", "analogy", "visual", "step_by_step", "eli5",
    ]
    pacing_preference: Literal["slow", "medium", "fast"]

    # Agent control
    encouragement_due: bool
    error_count: int
    pending_tasks: list[str]  # For compound request decomposition


def create_initial_state(session_id: str, upload_id: str) -> dict[str, Any]:
    """Create the initial state for a new tutoring session."""
    return {
        "messages": [],
        "session_id": session_id,
        "upload_id": upload_id,
        "current_phase": "greeting",
        "current_slide": None,
        "current_topic": None,
        "retrieval_context": [],
        "topics_covered": [],
        "quiz_score": {"correct": 0, "total": 0, "by_topic": {}},
        "student_profile": {
            "confidence_level": 0.5,
            "preferred_mode": "standard",
            "consecutive_correct": 0,
            "consecutive_incorrect": 0,
        },
        "explanation_mode": "standard",
        "pacing_preference": "medium",
        "encouragement_due": False,
        "error_count": 0,
        "pending_tasks": [],
    }
