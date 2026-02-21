"""
Chat API endpoints for tutoring sessions.

POST /api/session           — Create a new tutoring session
POST /api/session/{id}/message — Send a message (SSE streaming response)
GET  /api/session/{id}      — Get session state
GET  /api/session/{id}/history — Get message history
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage
from sse_starlette.sse import EventSourceResponse

from backend.agent.graph import compile_graph
from backend.agent.state import create_initial_state
from backend.memory.session_memory import SessionMemory
from backend.memory.student_progress import StudentProgressTracker
from backend.models.schemas import (
    CreateSessionRequest,
    SendMessageRequest,
    SessionState,
)
from backend.monitoring.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/session", tags=["chat"])

# Shared instances (initialized lazily)
_graph = None
_session_memory = SessionMemory()


def get_graph():
    """Lazy-initialize the LangGraph agent."""
    global _graph
    if _graph is None:
        _graph = compile_graph()
    return _graph


# ── Create Session ───────────────────────────────────────────────────────────


@router.post("", response_model=SessionState)
async def create_session(request: Request, body: CreateSessionRequest):
    """
    Create a new tutoring session for an uploaded document.

    Initializes agent state, creates DB records, and returns
    the session ID for subsequent message calls.
    """
    db = request.app.state.db

    # Verify upload exists and is ready
    upload = await db.upload.find_unique(where={"id": body.upload_id})
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")
    if upload.status != "READY":
        raise HTTPException(
            status_code=400,
            detail=f"Upload not ready: {upload.status}",
        )

    # Create session in database
    session = await db.session.create(
        data={
            "uploadId": body.upload_id,
            "phase": "GREETING",
        }
    )

    # Create initial progress record
    progress_tracker = StudentProgressTracker(db)
    await progress_tracker.get_or_create(session.id, body.upload_id)

    # Initialize graph state (LangGraph persists via checkpointer)
    graph = get_graph()
    initial_state = create_initial_state(session.id, body.upload_id)

    # Run initial greeting turn
    config = {"configurable": {"thread_id": session.id}}
    try:
        result = await graph.ainvoke(initial_state, config)

        # Extract greeting message
        greeting = ""
        if result.get("messages"):
            last_msg = result["messages"][-1]
            if isinstance(last_msg, AIMessage):
                greeting = last_msg.content

        # Persist greeting to database
        if greeting:
            await db.message.create(
                data={
                    "sessionId": session.id,
                    "role": "ASSISTANT",
                    "content": greeting,
                }
            )

    except Exception as e:
        logger.error(
            "greeting_generation_failed",
            session_id=session.id,
            error=str(e),
        )
        greeting = (
            "Welcome to SlideGuide! I'm ready to help you study. "
            "What topic would you like to start with?"
        )

    logger.info(
        "session_created",
        session_id=session.id,
        upload_id=body.upload_id,
    )

    return SessionState(
        session_id=session.id,
        upload_id=body.upload_id,
        phase="greeting",
        message_count=1 if greeting else 0,
    )


# ── Send Message (SSE Streaming) ─────────────────────────────────────────────


@router.post("/{session_id}/message")
async def send_message(
    request: Request,
    session_id: str,
    body: SendMessageRequest,
):
    """
    Send a student message and stream the agent's response via SSE.

    The agent graph processes the message through routing, retrieval,
    and generation nodes. The final AI response is streamed token-by-token.
    """
    db = request.app.state.db

    # Verify session exists
    session = await db.session.find_unique(where={"id": session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Persist student message
    await db.message.create(
        data={
            "sessionId": session_id,
            "role": "USER",
            "content": body.content,
        }
    )

    logger.info(
        "message_received",
        session_id=session_id,
        content_length=len(body.content),
    )

    async def event_generator():
        """Generate SSE events from the agent's response."""
        graph = get_graph()
        config = {"configurable": {"thread_id": session_id}}

        try:
            # Send stream start
            yield _sse_event("stream_start", {"session_id": session_id})

            # Invoke the graph with the new message
            input_state = {
                "messages": [HumanMessage(content=body.content)],
            }

            result = await graph.ainvoke(input_state, config)

            # Extract the AI response from the result
            ai_content = ""
            tool_calls_data = []
            phase = result.get("current_phase", "teaching")

            if result.get("messages"):
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage) and msg.content:
                        ai_content = msg.content
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            tool_calls_data = msg.tool_calls
                        break

            if not ai_content:
                ai_content = "I'm thinking about that. Could you rephrase?"

            # Stream the response token by token (simulated chunking)
            # Real token-by-token streaming requires graph streaming mode
            chunks = _chunk_response(ai_content)
            for chunk in chunks:
                yield _sse_event("token", {"text": chunk})

            # Report tool calls if any
            if tool_calls_data:
                yield _sse_event("tool_calls", {
                    "tools": [
                        {"name": tc.get("name", ""), "args": tc.get("args", {})}
                        for tc in tool_calls_data
                    ],
                })

            # Report phase change if different
            yield _sse_event("phase_change", {"phase": phase})

            # Stream end
            yield _sse_event("stream_end", {"finish_reason": "stop"})

            # Persist AI response to database
            await db.message.create(
                data={
                    "sessionId": session_id,
                    "role": "ASSISTANT",
                    "content": ai_content,
                    "toolCalls": json.dumps(tool_calls_data) if tool_calls_data else None,
                }
            )

            # Update session phase
            phase_map = {
                "greeting": "GREETING",
                "topic_selection": "TOPIC_SELECTION",
                "teaching": "TEACHING",
                "quiz": "QUIZ",
                "review": "REVIEW",
                "wrap_up": "WRAP_UP",
            }
            db_phase = phase_map.get(phase, "TEACHING")
            await db.session.update(
                where={"id": session_id},
                data={"phase": db_phase},
            )

            # Run summarization if needed
            full_messages = result.get("messages", [])
            await _session_memory.maybe_summarize(full_messages, session_id)

            # Update progress tracking
            progress_tracker = StudentProgressTracker(db)
            topics = result.get("topics_covered", [])
            for topic in topics:
                await progress_tracker.update_topic_covered(session_id, topic)

        except Exception as e:
            logger.error(
                "message_processing_failed",
                session_id=session_id,
                error=str(e),
            )
            yield _sse_event("error", {"message": f"Processing error: {str(e)}"})

    return EventSourceResponse(event_generator())


# ── Get Session State ────────────────────────────────────────────────────────


@router.get("/{session_id}", response_model=SessionState)
async def get_session(request: Request, session_id: str):
    """Get the current state of a tutoring session."""
    db = request.app.state.db

    session = await db.session.find_unique(
        where={"id": session_id},
        include={"progress": True},
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    message_count = await db.message.count(where={"sessionId": session_id})

    # Get progress data
    topics_covered: list[str] = []
    quiz_score: dict[str, Any] = {}
    if session.progress:
        topics_covered = (
            json.loads(session.progress.topicsCovered)
            if session.progress.topicsCovered
            else []
        )
        quiz_score = {
            "correct": session.progress.correctAnswers,
            "total": session.progress.totalQuestions,
            "confidence": session.progress.confidenceLevel,
        }

    return SessionState(
        session_id=session.id,
        upload_id=session.uploadId,
        phase=session.phase.lower() if hasattr(session.phase, "lower") else str(session.phase).lower(),
        current_slide=None,
        topics_covered=topics_covered,
        quiz_score=quiz_score,
        message_count=message_count,
    )


# ── Get Message History ──────────────────────────────────────────────────────


@router.get("/{session_id}/history")
async def get_history(
    request: Request,
    session_id: str,
    limit: int = 50,
    offset: int = 0,
):
    """Get message history for a session with pagination."""
    db = request.app.state.db

    session = await db.session.find_unique(where={"id": session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = await db.message.find_many(
        where={"sessionId": session_id},
        order={"createdAt": "asc"},
        take=limit,
        skip=offset,
    )

    total = await db.message.count(where={"sessionId": session_id})

    return {
        "session_id": session_id,
        "messages": [
            {
                "id": msg.id,
                "role": msg.role.lower() if hasattr(msg.role, "lower") else str(msg.role).lower(),
                "content": msg.content,
                "tool_calls": json.loads(msg.toolCalls) if msg.toolCalls else None,
                "created_at": msg.createdAt.isoformat(),
            }
            for msg in messages
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


# ── Helpers ──────────────────────────────────────────────────────────────────


def _sse_event(event: str, data: dict[str, Any]) -> dict[str, str]:
    """Format data as an SSE event dict for sse-starlette."""
    return {
        "event": event,
        "data": json.dumps(data),
    }


def _chunk_response(text: str, chunk_size: int = 12) -> list[str]:
    """
    Split response text into small chunks for streaming effect.

    Splits on word boundaries to avoid breaking mid-word.
    """
    words = text.split(" ")
    chunks: list[str] = []
    current = ""

    for word in words:
        if len(current) + len(word) + 1 > chunk_size and current:
            chunks.append(current + " ")
            current = word
        else:
            current = f"{current} {word}" if current else word

    if current:
        chunks.append(current)

    return chunks
