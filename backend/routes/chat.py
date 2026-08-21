"""
Chat API endpoints for tutoring sessions.

POST /api/session           — Create a new tutoring session
POST /api/session/{id}/message — Send a message (SSE streaming response)
GET  /api/session/{id}      — Get session state
GET  /api/session/{id}/history — Get message history
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage
from sse_starlette.sse import EventSourceResponse

from backend.agent.graph import compile_graph
from backend.agent.state import create_initial_state
from backend.db.repositories.messages import MessageRepository
from backend.db.repositories.progress import ProgressRepository
from backend.db.repositories.sessions import SessionRepository
from backend.db.repositories.uploads import UploadRepository
from backend.llm.cursor import cancel_active_runs
from backend.llm.runtime import (
    bind_chat_sdk,
    default_provider,
    reset_chat_sdk,
    resolve_provider,
    session_chat_sdk,
)
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
    supabase = request.app.state.supabase
    upload_repo = UploadRepository(supabase)
    session_repo = SessionRepository(supabase)
    msg_repo = MessageRepository(supabase)

    # Verify upload exists and is ready
    upload = upload_repo.get_by_id(body.upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")
    if upload["status"] != "READY":
        raise HTTPException(
            status_code=400,
            detail=f"Upload not ready: {upload['status']}",
        )

    try:
        chat_sdk = resolve_provider(body.provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    session = session_repo.create(
        upload_id=body.upload_id,
        phase="GREETING",
        metadata={"chat_sdk": chat_sdk},
    )

    # Create initial progress record
    progress_tracker = StudentProgressTracker(supabase)
    progress_tracker.get_or_create(session["id"], body.upload_id)

    # Initialize graph state (LangGraph persists via checkpointer)
    graph = get_graph()
    initial_state = create_initial_state(
        session["id"], body.upload_id, chat_sdk=chat_sdk
    )

    # Run initial greeting turn
    config = {"configurable": {"thread_id": session["id"]}}
    token = bind_chat_sdk(chat_sdk, session["id"])
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
            msg_repo.create(
                session_id=session["id"],
                role="ASSISTANT",
                content=greeting,
            )

    except Exception as e:
        logger.error(
            "greeting_generation_failed",
            session_id=session["id"],
            error=str(e),
        )
        greeting = (
            "Welcome to SlideGuide! I'm ready to help you study. "
            "What topic would you like to start with?"
        )
    finally:
        reset_chat_sdk(token)

    logger.info(
        "session_created",
        session_id=session["id"],
        upload_id=body.upload_id,
        chat_sdk=chat_sdk,
    )

    return SessionState(
        session_id=session["id"],
        upload_id=body.upload_id,
        phase="greeting",
        message_count=1 if greeting else 0,
        chat_sdk=chat_sdk,
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
    supabase = request.app.state.supabase
    session_repo = SessionRepository(supabase)
    msg_repo = MessageRepository(supabase)

    # Verify session exists
    session = session_repo.get_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    stored = session_chat_sdk(session)
    try:
        sdk = resolve_provider(body.provider) if body.provider else resolve_provider(stored)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if body.provider and stored != sdk:
        meta = dict(session.get("metadata") or {})
        meta["chat_sdk"] = sdk
        session_repo.update(session_id, metadata=meta)

    # Persist student message
    msg_repo.create(
        session_id=session_id,
        role="USER",
        content=body.content,
    )

    logger.info(
        "message_received",
        session_id=session_id,
        content_length=len(body.content),
        chat_sdk=sdk,
    )

    async def event_generator():
        """Generate SSE events from the agent's response."""
        graph = get_graph()
        config = {"configurable": {"thread_id": session_id}}
        token = bind_chat_sdk(sdk, session_id)
        invoke_task: asyncio.Task[Any] | None = None

        try:
            # Send stream start
            yield _sse_event("stream_start", {"session_id": session_id})

            # Invoke the graph with the new message
            input_state = {
                "messages": [HumanMessage(content=body.content)],
                "chat_sdk": sdk,
            }

            invoke_task = asyncio.create_task(graph.ainvoke(input_state, config))
            while not invoke_task.done():
                if await request.is_disconnected():
                    invoke_task.cancel()
                    cancel_active_runs()
                    logger.info("sse_client_disconnected", session_id=session_id)
                    return
                await asyncio.wait({invoke_task}, timeout=0.2)
            result = await invoke_task

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
            msg_repo.create(
                session_id=session_id,
                role="ASSISTANT",
                content=ai_content,
                tool_calls=tool_calls_data if tool_calls_data else None,
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
            session_repo.update(session_id, phase=db_phase)

            # Run summarization if needed
            full_messages = result.get("messages", [])
            await _session_memory.maybe_summarize(full_messages, session_id)

            # Update progress tracking
            progress_tracker = StudentProgressTracker(supabase)
            topics = result.get("topics_covered", [])
            for topic in topics:
                progress_tracker.update_topic_covered(session_id, topic)

        except asyncio.CancelledError:
            cancel_active_runs()
            raise
        except Exception as e:
            logger.error(
                "message_processing_failed",
                session_id=session_id,
                error=str(e),
            )
            yield _sse_event("error", {"message": f"Processing error: {str(e)}"})
        finally:
            if invoke_task is not None and not invoke_task.done():
                invoke_task.cancel()
                cancel_active_runs()
            reset_chat_sdk(token)

    return EventSourceResponse(event_generator())


# ── Get Session State ────────────────────────────────────────────────────────


@router.get("/{session_id}", response_model=SessionState)
async def get_session(request: Request, session_id: str):
    """Get the current state of a tutoring session."""
    supabase = request.app.state.supabase
    session_repo = SessionRepository(supabase)
    msg_repo = MessageRepository(supabase)
    progress_repo = ProgressRepository(supabase)

    session = session_repo.get_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    message_count = msg_repo.count(session_id)

    # Get progress data (separate query — no Prisma includes)
    progress = progress_repo.get_by_session_id(session_id)

    topics_covered: list[str] = []
    quiz_score: dict[str, Any] = {}
    if progress:
        # JSONB columns return native Python objects — no json.loads needed
        topics_covered = progress["topics_covered"] or []
        quiz_score = {
            "correct": progress["correct_answers"],
            "total": progress["total_questions"],
            "confidence": progress["confidence_level"],
        }

    phase_str = session["phase"]
    if isinstance(phase_str, str):
        phase_str = phase_str.lower()

    return SessionState(
        session_id=session["id"],
        upload_id=session["upload_id"],
        phase=phase_str,
        current_slide=None,
        topics_covered=topics_covered,
        quiz_score=quiz_score,
        message_count=message_count,
        chat_sdk=session_chat_sdk(session) or default_provider(),
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
    supabase = request.app.state.supabase
    session_repo = SessionRepository(supabase)
    msg_repo = MessageRepository(supabase)

    session = session_repo.get_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = msg_repo.get_by_session(session_id, limit=limit, offset=offset)
    total = msg_repo.count(session_id)

    return {
        "session_id": session_id,
        "messages": [
            {
                "id": msg["id"],
                "role": msg["role"].lower() if isinstance(msg["role"], str) else str(msg["role"]).lower(),
                "content": msg["content"],
                "tool_calls": msg["tool_calls"],  # JSONB — already native
                "created_at": msg["created_at"],
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
