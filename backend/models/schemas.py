"""
Pydantic models for all data types used across the application.

These models serve as the contract between parsers, RAG, agent, and API layers.
Each model includes validation and serialization support.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Document Parsing ──────────────────────────────────────────────────────────


class SlideContent(BaseModel):
    """Content extracted from a single slide."""

    slide_number: int
    title: str | None = None
    text_content: str = ""
    has_images: bool = False
    image_paths: list[str] = Field(default_factory=list)
    speaker_notes: str = ""
    tables: list[str] = Field(default_factory=list)  # Markdown-formatted tables
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """Complete parsed output from a PDF or PPTX file."""

    upload_id: str
    file_type: Literal["pdf", "pptx"]
    slides: list[SlideContent]
    total_slides: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.total_slides == 0:
            self.total_slides = len(self.slides)


# ── RAG / Retrieval ───────────────────────────────────────────────────────────


class ChunkMetadata(BaseModel):
    """Metadata attached to each vector store chunk."""

    upload_id: str
    slide_number: int
    chunk_index: int = 0
    title: str = ""
    content_type: Literal["text", "image_description", "table", "speaker_notes"] = "text"


class RetrievalResult(BaseModel):
    """A single retrieval result from the hybrid search pipeline."""

    content: str
    metadata: ChunkMetadata
    score: float = 0.0
    source: Literal["semantic", "keyword", "hybrid"] = "hybrid"


# ── Upload / API ──────────────────────────────────────────────────────────────


class UploadRequest(BaseModel):
    """Metadata submitted alongside a file upload (optional)."""

    description: str = ""


class UploadResponse(BaseModel):
    """Response after file upload."""

    id: str
    filename: str
    file_type: str
    status: str
    total_slides: int = 0
    created_at: datetime


class RetrieveRequest(BaseModel):
    """Request body for the retrieval endpoint."""

    upload_id: str
    query: str
    n_results: int = Field(default=5, ge=1, le=50)
    slide_range: list[int] | None = None  # Optional [start, end] filter


# ── Chat / Session ────────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    """A single message in a tutoring session."""

    role: Literal["user", "assistant", "system", "tool"]
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now())


class SessionState(BaseModel):
    """Current state of a tutoring session (exposed to frontend)."""

    session_id: str
    upload_id: str
    phase: str = "greeting"
    current_slide: int | None = None
    topics_covered: list[str] = Field(default_factory=list)
    quiz_score: dict[str, Any] = Field(default_factory=dict)
    message_count: int = 0


class CreateSessionRequest(BaseModel):
    """Request to create a new tutoring session."""

    upload_id: str


class SendMessageRequest(BaseModel):
    """Request to send a message in a session."""

    content: str


# ── Quiz ──────────────────────────────────────────────────────────────────────


class QuizQuestion(BaseModel):
    """A generated quiz question."""

    question: str
    options: list[str] | None = None  # For multiple choice
    correct_answer: str
    explanation: str = ""
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    question_type: Literal["multiple_choice", "short_answer", "true_false"] = "multiple_choice"
    slide_reference: int | None = None


class QuizEvaluation(BaseModel):
    """Result of evaluating a student's answer."""

    is_correct: bool
    partial_credit: float = 0.0  # 0.0 to 1.0
    feedback: str = ""
    correct_answer: str = ""
    explanation: str = ""


# ── Student Profile ───────────────────────────────────────────────────────────


class StudentProfile(BaseModel):
    """Student's learning profile and progress summary."""

    topics_covered: list[str] = Field(default_factory=list)
    quiz_scores: dict[str, float] = Field(default_factory=dict)  # topic -> avg score
    total_questions: int = 0
    correct_answers: int = 0
    confidence_level: float = 0.0
    preferred_explanation_mode: str = "standard"
    pacing_preference: str = "medium"


# ── Streaming / Events ────────────────────────────────────────────────────────


class StreamEvent(BaseModel):
    """Server-sent event for streaming responses."""

    event: Literal[
        "stream_start", "token", "tool_call", "tool_result",
        "stream_end", "error", "heartbeat", "phase_change",
    ]
    data: str | dict[str, Any] = ""


# ── Errors ────────────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    """Standardized error response."""

    error: str
    detail: str = ""
    request_id: str = ""


# ── Custom Exceptions ─────────────────────────────────────────────────────────


class SlideGuideError(Exception):
    """Base exception for all SlideGuide errors."""

    def __init__(self, message: str, detail: str = ""):
        self.message = message
        self.detail = detail
        super().__init__(message)


class SlideParsingError(SlideGuideError):
    """Error during document parsing."""


class RetrievalError(SlideGuideError):
    """Error during RAG retrieval."""


class LLMError(SlideGuideError):
    """Error from LLM API calls."""


class ToolExecutionError(SlideGuideError):
    """Error during agent tool execution."""
