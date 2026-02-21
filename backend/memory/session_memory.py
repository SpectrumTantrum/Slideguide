"""
Short-term session memory management.

Handles conversation context window construction and automatic
summarization when the message list exceeds a configurable threshold.
Uses LangGraph's checkpointer for persistence — this module sits
on top, managing the *content* of what gets stored.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.config import settings
from backend.llm.client import LLMClient
from backend.monitoring.logger import get_logger

logger = get_logger(__name__)

# When message count exceeds this, summarize older messages
SUMMARIZE_THRESHOLD = 20

# Keep this many recent messages verbatim (not summarized)
RECENT_MESSAGES_TO_KEEP = 8

SUMMARY_SYSTEM_PROMPT = (
    "You are a conversation summarizer for a tutoring session. "
    "Summarize the key points from this conversation so far, including: "
    "topics discussed, questions asked, quiz results, and any areas "
    "the student struggled with. Be concise — 3-5 bullet points."
)


class SessionMemory:
    """
    Manages the conversation context window for a tutoring session.

    Responsibilities:
    - Build the context window from message history
    - Summarize older messages when the list grows too long
    - Inject retrieval context into the conversation
    """

    def __init__(self) -> None:
        self._llm = LLMClient()
        self._summaries: dict[str, str] = {}  # session_id -> summary

    async def build_context_window(
        self,
        messages: list[Any],
        session_id: str,
        system_prompt: str,
        retrieval_context: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, str]]:
        """
        Build an optimized context window for an LLM call.

        Combines: system prompt + summary (if any) + retrieval context + recent messages.
        """
        context: list[dict[str, str]] = []

        # System prompt is always first
        system_content = system_prompt

        # Prepend conversation summary if we have one
        summary = self._summaries.get(session_id)
        if summary:
            system_content += (
                f"\n\n--- Previous conversation summary ---\n{summary}"
            )

        # Append retrieval context if available
        if retrieval_context:
            formatted = self._format_retrieval_context(retrieval_context)
            system_content += f"\n\n--- Relevant slide content ---\n{formatted}"

        context.append({"role": "system", "content": system_content})

        # Add recent messages (keep the last N for full detail)
        recent = messages[-RECENT_MESSAGES_TO_KEEP:]
        for msg in recent:
            if isinstance(msg, HumanMessage):
                context.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                context.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                context.append({"role": "system", "content": msg.content})

        return context

    async def maybe_summarize(
        self,
        messages: list[Any],
        session_id: str,
    ) -> str | None:
        """
        Summarize older messages if the conversation is long enough.

        Returns the summary text if summarization was performed, None otherwise.
        """
        if len(messages) < SUMMARIZE_THRESHOLD:
            return None

        # Summarize everything except the most recent messages
        messages_to_summarize = messages[:-RECENT_MESSAGES_TO_KEEP]
        if not messages_to_summarize:
            return None

        # Build conversation text for summarization
        conversation_text = self._messages_to_text(messages_to_summarize)

        try:
            response = await self._llm.chat(
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Summarize this tutoring conversation:\n\n{conversation_text}"},
                ],
                model=settings.routing_model,  # Haiku — cheap and fast
                temperature=0.3,
                max_tokens=500,
            )

            summary = response["choices"][0]["message"].get("content", "")
            self._summaries[session_id] = summary

            logger.info(
                "conversation_summarized",
                session_id=session_id,
                messages_summarized=len(messages_to_summarize),
                summary_length=len(summary),
            )

            return summary

        except Exception as e:
            logger.error(
                "summarization_failed",
                session_id=session_id,
                error=str(e),
            )
            return None

    def get_summary(self, session_id: str) -> str | None:
        """Get the current conversation summary for a session."""
        return self._summaries.get(session_id)

    def clear_session(self, session_id: str) -> None:
        """Remove stored summary for a session."""
        self._summaries.pop(session_id, None)

    @staticmethod
    def _messages_to_text(messages: list[Any]) -> str:
        """Convert message objects to readable text for summarization."""
        lines: list[str] = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                lines.append(f"Student: {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"Tutor: {msg.content}")
        return "\n".join(lines)

    @staticmethod
    def _format_retrieval_context(context: list[dict[str, Any]]) -> str:
        """Format retrieval results as readable context for the LLM."""
        parts: list[str] = []
        for item in context[:5]:  # Cap at 5 results
            slide = item.get("slide_number", "?")
            title = item.get("title", "")
            content = item.get("content", "")
            header = f"[Slide {slide}]"
            if title:
                header += f" {title}"
            parts.append(f"{header}\n{content}")
        return "\n\n".join(parts)
