"""
Vision Language Model (VLM) client for image understanding.

On the Cursor route, images are sent through cursor-sdk (Composer) and
billed to Cursor usage. On the OpenAI-compatible route, the configured
VISION_MODEL is used. Falls back gracefully when vision is unavailable.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

import openai

from backend.config import settings
from backend.llm.cursor import CursorTranslator
from backend.llm.providers import get_embedding_config
from backend.llm.runtime import current_chat_sdk, models_for
from backend.monitoring.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_IMAGE_MIME = "image/png"

DESCRIBE_IMAGE_PROMPT = (
    "Describe this lecture slide image in detail for a student studying the material. "
    "Include: what the image shows, any labels or text visible, key relationships, "
    "and what concept it illustrates. Be thorough but concise (3-5 sentences)."
)

DESCRIBE_CHART_PROMPT = (
    "Analyze this chart or graph from a lecture slide. Describe: "
    "1) The type of chart (bar, line, pie, etc.) "
    "2) What the axes/labels represent "
    "3) The key data points or trends "
    "4) What conclusion a student should draw from it. "
    "Be specific about numbers and trends visible."
)

EXTRACT_DIAGRAM_PROMPT = (
    "This image is a diagram from a lecture slide. Extract: "
    "1) All labeled components or nodes "
    "2) The relationships or connections between them (arrows, lines) "
    "3) The flow or hierarchy if applicable "
    "4) What concept this diagram represents. "
    "Format as structured bullet points."
)


class VisionClient:
    """
    Client for describing images using vision-capable models.

    Uses the active chat provider SDK so vision tokens follow the same
    subscription as tutoring chat.
    """

    def __init__(self) -> None:
        self._openai_client: openai.AsyncOpenAI | None = None

    def _openai(self) -> openai.AsyncOpenAI:
        if self._openai_client is None:
            self._openai_client = openai.AsyncOpenAI(**get_embedding_config().client_kwargs())
        return self._openai_client

    async def describe_image(
        self,
        image_path: str,
        context: str = "",
    ) -> str:
        """
        Generate a text description of an image from a slide.

        Args:
            image_path: Path to the image file.
            context: Optional surrounding text context from the slide.

        Returns:
            A text description of the image content.
        """
        encoded = self._encode_image(image_path)
        if not encoded:
            return ""
        image_data, mime_type = encoded

        prompt = DESCRIBE_IMAGE_PROMPT
        if context:
            prompt += f"\n\nContext from the slide: {context}"

        return await self._call_vision(image_data, prompt, mime_type=mime_type)

    async def describe_chart(
        self,
        image_path: str,
        context: str = "",
    ) -> str:
        """Describe a chart or graph image in detail."""
        encoded = self._encode_image(image_path)
        if not encoded:
            return ""
        image_data, mime_type = encoded

        prompt = DESCRIBE_CHART_PROMPT
        if context:
            prompt += f"\n\nSlide context: {context}"

        return await self._call_vision(image_data, prompt, mime_type=mime_type)

    async def extract_diagram_relationships(
        self,
        image_path: str,
        context: str = "",
    ) -> str:
        """Extract components and relationships from a diagram."""
        encoded = self._encode_image(image_path)
        if not encoded:
            return ""
        image_data, mime_type = encoded

        prompt = EXTRACT_DIAGRAM_PROMPT
        if context:
            prompt += f"\n\nSlide context: {context}"

        return await self._call_vision(image_data, prompt, mime_type=mime_type)

    async def _call_vision(
        self,
        image_base64: str,
        prompt: str,
        mime_type: str = _DEFAULT_IMAGE_MIME,
    ) -> str:
        """Send an image to the active vision-capable provider."""
        provider = current_chat_sdk()
        if provider == "cursor":
            return await self._call_cursor_vision(image_base64, prompt, mime_type)
        return await self._call_openai_vision(image_base64, prompt, mime_type)

    async def _call_cursor_vision(
        self, image_base64: str, prompt: str, mime_type: str = _DEFAULT_IMAGE_MIME
    ) -> str:
        vision_model = models_for("cursor").vision
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}",
                        },
                    },
                ],
            }
        ]
        try:
            reply = await CursorTranslator().chat_async(messages, vision_model)
            logger.info(
                "vlm_description_generated",
                model=reply.model or vision_model,
                provider="cursor",
                description_length=len(reply.text),
            )
            return reply.text
        except Exception as e:
            logger.error("vlm_call_failed", error=str(e), provider="cursor")
            return ""

    async def _call_openai_vision(
        self, image_base64: str, prompt: str, mime_type: str = _DEFAULT_IMAGE_MIME
    ) -> str:
        if not settings.vision_model:
            return (
                "[Vision unavailable] Image analysis requires a vision-capable model. "
                "Set VISION_MODEL or bill this session to the Cursor SDK."
            )

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}",
                        },
                    },
                ],
            },
        ]

        try:
            response = await self._openai().chat.completions.create(
                model=settings.vision_model,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
            )

            content = response.choices[0].message.content or ""

            logger.info(
                "vlm_description_generated",
                model=settings.vision_model,
                provider="openai",
                description_length=len(content),
            )

            return content

        except Exception as e:
            logger.error("vlm_call_failed", error=str(e), provider="openai")
            return ""

    @staticmethod
    def _encode_image(image_path: str) -> tuple[str, str] | None:
        """Read and base64-encode an image file; return ``(data, mime_type)``."""
        path = Path(image_path)
        if not path.exists():
            logger.warning("image_not_found", path=image_path)
            return None

        try:
            image_bytes = path.read_bytes()
            mime_type, _ = mimetypes.guess_type(str(path))
            return (
                base64.b64encode(image_bytes).decode("utf-8"),
                mime_type or _DEFAULT_IMAGE_MIME,
            )
        except Exception as e:
            logger.error("image_encode_failed", path=image_path, error=str(e))
            return None
