"""
Vision Language Model (VLM) client for image understanding.

Uses a vision-capable provider to send images (base64-encoded) for
description, chart analysis, and diagram relationship extraction.
Defaults to OpenRouter (cloud VLMs) even when the main LLM provider
is set to LM Studio, because most local models lack vision support.
Falls back gracefully when no vision provider is available.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import openai

from backend.config import settings
from backend.llm.providers import get_vision_provider_config
from backend.monitoring.logger import get_logger

logger = get_logger(__name__)

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
    Client for describing images using vision-capable LLMs.

    Uses a dedicated client pointed at the vision provider (defaults to
    OpenRouter). Falls back to a text message when no vision provider
    is configured.
    """

    def __init__(self) -> None:
        self._provider_config = get_vision_provider_config()
        self._available = bool(self._provider_config.api_key and self._provider_config.api_key != "lm-studio")

        # For lmstudio vision provider, we trust the user knows they have a vision model
        if settings.vision_provider == "lmstudio":
            self._available = True

        if self._available:
            self._client = openai.AsyncOpenAI(**self._provider_config.client_kwargs())
        else:
            self._client = None
            logger.warning(
                "vision_unavailable",
                reason="No API key for vision provider. Set OPENROUTER_API_KEY or VISION_PROVIDER=lmstudio.",
            )

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
        image_data = self._encode_image(image_path)
        if not image_data:
            return ""

        prompt = DESCRIBE_IMAGE_PROMPT
        if context:
            prompt += f"\n\nContext from the slide: {context}"

        return await self._call_vision(image_data, prompt)

    async def describe_chart(
        self,
        image_path: str,
        context: str = "",
    ) -> str:
        """Describe a chart or graph image in detail."""
        image_data = self._encode_image(image_path)
        if not image_data:
            return ""

        prompt = DESCRIBE_CHART_PROMPT
        if context:
            prompt += f"\n\nSlide context: {context}"

        return await self._call_vision(image_data, prompt)

    async def extract_diagram_relationships(
        self,
        image_path: str,
        context: str = "",
    ) -> str:
        """Extract components and relationships from a diagram."""
        image_data = self._encode_image(image_path)
        if not image_data:
            return ""

        prompt = EXTRACT_DIAGRAM_PROMPT
        if context:
            prompt += f"\n\nSlide context: {context}"

        return await self._call_vision(image_data, prompt)

    async def _call_vision(
        self,
        image_base64: str,
        prompt: str,
    ) -> str:
        """Send an image to the vision model and return the description."""
        if not self._available or self._client is None:
            return (
                "[Vision unavailable] Image analysis requires a vision-capable model. "
                "Configure OPENROUTER_API_KEY or set VISION_PROVIDER=lmstudio with a vision model loaded."
            )

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                        },
                    },
                ],
            },
        ]

        try:
            response = await self._client.chat.completions.create(
                model=settings.active_vision_model,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
            )

            content = response.choices[0].message.content or ""

            logger.info(
                "vlm_description_generated",
                model=settings.active_vision_model,
                provider=self._provider_config.name,
                description_length=len(content),
            )

            return content

        except Exception as e:
            logger.error("vlm_call_failed", error=str(e), provider=self._provider_config.name)
            return ""

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Read and base64-encode an image file."""
        path = Path(image_path)
        if not path.exists():
            logger.warning("image_not_found", path=image_path)
            return ""

        try:
            image_bytes = path.read_bytes()
            return base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e:
            logger.error("image_encode_failed", path=image_path, error=str(e))
            return ""
