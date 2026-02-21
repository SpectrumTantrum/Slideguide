"""
Vision Language Model (VLM) client for image understanding.

Uses OpenRouter to send images (base64-encoded) to vision-capable
models for description, chart analysis, and diagram relationship
extraction. Results are stored as text chunks in the vector store.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.llm.client import LLMClient
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

    Sends base64-encoded images via OpenRouter to models that
    support the vision/image content type.
    """

    def __init__(self) -> None:
        self._llm = LLMClient()

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
            response = await self._llm.chat(
                messages=messages,
                model=settings.vision_model,
                temperature=0.3,
                max_tokens=1024,
            )

            content = response["choices"][0]["message"].get("content", "")

            logger.info(
                "vlm_description_generated",
                model=settings.vision_model,
                description_length=len(content),
            )

            return content

        except Exception as e:
            logger.error("vlm_call_failed", error=str(e))
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
