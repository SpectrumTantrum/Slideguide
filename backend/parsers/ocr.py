"""
Tesseract OCR pipeline for extracting text from slide images.

Handles preprocessing (grayscale, contrast, threshold) and provides
confidence scores. Falls back to VLM description when OCR confidence
is below threshold. Falls back gracefully if Tesseract is not installed.
"""

from __future__ import annotations

from pathlib import Path

from backend.config import settings
from backend.monitoring.logger import get_logger

logger = get_logger(__name__)


# Below this confidence threshold, fall back to VLM for description
VLM_FALLBACK_THRESHOLD = 0.6


class OcrPipeline:
    """Extract text from images using Tesseract OCR with preprocessing."""

    def __init__(self) -> None:
        self._available: bool | None = None
        self._vision_client = None

    @property
    def is_available(self) -> bool:
        """Check if Tesseract is installed and accessible."""
        if self._available is None:
            try:
                import pytesseract

                pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
                pytesseract.get_tesseract_version()
                self._available = True
                logger.info("ocr_available", tesseract_cmd=settings.tesseract_cmd)
            except Exception:
                self._available = False
                logger.warning("ocr_unavailable", tesseract_cmd=settings.tesseract_cmd)
        return self._available

    def extract_text(self, image_path: str) -> tuple[str, float]:
        """
        Extract text from a single image.

        Returns (extracted_text, confidence_score).
        Confidence is 0.0-1.0 where higher is better.
        """
        if not self.is_available:
            logger.warning("ocr_skipped", reason="tesseract_not_available")
            return "", 0.0

        path = Path(image_path)
        if not path.exists():
            logger.warning("ocr_file_not_found", path=image_path)
            return "", 0.0

        try:
            import pytesseract
            from PIL import Image, ImageEnhance, ImageFilter

            img = Image.open(path)

            # Preprocessing pipeline
            img = img.convert("L")  # Grayscale
            img = ImageEnhance.Contrast(img).enhance(2.0)  # Boost contrast
            img = img.filter(ImageFilter.SHARPEN)  # Sharpen
            img = img.point(lambda x: 0 if x < 128 else 255)  # Binary threshold

            # Extract with confidence data
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

            # Calculate average confidence (ignoring -1 which means no text detected)
            confidences = [c for c in data["conf"] if c > 0]
            avg_confidence = sum(confidences) / len(confidences) / 100 if confidences else 0.0

            text = pytesseract.image_to_string(img).strip()

            logger.debug(
                "ocr_extracted",
                path=image_path,
                text_length=len(text),
                confidence=round(avg_confidence, 2),
            )

            return text, avg_confidence

        except Exception as e:
            logger.error("ocr_extraction_failed", path=image_path, error=str(e))
            return "", 0.0

    async def extract_with_vlm_fallback(
        self,
        image_path: str,
        slide_context: str = "",
    ) -> tuple[str, float, str]:
        """
        Extract text with automatic VLM fallback for low-confidence results.

        Returns (text, confidence, source) where source is "ocr" or "vlm".
        """
        text, confidence = self.extract_text(image_path)

        # If OCR confidence is acceptable, use OCR result
        if confidence >= VLM_FALLBACK_THRESHOLD and text.strip():
            return text, confidence, "ocr"

        # Fall back to VLM description
        vlm_text = await self._get_vlm_description(image_path, slide_context)
        if vlm_text:
            logger.info(
                "vlm_fallback_used",
                path=image_path,
                ocr_confidence=round(confidence, 2),
                vlm_length=len(vlm_text),
            )
            return vlm_text, 0.9, "vlm"  # VLM descriptions get high confidence

        # Neither worked — return whatever OCR got
        return text, confidence, "ocr"

    async def _get_vlm_description(
        self,
        image_path: str,
        context: str = "",
    ) -> str:
        """Get a VLM description of an image, lazy-loading the client."""
        try:
            if self._vision_client is None:
                from backend.llm.vision import VisionClient
                self._vision_client = VisionClient()

            return await self._vision_client.describe_image(image_path, context)
        except Exception as e:
            logger.error("vlm_fallback_failed", path=image_path, error=str(e))
            return ""

    def extract_from_slide_images(
        self, image_paths: list[str]
    ) -> list[tuple[str, float]]:
        """
        Extract text from multiple slide images (sync, OCR only).

        Returns list of (text, confidence) tuples in the same order.
        """
        results = []
        for path in image_paths:
            text, confidence = self.extract_text(path)
            results.append((text, confidence))
        return results
