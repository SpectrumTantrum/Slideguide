"""Tests for document parsers."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path

from backend.models.schemas import SlideContent, ParsedDocument


class TestPdfParser:
    """Tests for the PDF parser."""

    def test_parsed_document_total_slides(self):
        """ParsedDocument auto-computes total_slides from slides list."""
        doc = ParsedDocument(
            upload_id="test-upload",
            file_type="pdf",
            slides=[
                SlideContent(slide_number=1, text_content="Hello"),
                SlideContent(slide_number=2, text_content="World"),
            ],
        )
        assert doc.total_slides == 2

    def test_parsed_document_empty(self):
        """ParsedDocument handles empty slides list."""
        doc = ParsedDocument(
            upload_id="test-upload",
            file_type="pdf",
            slides=[],
        )
        assert doc.total_slides == 0

    def test_slide_content_defaults(self):
        """SlideContent has sensible defaults."""
        slide = SlideContent(slide_number=1)
        assert slide.title is None
        assert slide.text_content == ""
        assert slide.has_images is False
        assert slide.image_paths == []
        assert slide.speaker_notes == ""
        assert slide.tables == []


class TestOcrPipeline:
    """Tests for the OCR pipeline."""

    def test_ocr_not_available_returns_empty(self):
        """When Tesseract isn't installed, extraction returns empty."""
        from backend.parsers.ocr import OcrPipeline

        pipeline = OcrPipeline()
        pipeline._available = False

        text, confidence = pipeline.extract_text("nonexistent.png")
        assert text == ""
        assert confidence == 0.0

    def test_ocr_file_not_found(self):
        """Non-existent file returns empty result."""
        from backend.parsers.ocr import OcrPipeline

        pipeline = OcrPipeline()
        pipeline._available = True

        text, confidence = pipeline.extract_text("/nonexistent/path/image.png")
        assert text == ""
        assert confidence == 0.0


class TestParserDispatcher:
    """Tests for the parse_document dispatcher."""

    @pytest.mark.asyncio
    async def test_unsupported_file_type_raises(self):
        """Unsupported file types raise SlideParsingError."""
        from backend.parsers import parse_document
        from backend.models.schemas import SlideParsingError

        with pytest.raises(SlideParsingError):
            await parse_document("test.docx", "upload-1")
