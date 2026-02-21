"""
PDF parser using PyMuPDF (fitz) for structured text and image extraction.

Extracts per-page text preserving headings, extracts embedded images,
and applies title heuristics based on font size. For image-heavy pages
with little text, falls back to OCR.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import fitz  # PyMuPDF

from backend.models.schemas import ParsedDocument, SlideContent
from backend.monitoring.logger import get_logger
from backend.parsers.ocr import OcrPipeline

logger = get_logger(__name__)


class PdfParser:
    """Parse PDF files into structured slide content."""

    def __init__(self) -> None:
        self.ocr = OcrPipeline()

    async def parse(self, file_path: str, upload_id: str) -> ParsedDocument:
        """
        Parse a PDF file into a ParsedDocument.

        Each page becomes a slide. For scanned/image-heavy pages,
        OCR is applied automatically when text extraction yields little content.
        """
        logger.info("pdf_parse_start", file_path=file_path, upload_id=upload_id)

        doc = fitz.open(file_path)
        slides: list[SlideContent] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            slide = await self._extract_page(page, page_num + 1, upload_id)
            slides.append(slide)
            logger.debug(
                "pdf_page_parsed",
                slide_number=page_num + 1,
                text_length=len(slide.text_content),
                has_images=slide.has_images,
            )

        doc.close()

        result = ParsedDocument(
            upload_id=upload_id,
            file_type="pdf",
            slides=slides,
            metadata={"source_file": Path(file_path).name},
        )

        logger.info(
            "pdf_parse_complete",
            upload_id=upload_id,
            total_slides=result.total_slides,
        )
        return result

    async def _extract_page(
        self, page: fitz.Page, slide_number: int, upload_id: str
    ) -> SlideContent:
        """Extract content from a single PDF page."""
        # Get structured text blocks with font info
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        text_parts: list[str] = []
        title: str | None = None
        max_font_size = 0.0

        for block in blocks:
            if block["type"] != 0:  # Skip non-text blocks (images etc.)
                continue
            for line in block.get("lines", []):
                line_text = ""
                line_font_size = 0.0
                for span in line.get("spans", []):
                    line_text += span["text"]
                    line_font_size = max(line_font_size, span["size"])

                line_text = line_text.strip()
                if not line_text:
                    continue

                text_parts.append(line_text)

                # Title heuristic: largest font on the page
                if line_font_size > max_font_size and len(line_text) > 2:
                    max_font_size = line_font_size
                    title = line_text

        text_content = "\n".join(text_parts)

        # Extract images
        image_paths: list[str] = []
        image_list = page.get_images(full=True)
        has_images = len(image_list) > 0

        if has_images:
            image_paths = self._extract_images(page, image_list, upload_id, slide_number)

        # If very little text but images exist, try OCR
        if len(text_content.strip()) < 50 and image_paths:
            ocr_results = self.ocr.extract_from_slide_images(image_paths)
            ocr_texts = [text for text, conf in ocr_results if conf > 0.3]
            if ocr_texts:
                ocr_text = "\n".join(ocr_texts)
                text_content = f"{text_content}\n\n[OCR extracted]\n{ocr_text}".strip()
                logger.debug(
                    "pdf_ocr_fallback",
                    slide_number=slide_number,
                    ocr_text_length=len(ocr_text),
                )

        return SlideContent(
            slide_number=slide_number,
            title=title,
            text_content=text_content,
            has_images=has_images,
            image_paths=image_paths,
            metadata={
                "page_width": page.rect.width,
                "page_height": page.rect.height,
                "image_count": len(image_list),
            },
        )

    def _extract_images(
        self,
        page: fitz.Page,
        image_list: list,
        upload_id: str,
        slide_number: int,
    ) -> list[str]:
        """Extract images from a PDF page and save to temp directory."""
        image_paths: list[str] = []
        doc = page.parent

        for img_idx, img_info in enumerate(image_list):
            try:
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                image_bytes = base_image["image"]
                image_ext = base_image.get("ext", "png")

                # Save to temp file
                temp_dir = Path(tempfile.gettempdir()) / "slideguide" / upload_id
                temp_dir.mkdir(parents=True, exist_ok=True)

                image_path = temp_dir / f"slide_{slide_number}_img_{img_idx}.{image_ext}"
                image_path.write_bytes(image_bytes)
                image_paths.append(str(image_path))

            except Exception as e:
                logger.warning(
                    "pdf_image_extract_failed",
                    slide_number=slide_number,
                    image_index=img_idx,
                    error=str(e),
                )

        return image_paths
