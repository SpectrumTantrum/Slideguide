"""Document parsers for PDF, PPTX, and OCR extraction."""

from __future__ import annotations

from pathlib import Path

from backend.models.schemas import ParsedDocument


async def parse_document(file_path: str, upload_id: str) -> ParsedDocument:
    """
    Dispatch document parsing based on file extension.

    Routes to the appropriate parser (PDF or PPTX) and returns
    a structured ParsedDocument with per-slide content.
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        from backend.parsers.pdf_parser import PdfParser

        parser = PdfParser()
        return await parser.parse(file_path, upload_id)
    elif ext in (".pptx", ".ppt"):
        from backend.parsers.pptx_parser import PptxParser

        parser = PptxParser()
        return await parser.parse(file_path, upload_id)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Only PDF and PPTX are supported.")
