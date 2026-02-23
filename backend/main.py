"""
FastAPI application entry point for SlideGuide.

Provides file upload, document processing, retrieval, and
tutoring chat endpoints. Initializes Supabase, vector store, and
LLM clients on startup.
"""

from __future__ import annotations

import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.db.client import get_supabase
from backend.db.repositories.uploads import UploadRepository
from backend.db.repositories.slides import SlideRepository
from backend.db.repositories.storage import StorageRepository
from backend.models.schemas import (
    ErrorResponse,
    RetrieveRequest,
    SlideGuideError,
    UploadResponse,
)
from backend.monitoring.health import router as health_router
from backend.monitoring.logger import configure_logging, get_logger
from backend.monitoring.metrics import metrics
from backend.agent.nodes import tool_compat
from backend.routes.chat import router as chat_router
from backend.routes.settings import router as settings_router, set_tool_compat
from backend.rag.ingestion import IngestionPipeline
from backend.rag.retriever import HybridRetriever
from backend.rag.vectorstore import VectorStore

logger = get_logger(__name__)

# Shared instances (initialized at startup)
vectorstore = VectorStore()
ingestion_pipeline = IngestionPipeline(vectorstore)
retriever = HybridRetriever(vectorstore)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and tear down application resources."""
    configure_logging()
    logger.info("app_starting", environment=settings.environment)

    # Initialize Supabase client
    app.state.supabase = get_supabase()
    logger.info("supabase_connected")

    yield

    logger.info("app_shutdown")


app = FastAPI(
    title="SlideGuide API",
    description="AI-powered tutoring from lecture slides",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not settings.is_production else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health/metrics routes
app.include_router(health_router)

# Chat/session routes
app.include_router(chat_router)

# Settings/provider routes
app.include_router(settings_router)
set_tool_compat(tool_compat)


# ── Middleware ─────────────────────────────────────────────────────────────────


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach a unique request ID to every request for tracing."""
    request_id = str(uuid.uuid4())[:8]
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── Exception handlers ────────────────────────────────────────────────────────


@app.exception_handler(SlideGuideError)
async def slideguide_error_handler(request: Request, exc: SlideGuideError):
    metrics.record_error(type(exc).__name__, exc.detail)
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=exc.message,
            detail=exc.detail,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    metrics.record_error("InternalError", str(exc))
    logger.exception("unhandled_error", error=str(exc))
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred. Please try again.",
        ).model_dump(),
    )


# ── Upload endpoints ──────────────────────────────────────────────────────────


@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    """
    Upload a PDF or PPTX file for processing.

    The file is parsed, chunked, embedded, and stored in the vector database.
    Returns upload metadata including the upload_id for subsequent queries.
    """
    supabase = request.app.state.supabase
    upload_repo = UploadRepository(supabase)
    slide_repo = SlideRepository(supabase)

    # Validate file type
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    if ext not in (".pdf", ".pptx"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Validate file size
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.max_upload_size_mb}MB",
        )

    # Create upload record
    upload = upload_repo.create(
        filename=filename,
        file_type=ext.lstrip("."),
        file_size=len(content),
        status="PROCESSING",
    )

    upload_id = upload["id"]
    logger.info("upload_created", upload_id=upload_id, filename=filename, size=len(content))
    metrics.total_uploads += 1

    # Upload to Supabase Storage
    storage_repo = StorageRepository(supabase)
    content_type = "application/pdf" if ext == ".pdf" else "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    storage_path = storage_repo.upload_file(upload_id, filename, content, content_type)
    upload_repo.update(upload_id, storage_path=storage_path)

    # Save to temp file for parsing
    temp_dir = Path(tempfile.gettempdir()) / "slideguide" / "uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{upload_id}{ext}"
    temp_path.write_bytes(content)

    try:
        # Parse document
        from backend.parsers import parse_document

        parsed = await parse_document(str(temp_path), upload_id)

        # Store slides in database
        for slide in parsed.slides:
            slide_repo.create(
                upload_id=upload_id,
                slide_number=slide.slide_number,
                title=slide.title,
                text_content=slide.text_content,
                has_images=slide.has_images,
                image_paths=slide.image_paths,
                metadata=slide.metadata,
            )

        # Ingest into RAG pipeline
        chunk_count = await ingestion_pipeline.ingest(parsed)

        # Update upload status
        upload = upload_repo.update(
            upload_id,
            status="READY",
            total_slides=parsed.total_slides,
            metadata={"chunk_count": chunk_count},
        )

        logger.info(
            "upload_processed",
            upload_id=upload_id,
            total_slides=parsed.total_slides,
            chunks=chunk_count,
        )

    except Exception as e:
        upload_repo.update(
            upload_id,
            status="ERROR",
            metadata={"error": str(e)},
        )
        logger.error("upload_processing_failed", upload_id=upload_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    finally:
        # Cleanup temp file
        temp_path.unlink(missing_ok=True)

    return UploadResponse(
        id=upload["id"],
        filename=upload["filename"],
        file_type=upload["file_type"],
        status=upload["status"],
        total_slides=upload["total_slides"],
        created_at=upload["created_at"],
    )


@app.get("/api/upload/{upload_id}")
async def get_upload(request: Request, upload_id: str):
    """Get upload status and metadata."""
    supabase = request.app.state.supabase
    upload_repo = UploadRepository(supabase)

    upload = upload_repo.get_by_id(upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    return {
        "id": upload["id"],
        "filename": upload["filename"],
        "file_type": upload["file_type"],
        "file_size": upload["file_size"],
        "status": upload["status"],
        "total_slides": upload["total_slides"],
        "metadata": upload["metadata"],
        "created_at": upload["created_at"],
    }


@app.get("/api/upload/{upload_id}/slides")
async def get_slides(request: Request, upload_id: str):
    """Get all slides for an upload."""
    supabase = request.app.state.supabase
    upload_repo = UploadRepository(supabase)
    slide_repo = SlideRepository(supabase)

    upload = upload_repo.get_by_id(upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    slides = slide_repo.get_by_upload(upload_id)

    return {
        "upload_id": upload_id,
        "total_slides": len(slides),
        "slides": [
            {
                "slide_number": s["slide_number"],
                "title": s["title"],
                "text_content": s["text_content"],
                "has_images": s["has_images"],
            }
            for s in slides
        ],
    }


# ── Retrieval endpoint ────────────────────────────────────────────────────────


@app.post("/api/retrieve")
async def retrieve(request: Request, body: RetrieveRequest):
    """
    Search the knowledge base for content relevant to a query.

    Uses hybrid search (semantic + keyword) with RRF fusion and MMR diversity.
    Returns ranked results with slide citations.
    """
    supabase = request.app.state.supabase
    upload_repo = UploadRepository(supabase)

    # Verify upload exists and is ready
    upload = upload_repo.get_by_id(body.upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")
    if upload["status"] != "READY":
        raise HTTPException(status_code=400, detail=f"Upload not ready: {upload['status']}")

    # Apply slide range filter
    slide_filter = None
    if body.slide_range and len(body.slide_range) == 2:
        slide_filter = body.slide_range[0]  # Simplified: filter by start slide

    results = await retriever.retrieve(
        query=body.query,
        upload_id=body.upload_id,
        n_results=body.n_results,
        slide_filter=slide_filter,
    )

    return {
        "query": body.query,
        "upload_id": body.upload_id,
        "results": [
            {
                "content": r.content,
                "slide_number": r.metadata.slide_number,
                "title": r.metadata.title,
                "content_type": r.metadata.content_type,
                "score": round(r.score, 4),
                "source": r.source,
            }
            for r in results
        ],
        "total_results": len(results),
    }
