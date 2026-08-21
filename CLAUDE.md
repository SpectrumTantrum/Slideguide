# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SlideGuide is an AI-powered tutoring app that processes lecture slides (PDF/PPTX) and creates an interactive tutoring experience. It uses a LangGraph agent with a 6-phase state machine, hybrid RAG retrieval, and SSE streaming.

## Development Commands

### Backend (Python — run from repo root)

```bash
# Install dependencies
pip install -e ".[dev]"

# Start backend server
uvicorn backend.main:app --reload --port 8000

# Run all tests
pytest

# Run a single test file
pytest tests/test_agent.py

# Run a specific test
pytest tests/test_agent.py::test_initial_state_creation -v

# Cursor SDK e2e (live tests also need SLIDEGUIDE_LIVE_CURSOR=1)
pytest tests/e2e -v
# Live only:
# SLIDEGUIDE_LIVE_CURSOR=1 pytest tests/e2e -m live -v

# Lint
ruff check backend/ tests/

# Type check
mypy backend/
```

### Frontend (Next.js — run from `frontend/`)

```bash
cd frontend
npm install
npm run dev      # Dev server on :3000 (proxies API to :8000)
npm run build    # Production build
npm run lint     # ESLint
```

### Supabase (local)

```bash
supabase start          # Starts local Supabase (provides keys in output)
supabase db reset       # Reset and re-run migrations
supabase migration new <name>   # Create a new migration
```

## Architecture

**Two-process dev setup**: Next.js frontend (:3000) proxies `/api/*` to FastAPI backend (:8000) via `next.config.js` rewrites.

### Backend Layers (`backend/`)

- **`agent/`** — LangGraph tutoring agent. State machine with 6 phases: `greeting → topic_selection → teaching → quiz → review → wrap_up`. Nodes: `router`, `explain`, `quiz`, `summarize`, `encourage`, `clarify`, `tool_executor`. Conditional routing based on phase and tool calls.
- **`rag/`** — Hybrid retrieval pipeline. Semantic search (OpenAI-compatible embeddings in pgvector) + keyword search (PostgreSQL full-text) fused via Reciprocal Rank Fusion, then diversified with MMR. Slide-aware chunking preserves metadata (slide number, content type).
- **`llm/`** — OpenAI-compatible HTTP stays in `LLMClient`. Cursor is one translator (`cursor.py`). Which SDK bills a turn is read only from `llm/runtime.py` (process default `LLM_PROVIDER`, plus a per-session / request bind). Embeddings stay on the OpenAI-compatible endpoint. Includes retry, **per-SDK** circuit breakers, and a tool compatibility adapter (native ↔ prompt-based; Cursor is always prompt-based).
- **`db/`** — Repository pattern over Supabase. Separate repository classes per entity (`uploads`, `sessions`, `messages`, `progress`, `slides`, `storage`). Client singleton via `get_supabase()`.
- **`routes/`** — FastAPI routers for chat (SSE streaming) and settings/provider management.
- **`parsers/`** — PDF (PyMuPDF) and PPTX (python-pptx) parsing with optional Tesseract OCR for images.
- **`monitoring/`** — structlog with request ID middleware, health endpoints (`/health/live`, `/health/ready`), cost/error metrics.
- **`models/`** — Pydantic schemas shared across layers.
- **`config.py`** — Single `Settings` class (pydantic-settings) loading from `.env`. Holds the OpenAI-compatible endpoint (`OPENAI_BASE_URL`, `OPENAI_API_KEY`) and model names. `active_*` properties are env-only; session SDK choice lives in `llm/runtime.py`.
- **`memory/`** — Session context window management (summarizes overflow) and student progress tracking (confidence, quiz scores, coverage). Sits on top of LangGraph checkpointer and Supabase tables.

### Frontend (`frontend/`)

- Next.js 14 App Router with TypeScript
- Zustand store (`lib/store.ts`) for session/message state
- SSE streaming consumption with AbortController for cancellation
- TailwindCSS with dark mode support
- Path alias: `@/*` maps to project root

### Database (`supabase/migrations/`)

Three migrations: initial schema, pgvector chunks, and storage bucket setup. Local Supabase with PostgreSQL + pgvector extension.

## Key Patterns

- **Chat SDK**: `LLM_PROVIDER=openai|cursor` is the process default for new sessions. Cursor requires `CURSOR_API_KEY` and prefers `grok-4.6` at `CURSOR_REASONING_EFFORT=high`, then Composer 2.5 / Composer 2 (not `auto-smart`). `POST /api/settings/provider` requires `session_id` and writes `sessions.metadata.chat_sdk` for that session only. The banner does not auto-POST from `localStorage`. Embeddings always use `OPENAI_BASE_URL` + `EMBEDDING_MODEL` (1536-d). Empty OpenAI API keys get a placeholder so keyless local endpoints work.
- **SSE events**: Chat streaming emits `token`, `phase_change`, `error`, `done` event types.
- **Agent state** (`TutorState`): Append-only messages, phase tracking, student profile (confidence, consecutive correct/incorrect), teaching preferences (explanation mode, pacing level).
- **Shared instances**: `vectorstore`, `ingestion_pipeline`, `retriever` are module-level singletons in `main.py`, initialized at import time.
- **Tool compatibility**: `ToolCompatibilityLayer` auto-switches between native (OpenAI format) and prompt-based tool calling after 3 consecutive parse failures. The Cursor SDK route is always prompt-based. See `llm/tool_compatibility.py`.
- **Model discovery**: `llm/discovery.py` lists models for a given chat SDK (`/v1/models` or `Cursor.models.list()`). Cursor-owned models are sorted first on that route.
- **Checkpointing**: LangGraph uses `MemorySaver` (in-memory) by default; `AsyncPostgresSaver` available for persistent state. Thread ID = session ID.
- **Cost tracking**: Per-model token/cost aggregation in `monitoring/metrics.py`. Recognized model IDs are priced via `MODEL_PRICING`; unknown/local models are tracked at $0.00.

## Code Style

- **Python**: ruff with rules `E, F, I, N, W, UP`. Line length 100. Target Python 3.11+. Async-first with `asyncio_mode = "auto"` in pytest.
- **Frontend**: ESLint via `eslint-config-next`. TailwindCSS for styling. Strict TypeScript.
- **Imports**: Use `from __future__ import annotations` in Python modules. Backend modules imported as `from backend.x.y import Z`.
