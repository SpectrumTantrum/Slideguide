# SlideGuide

AI-powered tutoring from your lecture slides. Upload a PDF or PPTX, and SlideGuide creates a personalized study session with adaptive explanations, interactive quizzes, and progress tracking — designed for neurodivergent learners.

## Architecture

```mermaid
graph TB
    subgraph Frontend["Next.js Frontend"]
        Upload[Upload Page]
        Session[Session Page]
        Chat[Chat Interface]
        Slides[Slide Viewer]
        Progress[Progress Dashboard]
    end

    subgraph API["FastAPI Backend"]
        Router[API Router]
        Parsers[Document Parsers]
        RAG[Hybrid RAG Pipeline]
        Agent[LangGraph Agent]
        Memory[Memory System]
    end

    subgraph External["External Services"]
        OpenRouter[OpenRouter LLM]
        OpenAI[OpenAI Embeddings]
    end

    subgraph Supabase["Supabase"]
        Postgres[(PostgreSQL + pgvector)]
        Storage[Supabase Storage]
    end

    subgraph Local["Local (optional)"]
        LMStudio[LM Studio]
    end

    Upload -->|PDF/PPTX| Router
    Chat -->|SSE Stream| Router
    Router --> Parsers --> RAG
    Router --> Agent
    Agent -->|Semantic + full-text| RAG
    Agent -->|State Persistence| Memory
    RAG -->|Vector search| Postgres
    RAG --> OpenAI
    Agent --> OpenRouter
    Agent -.->|Local alternative| LMStudio
    Memory --> Postgres
    Parsers -->|File storage| Storage
```

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | Next.js 14, TypeScript, Tailwind CSS, Zustand | UI and state management |
| Backend | FastAPI, Python 3.11+ | API server |
| Agent | LangGraph | Multi-node stateful tutoring agent |
| LLM | OpenRouter (Claude Sonnet/Haiku, DeepSeek fallback) or LM Studio (local) | Reasoning and generation |
| Embeddings | OpenAI text-embedding-3-small or local embedding model | Semantic search vectors |
| Database | Supabase (PostgreSQL + pgvector) | Vector search, sessions, progress, cost tracking |
| Storage | Supabase Storage | Uploaded file persistence |
| RAG | Hybrid search (semantic + full-text) → RRF → MMR | Retrieval pipeline |

## Key Features

- **5 explanation modes**: Standard, Analogy, Visual, Step-by-Step, ELI5
- **3 pacing levels**: Slow, Medium, Fast
- **Adaptive quizzes**: Difficulty auto-adjusts based on performance
- **Hybrid retrieval**: Semantic + keyword search with diversity ranking
- **VLM image understanding**: Describes charts, diagrams, and images from slides
- **Progress tracking**: Topics covered, quiz scores, confidence levels
- **SSE streaming**: Real-time token-by-token response streaming
- **Circuit breaker**: Automatic fallback between LLM providers
- **Local LLM support**: Run entirely offline with LM Studio — auto-discovers models, adapts tool calling

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- [Docker](https://docs.docker.com/get-docker/) (required by Supabase CLI)
- [Supabase CLI](https://supabase.com/docs/guides/cli) (or a hosted Supabase project)
- OpenRouter API key (cloud mode) **or** [LM Studio](https://lmstudio.ai/) (local mode)
- OpenAI API key (for embeddings in cloud mode)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (optional — only needed for OCR on image-heavy slides)

### 1. Clone and configure

```bash
git clone https://github.com/yourusername/slideguide.git
cd slideguide
cp .env.example .env
# Edit .env with your API keys and Supabase credentials
```

### 2. Start Supabase

```bash
# Start local Supabase (runs PostgreSQL with pgvector, Storage, and more)
supabase start

# Apply database migrations
supabase db reset
```

This starts PostgreSQL with pgvector (port 54322), Supabase API (port 54321), and Supabase Storage. The migrations create all required tables, enable pgvector, and configure the storage bucket.

After `supabase start` finishes, it prints your local credentials. Copy the `anon key` and `service_role key` values into your `.env`:

```bash
SUPABASE_URL=http://127.0.0.1:54321
SUPABASE_ANON_KEY=<anon key from supabase start output>
SUPABASE_SERVICE_ROLE_KEY=<service_role key from supabase start output>
```

### 3. Backend setup

```bash
# Install Python dependencies
pip install -e ".[dev]"

# Start the API server
uvicorn backend.main:app --reload --port 8000
```

### 4. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000` to start using SlideGuide.

### 5. Run tests

```bash
pytest tests/ -v
```

## Using Local LLMs (LM Studio)

SlideGuide can run entirely offline using local models via [LM Studio](https://lmstudio.ai/), with no API keys required for chat. This uses the same OpenAI-compatible API that the cloud path uses, so the switch is purely configuration.

### Quick start

1. **Install and launch [LM Studio](https://lmstudio.ai/)**
2. **Download a model** — any GGUF model works. Recommended:
   - Chat: `mistral-nemo-instruct`, `llama-3.1-8b-instruct`, or `qwen2.5-7b-instruct`
   - Embeddings: `nomic-embed-text-v1.5` or `bge-small-en-v1.5`
3. **Load the model** and start LM Studio's local server (default: `http://localhost:1234`)
4. **Set your `.env`**:

```bash
# Switch providers to local
LLM_PROVIDER=lmstudio
EMBEDDING_PROVIDER=lmstudio    # optional — keeps OpenAI embeddings if omitted
VISION_PROVIDER=lmstudio       # optional — only if your model supports vision

# LM Studio connection
LMSTUDIO_BASE_URL=http://localhost:1234/v1

# Model names — leave empty to auto-discover from LM Studio
LMSTUDIO_PRIMARY_MODEL=
LMSTUDIO_ROUTING_MODEL=
LMSTUDIO_EMBEDDING_MODEL=
```

5. **Start SlideGuide normally** — the backend auto-discovers loaded models from LM Studio.

### Provider configuration

Each capability (chat, embeddings, vision) can be pointed at a different provider independently:

| Variable | Options | Default |
|----------|---------|---------|
| `LLM_PROVIDER` | `openrouter`, `lmstudio` | `openrouter` |
| `EMBEDDING_PROVIDER` | `openrouter`, `lmstudio` | `openrouter` |
| `VISION_PROVIDER` | `openrouter`, `lmstudio` | `openrouter` |

**Hybrid example** — local chat with cloud embeddings (best quality retrieval, free generation):

```bash
LLM_PROVIDER=lmstudio
EMBEDDING_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
```

### How it works

- **Auto-discovery**: On startup, the backend queries `GET /v1/models` on LM Studio to find loaded models. If `LMSTUDIO_PRIMARY_MODEL` is empty, it picks the first available model.
- **Tool compatibility**: Local models have inconsistent function-calling support. SlideGuide starts with native OpenAI-format tool calling, and if the model fails to produce valid tool calls 3 times in a row, it auto-switches to a prompt-based fallback that injects tool schemas into the system prompt and parses JSON blocks from the response.
- **Zero cost tracking**: All local model calls are tracked at $0.00 — no cost limits apply.
- **Health checks**: `GET /api/settings/provider` reports LM Studio connectivity and loaded model count.
- **No fallback chain**: Unlike cloud mode (which falls back from Claude to DeepSeek), local mode uses a single model with no fallback.

### Verifying the connection

Once running, check the provider status:

```bash
curl http://localhost:8000/api/settings/provider
```

You should see:

```json
{
  "llm_provider": "lmstudio",
  "models": { "primary": "your-model-name", ... },
  "lmstudio": { "status": "ok", "models_loaded": 1 }
}
```

### Tips

- **RAM**: 7B models need ~6 GB RAM, 13B models need ~10 GB. Keep this in mind alongside Supabase services.
- **GPU offloading**: Enable GPU layers in LM Studio for much faster inference.
- **Routing model**: If unset, the primary model handles both reasoning and routing. For faster routing, load a smaller model and set `LMSTUDIO_ROUTING_MODEL` to its name.
- **Embedding model**: Must be loaded separately in LM Studio alongside your chat model. If you skip local embeddings, keep `EMBEDDING_PROVIDER=openai` — OpenAI's embeddings are cheap and high quality.

## Project Structure

```
slideguide/
├── backend/
│   ├── agent/          # LangGraph tutoring agent
│   │   ├── graph.py    # Graph assembly and routing
│   │   ├── nodes.py    # Agent nodes (router, explain, quiz, etc.)
│   │   ├── prompts.py  # Neurodivergent-friendly prompt templates
│   │   ├── state.py    # TutorState schema
│   │   └── tools.py    # 7 agent tools (search, quiz, progress, etc.)
│   ├── db/             # Supabase data layer
│   │   ├── client.py   # Supabase client singleton
│   │   └── repositories/
│   │       ├── chunks.py    # pgvector chunk storage + search
│   │       ├── messages.py  # Chat history CRUD
│   │       ├── progress.py  # Student progress CRUD
│   │       ├── sessions.py  # Session CRUD
│   │       ├── slides.py    # Slide content CRUD
│   │       ├── storage.py   # Supabase Storage operations
│   │       └── uploads.py   # Upload metadata CRUD
│   ├── llm/            # LLM clients
│   │   ├── client.py   # OpenRouter with retry + circuit breaker
│   │   ├── discovery.py # LM Studio model auto-discovery
│   │   ├── models.py   # Model configs and pricing
│   │   ├── providers.py # Provider config resolution (cloud vs local)
│   │   ├── streaming.py # SSE stream handler
│   │   ├── tool_compatibility.py # Native ↔ prompt-based tool use adapter
│   │   └── vision.py   # VLM image understanding
│   ├── memory/         # Persistence layer
│   │   ├── session_memory.py    # Conversation summarization
│   │   └── student_progress.py  # Long-term progress tracking
│   ├── models/
│   │   └── schemas.py  # All Pydantic models
│   ├── monitoring/     # Observability
│   │   ├── health.py   # Health checks (liveness, readiness)
│   │   ├── logger.py   # Structured logging (structlog)
│   │   └── metrics.py  # Cost and performance tracking
│   ├── parsers/        # Document parsing
│   │   ├── pdf_parser.py   # PyMuPDF
│   │   ├── pptx_parser.py  # python-pptx
│   │   └── ocr.py          # Tesseract + VLM fallback
│   ├── rag/            # Retrieval pipeline
│   │   ├── vectorstore.py  # pgvector wrapper via Supabase
│   │   ├── ingestion.py    # Chunking + embedding
│   │   ├── retriever.py    # Hybrid search → RRF → MMR
│   │   └── evaluation.py   # Retrieval metrics logging
│   ├── routes/
│   │   ├── chat.py     # Session and message API endpoints
│   │   └── settings.py # Provider status and model listing
│   ├── config.py       # Application settings
│   └── main.py         # FastAPI app entry point
├── frontend/
│   ├── app/            # Next.js app router pages
│   ├── components/     # React components
│   ├── lib/            # API client, store, types, utils
│   └── package.json
├── supabase/
│   ├── config.toml     # Local Supabase CLI config
│   └── migrations/     # SQL migrations (schema, pgvector, storage)
├── tests/              # Python tests
└── pyproject.toml      # Python project config
```

## Skills Showcase

| Skill | Implementation |
|-------|---------------|
| **RAG Pipeline** | Hybrid search (semantic + PostgreSQL full-text), Reciprocal Rank Fusion, MMR diversity ranking |
| **Agentic AI** | LangGraph multi-node graph with conditional routing, tool calling, state persistence |
| **LLM Engineering** | Retry with exponential backoff, circuit breaker, model fallback chain, cost tracking, local LLM support via LM Studio |
| **Provider Abstraction** | Pluggable provider config, auto-discovery of local models, adaptive tool-calling compatibility layer |
| **Prompt Engineering** | 5 explanation modes, adaptive quiz difficulty, neurodivergent-friendly formatting |
| **Document Processing** | PDF (PyMuPDF) + PPTX parsing, OCR with VLM fallback, slide-aware chunking |
| **Multimodal** | VLM image descriptions for charts/diagrams, base64 encoding, context injection |
| **Streaming** | SSE token-by-token streaming, tool call assembly, heartbeat keepalive |
| **Observability** | Structured logging (structlog), per-model metrics, health checks (live/ready) |
| **Database Design** | Supabase (PostgreSQL + pgvector), repository pattern, SQL migrations, Storage API |
| **Frontend** | Next.js 14, Zustand state, SSE consumption, responsive 3-column layout, dark mode |

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload a PDF or PPTX file for processing |
| `GET` | `/api/upload/{upload_id}` | Get upload status and metadata |
| `GET` | `/api/upload/{upload_id}/slides` | List all slides for an upload |
| `POST` | `/api/session` | Create a new tutoring session |
| `GET` | `/api/session/{session_id}` | Get session state |
| `POST` | `/api/session/{session_id}/message` | Send a message (returns SSE stream) |
| `GET` | `/api/session/{session_id}/history` | Get chat history for a session |
| `GET` | `/api/settings/provider` | Get current provider configuration |
| `GET` | `/api/settings/models` | List available models |
| `GET` | `/health/live` | Liveness check |
| `GET` | `/health/ready` | Readiness check |

## License

This project is licensed under the AGPL-3.0 — see [LICENSE](LICENSE) for details.
