# Test Coverage Analysis

## Current State

### Existing Tests (5 files, ~56 test cases)

| Test File | Covers | Tests | What's Tested |
|---|---|---|---|
| `test_agent.py` | `agent/state.py`, `agent/graph.py`, `agent/prompts.py` | 13 | Initial state defaults, graph routing functions, prompt templates, quiz difficulty scaling, encouragement messages |
| `test_llm_client.py` | `llm/models.py`, `llm/client.py` (CircuitBreaker only), `monitoring/metrics.py` | 12 | Model registry, fallback chain, cost estimation, circuit breaker state machine, MetricsCollector recording |
| `test_parsers.py` | `models/schemas.py` (ParsedDocument, SlideContent), `parsers/ocr.py`, `parsers/__init__.py` | 5 | Schema defaults, OCR unavailability, unsupported file type error |
| `test_rag.py` | `models/schemas.py` (RetrievalResult, ChunkMetadata), `rag/retriever.py` | 8 | Data model creation, text similarity (Jaccard), result dedup key, reciprocal rank fusion |
| `test_tools.py` | `agent/tools.py` | 5 | Tool schema structure, registry completeness, parameter type validity |

### Backend Source Files: 27 modules

### Frontend: 0 tests (no test framework configured)

---

## Coverage Gap Analysis

### Severity: CRITICAL (no tests at all)

#### 1. `backend/routes/chat.py` — Chat API endpoints
- **Risk**: This is the core user-facing API. Session creation, SSE message streaming, history retrieval, and the `_chunk_response` helper are all untested.
- **What to test**:
  - `_chunk_response()` — pure function, easy to unit test for word-boundary splitting
  - `_sse_event()` — pure function, verify JSON structure
  - `create_session` — mock DB and graph, verify 404 on missing upload, verify correct `SessionState` response
  - `send_message` — mock graph invocation, test SSE event generation
  - `get_session` / `get_history` — mock DB, verify 404 handling and pagination

#### 2. `backend/routes/settings.py` — Settings API
- **Risk**: Low complexity but no validation of the response format.
- **What to test**:
  - `get_provider_config()` response structure with different provider configurations
  - `get_available_models()` response for openrouter vs lmstudio

#### 3. `backend/llm/tool_compatibility.py` — Tool compatibility layer
- **Risk**: High. This is the adaptive layer that switches between native and prompt-based tool calling. A bug here silently breaks all tool use.
- **What to test**:
  - `_build_tool_prompt()` — pure function, verify tool schema injection format
  - `_parse_tool_calls_from_text()` — pure function, test with valid fenced JSON, malformed JSON, multiple blocks, no blocks
  - `ToolCompatibilityLayer` mode switching — verify native failure threshold triggers switch to prompt mode
  - `wrap_chat_call()` — mock LLM client, test native path, prompt-based path, no-tools path

#### 4. `backend/llm/streaming.py` — SSE streaming handler
- **Risk**: Medium-high. Bugs here cause dropped tokens or broken tool calls in the UI.
- **What to test**:
  - `SSEHandler._format_event()` — pure function
  - `SSEHandler._assemble_tool_call()` — incremental tool call assembly from partial deltas, verify complete JSON detection
  - `SSEHandler.stream_response()` — mock async generator, verify event sequence

#### 5. `backend/llm/providers.py` — Provider config resolution
- **Risk**: Medium. Misconfiguration silently points at the wrong API.
- **What to test**:
  - `get_chat_provider_config()` — verify returns correct base_url/api_key for each provider
  - `get_embedding_provider_config()` — same
  - `get_vision_provider_config()` — same
  - `ProviderConfig.client_kwargs()` — verify headers are included/excluded correctly

#### 6. `backend/llm/vision.py` — Vision client
- **Risk**: Medium. VLM integration for image descriptions.
- **What to test**:
  - `VisionClient._encode_image()` — pure function, test with existing file, missing file, unreadable file
  - `VisionClient._call_vision()` — test unavailable client returns fallback message
  - Availability detection based on API key presence

#### 7. `backend/llm/discovery.py` — LM Studio model discovery
- **Risk**: Low-medium. Caching and health checks.
- **What to test**:
  - `invalidate_cache()` clears state
  - `discover_local_models()` — mock httpx, test success, timeout, error responses, caching behavior
  - `auto_select_model()` — test with/without loaded models

#### 8. `backend/rag/ingestion.py` — RAG ingestion pipeline
- **Risk**: High. Chunking bugs cause silent retrieval quality degradation.
- **What to test**:
  - `IngestionPipeline._split_text()` — pure function; test short text (single chunk), long text (multiple chunks with overlap), sentence boundary detection, empty text
  - `IngestionPipeline._chunk_slide()` — verify text chunks, speaker notes as separate chunk, tables as separate chunks
  - `IngestionPipeline._make_chunk_id()` — deterministic, verify same inputs produce same hash
  - `IngestionPipeline._bm25_index_path()` — path construction

#### 9. `backend/rag/vectorstore.py` — ChromaDB vector store
- **Risk**: Medium. Hard to unit test without ChromaDB, but `_collection_name()` and `_embedding_model_tag()` are pure.
- **What to test**:
  - `_embedding_model_tag()` — deterministic hash
  - `VectorStore._collection_name()` — safe ID generation, length limits, embedding tag included

#### 10. `backend/rag/evaluation.py` — Retrieval evaluator
- **Risk**: Low. Logging only, no business logic.
- **What to test**:
  - `log_retrieval()` with empty results vs populated results (verify no crashes)

#### 11. `backend/memory/session_memory.py` — Session memory management
- **Risk**: Medium-high. Context window construction and summarization trigger.
- **What to test**:
  - `SessionMemory._messages_to_text()` — pure function
  - `SessionMemory._format_retrieval_context()` — pure function, test with >5 results (capped)
  - `SessionMemory.build_context_window()` — verify system prompt construction with/without summary, with/without retrieval context
  - `maybe_summarize()` — verify threshold check (does nothing below 20 messages)

#### 12. `backend/memory/student_progress.py` — Student progress tracking
- **Risk**: Medium. Confidence computation has specific business logic.
- **What to test**:
  - `StudentProgressTracker._compute_confidence()` — pure method; test below MIN_QUIZ_ATTEMPTS, high accuracy, low accuracy, mixed topic coverage
  - `suggest_next_topic()` — mock DB, verify priority ordering (uncovered > low-accuracy > first available)

#### 13. `backend/config.py` — Settings and computed properties
- **Risk**: Low-medium. Misconfigured properties could route to wrong models.
- **What to test**:
  - Computed properties: `is_local_llm`, `active_primary_model`, `active_routing_model`, `active_embedding_model`, `active_vision_model`, `chromadb_url`, `max_upload_bytes` for both openrouter and lmstudio configurations

#### 14. `backend/main.py` — FastAPI app setup
- **Risk**: Medium. Exception handlers and middleware.
- **What to test**:
  - `request_id_middleware` — verify X-Request-ID header is set
  - `slideguide_error_handler` — verify 400 response shape
  - `general_error_handler` — verify 500 response shape
  - Upload endpoint validation: unsupported file type, file too large

#### 15. `backend/parsers/pdf_parser.py` and `backend/parsers/pptx_parser.py`
- **Risk**: High. These are complex parsers that handle real files.
- **What to test for PdfParser**:
  - Title heuristic (largest font size extraction)
  - OCR fallback trigger (text < 50 chars with images)
- **What to test for PptxParser**:
  - `_table_to_markdown()` — pure function, verify markdown table formatting
  - `_get_title()` — title placeholder vs fallback to first text shape

#### 16. Frontend — All components and utilities (0% coverage)
- **Risk**: High for `lib/api.ts` and `lib/store.ts`; medium for components.
- **What to test**:
  - `lib/api.ts`: `ApiError` class, `request()` helper error handling, SSE parsing in `streamMessage()`
  - `lib/store.ts`: Zustand store actions, optimistic message addition, streaming state management, reset behavior
  - Components: `ChatInterface`, `SlideUploader`, `QuizCard` user interactions

---

## Severity-Ranked Recommendations

### Priority 1: High-value pure functions (easiest wins, highest ROI)

These require zero mocking and can be added immediately:

| Function | Module | Why |
|---|---|---|
| `_split_text()` | `rag/ingestion.py` | Chunking bugs silently degrade retrieval quality |
| `_chunk_slide()` | `rag/ingestion.py` | Ensures speaker notes, tables get separate chunks |
| `_parse_tool_calls_from_text()` | `llm/tool_compatibility.py` | Prompt-based tool parsing is fragile |
| `_build_tool_prompt()` | `llm/tool_compatibility.py` | Incorrect prompt format breaks tool use |
| `_assemble_tool_call()` | `llm/streaming.py` | Partial delta assembly is error-prone |
| `_chunk_response()` | `routes/chat.py` | Word boundary splitting for SSE |
| `_table_to_markdown()` | `parsers/pptx_parser.py` | Markdown table formatting |
| `_messages_to_text()` | `memory/session_memory.py` | Conversation serialization |
| `_format_retrieval_context()` | `memory/session_memory.py` | Context capping at 5 results |
| `_compute_confidence()` | `memory/student_progress.py` | Business-critical confidence scoring |
| `_collection_name()` | `rag/vectorstore.py` | Collection name length/character safety |

### Priority 2: Critical integration paths (require mocking)

| Area | Module | Why |
|---|---|---|
| Tool compatibility mode switching | `llm/tool_compatibility.py` | Silent failures break all tool use |
| Provider config resolution | `llm/providers.py` | Wrong base URL = wrong API |
| Session creation API | `routes/chat.py` | Core user flow |
| Config computed properties | `config.py` | Model routing depends on these |

### Priority 3: Frontend test infrastructure

- Add Jest + React Testing Library to `frontend/package.json`
- Start with `lib/api.ts` unit tests (error handling, SSE parsing)
- Add `lib/store.ts` tests (Zustand store logic)
- Add component tests for `ChatInterface` and `SlideUploader`

### Priority 4: End-to-end and integration

- FastAPI `TestClient` tests for upload validation (file type, file size)
- Full agent graph smoke test with mocked LLM responses
- SSE streaming integration test verifying event sequence

---

## Structural Observations

1. **No test fixtures or conftest.py**: The tests lack shared fixtures for common objects like `TutorState`, `ParsedDocument`, or `RetrievalResult`. A `conftest.py` with factory functions would reduce duplication.

2. **No mocking infrastructure**: The existing tests avoid any mocking, which limits them to pure functions and data models. Adding `pytest-mock` or using `unittest.mock` would unlock testing of the LLM client, DB interactions, and HTTP endpoints.

3. **No CI test execution**: There is no GitHub Actions or CI configuration running tests on push/PR.

4. **`pytest-cov` is installed but unused**: The dev dependencies include `pytest-cov` but there's no coverage configuration or thresholds. Adding `--cov=backend --cov-fail-under=50` to pytest config would prevent coverage regression.

5. **Frontend has zero test infrastructure**: No test runner (Jest/Vitest), no testing library, no test scripts in `package.json`.
