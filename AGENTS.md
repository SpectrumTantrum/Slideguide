# AGENTS.md

## Cursor Cloud specific instructions

### Services overview

SlideGuide has three services: a **Next.js frontend** (:3000), a **FastAPI backend** (:8000), and **Supabase** (PostgreSQL + pgvector, managed via Supabase CLI + Docker). See `README.md` and `CLAUDE.md` for standard dev commands.

### Gotchas

- **`pyproject.toml` build backend**: The declared build backend (`setuptools.backends._legacy:_Backend`) does not exist in any released version of setuptools. To install Python deps, install them directly from `pyproject.toml` dependency lists rather than using `pip install -e ".[dev]"`. Alternatively, install all deps listed in `[project.dependencies]` and `[project.optional-dependencies.dev]` with plain `pip install`. You must also set `PYTHONPATH=/workspace` so the `backend` package is importable.
- **Docker in Cloud VM**: Docker requires `fuse-overlayfs` storage driver and `iptables-legacy` for the nested container environment. The daemon config at `/etc/docker/daemon.json` must specify `"storage-driver": "fuse-overlayfs"`. Docker socket permissions need `chmod 666 /var/run/docker.sock` after starting `dockerd`.
- **Supabase keys**: After `supabase start`, copy the `ANON_KEY` and `SERVICE_ROLE_KEY` from `supabase status` output into `.env`. The local Supabase keys are deterministic demo keys.
- **ESLint config**: The frontend ships without an `.eslintrc.json` file; `npx next lint` will prompt interactively. Create `frontend/.eslintrc.json` with `{"extends": "next/core-web-vitals"}` to enable non-interactive linting.
- **OpenRouter API key**: Full upload/chat functionality requires a valid `OPENROUTER_API_KEY` in `.env`. Without it, health checks, settings, and Supabase endpoints still work, but document upload and chat sessions will fail on embedding/LLM calls.
- **Pre-existing test issues**: `tests/test_llm_client.py` has a broken import (`FALLBACK_CHAIN`); `tests/test_parsers.py::test_unsupported_file_type_raises` expects `SlideParsingError` but code raises `ValueError`. Run tests with `--ignore=tests/test_llm_client.py` to avoid the collection error.

### Starting services

1. **Docker**: `sudo dockerd &>/tmp/dockerd.log &` then `sudo chmod 666 /var/run/docker.sock`
2. **Supabase**: `cd /workspace && supabase start` (takes ~90s first time, pulls Docker images)
3. **Backend**: `PYTHONPATH=/workspace uvicorn backend.main:app --reload --port 8000`
4. **Frontend**: `cd /workspace/frontend && npm run dev`

### Lint / Test / Build

- **Ruff**: `ruff check backend/ tests/` (pre-existing warnings exist)
- **Mypy**: `mypy backend/` (pre-existing type errors exist)
- **ESLint**: `cd frontend && npx next lint`
- **Pytest**: `PYTHONPATH=/workspace pytest tests/ -v --ignore=tests/test_llm_client.py`
- **Frontend build**: `cd frontend && npm run build`
