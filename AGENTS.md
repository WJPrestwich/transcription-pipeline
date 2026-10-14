# AGENTS.md

## Project Overview

CLI + MCP server for downloading YouTube and Twitch audio (Twitch clips and VODs), isolating vocals with Demucs, transcribing with whisperx, and diarizing with WhisperX's pyannote pipeline. Transcripts are persisted to PostgreSQL.

## Tech Stack

- Python 3.13+
- `uv` for package/environment management
- `click` for CLI interface
- `fastmcp` for MCP server (HTTP transport)
- `pydantic` + `pydantic-settings` for config (loaded from `.env`, `extra="ignore"`)
- `sqlalchemy[asyncio]` + `asyncpg` for async PostgreSQL access
- `whisperx` for transcription + alignment + speaker diarization (`DiarizationPipeline`, backed by pyannote)
- `demucs` for vocal isolation
- `torch` / `torchaudio` as ML backend
- `ruff` for linting and formatting
- `pytest` + `pytest-asyncio` + `testcontainers[postgres]` for testing

## Project Structure

```
src/youtube_transcription/
  __init__.py
  cli.py        # CLI — runs the pipeline and DB access directly (no MCP server needed)
  pipeline.py   # Core ML pipeline functions + run_pipeline orchestration
  server.py     # FastMCP server with tools: transcribe, list_transcripts, get_transcript
  schemas.py    # Pydantic models for MCP tool inputs/outputs (VideoMetadata base, Annotated params)
  db.py         # SQLAlchemy async database operations
  models.py     # SQLAlchemy ORM models (Transcript, Segment)
  settings.py   # Pydantic settings (env-based config)
tests/
  conftest.py       # Docker PostgreSQL fixture, settings reset
  test_pipeline.py  # Pipeline unit tests (diarize, validate_url, ProcessedResult, subprocess wrappers)
  test_db.py         # Database CRUD tests (real PostgreSQL)
  test_server.py     # MCP tool + REST route tests (mocked pipeline, real DB)
  test_cli.py        # CLI integration tests (CLI → HTTP → REST API → test DB)
scripts/
  download_models.py  # Runtime model download script
```

## Commands

### CLI (`youtube-transcription`)

The CLI runs everything directly in-process: `transcribe` executes the full ML pipeline locally (GPU required) and all commands access PostgreSQL via `DATABASE_URL`. No MCP server is needed — only the database (start with `docker compose up postgres`).

| Command            | Server tool called | Description                                        |
|--------------------|--------------------|----------------------------------------------------|
| `transcribe <url>` | `transcribe`       | Synchronous transcription (blocks until done)       |
| `list`             | `list_transcripts` | List past transcripts (rich table; `--json` for raw JSON) |
| `get <id>`         | `get_transcript`   | Retrieve a transcript by ID, written to `output/transcript.json` / `.md` (or `-o <path>`) |

### MCP Server (`youtube-transcription-server`)

| Tool               | Description                                         |
|--------------------|-----------------------------------------------------|
| `transcribe`       | Synchronous: download audio → (optionally isolate) → transcribe → align → (optionally diarize) → return results |
| `list_transcripts` | Paginated transcript listing                        |
| `get_transcript`   | Retrieve a transcript by ID (JSON or markdown)      |

### REST API (same server, same port)

Registered via `@mcp.custom_route` in `server.py`; shares logic with the MCP tools (`_transcribe_and_store`, `_list_transcripts`, `_get_transcript`).

| Endpoint                          | Description                                             |
|-----------------------------------|---------------------------------------------------------|
| `POST /api/transcribe`            | Blocking; body `{"url", "isolate"?, "diarize"?}` → transcript JSON. 400 bad URL, 422 invalid body, 502 download/metadata failure |
| `GET /api/transcripts`            | `?limit=20&offset=0` → `{transcripts, total}`           |
| `GET /api/transcripts/{id}`       | `?format=json\|markdown` (default json). 404 if missing |

## Development

```bash
# Install
uv sync

# Run CLI
uv run youtube-transcription <command> [args]

# Run MCP server
uv run youtube-transcription-server

# Test (requires Docker for PostgreSQL test container)
uv run pytest

# Lint & format
uv run ruff check .
uv run ruff format .
```

## Configuration

Settings are loaded from `.env` via pydantic-settings (`extra="ignore"` so other tools' env vars coexist):

| Variable        | Default                                                                       | Description                  |
|-----------------|-------------------------------------------------------------------------------|------------------------------|
| `DATABASE_URL`  | `postgresql+asyncpg://postgres:postgres@localhost:5432/youtube_transcription` | PostgreSQL connection string |
| `SERVER_HOST`   | `0.0.0.0`                                                                     | MCP server bind host         |
| `SERVER_PORT`   | `8000`                                                                        | MCP server bind port         |
| `WHISPER_MODEL` | `large-v3-turbo`                                                              | Whisper model name           |
| `DEVICE`        | `cuda`                                                                        | Torch device                 |
| `COMPUTE_TYPE`  | `int8`                                                                        | Whisper compute type         |
| `BATCH_SIZE`    | `8`                                                                           | Whisper batch size           |
| `HF_TOKEN`      | —                                                                             | HuggingFace token for the gated pyannote diarization model |

## Database

- PostgreSQL with async SQLAlchemy
- Tables: `transcripts`, `segments` (no `jobs` table — async jobs were removed)
- Auto-created on server startup via FastMCP lifespan (`db.init_db()`)

## Docker

The container runs as a non-root `app` user. Model weights are stored in Docker named volumes (`huggingface_cache`, `torch_cache`) rather than baked into the image, keeping the image small (~18GB vs ~48GB previously).

First-time setup after `docker compose up`:

```bash
docker compose exec server /app/.venv/bin/python scripts/download_models.py
```

This downloads Whisper, wav2vec2, Demucs, and pyannote diarization weights into the persistent volumes. It only needs to run once — the volumes survive `docker compose up`/`down`. Re-run only if you `docker compose down -v` (which destroys volumes). The pyannote model is gated: accept the terms at https://huggingface.co/pyannote/speaker-diarization-community-1 and set `HF_TOKEN` (compose passes it through).

## Conventions

- Use `uv` for all package operations
- Format and lint with `ruff` before committing
- Target `py313`, line length 120
- Ruff rules: E, F, I, UP, B, SIM, RUF
- Settings via `pydantic-settings` with `.env` file
- Heavy ML imports (`demucs`) are lazily imported inside functions to keep startup fast
- Pipeline logic lives in `pipeline.py`; `cli.py` and `server.py` are thin orchestration layers
- Comment intent, not mechanics; keep it brief
- When you need to understand a dependency's internals (output formats, API contracts), clone or inspect the source repo directly rather than relying on docs. External repos live under `/home/valec/repos/external/` (e.g. fastmcp is at `/home/valec/repos/external/fastmcp`)

### MCP tool schemas

- **Input parameters**: Use `Annotated[type, Field(description=..., ge=, le=, examples=)]` type aliases from `schemas.py` (e.g. `UrlParam`, `DiarizeParam`). These provide rich JSON Schema for LLM clients.
- **Output models**: All tools return Pydantic models from `schemas.py`. `TranscriptionResult` and `TranscriptDetailResult` inherit from `VideoMetadata` to avoid field duplication. FastMCP auto-generates `outputSchema` and `structuredContent` from these.
- **Tool annotations**: Every tool has `ToolAnnotations` with `title`, `readOnlyHint`, `destructiveHint`, `idempotentHint`, and `openWorldHint` so clients know which tools are safe vs. expensive.
- **Docstrings**: Follow the 5-rubric framework — Purpose, Guidelines, Limitations, Parameter Explanation, and brevity. LLMs rely on these to pick the right tool.

## Architecture

### Pipeline orchestration

`pipeline.run_pipeline()` is a pure synchronous function that runs the full ML pipeline (optionally isolate → transcribe → align → optionally diarize) and returns a `ProcessedResult` dataclass. Vocal isolation (`do_isolate`) and diarization (`do_diarize`) default to off; both the CLI (`--isolate`, `--diarize`) and the MCP `transcribe` tool (`isolate`, `diarize` params) can enable them. Each model is freed from GPU memory immediately after use via `free_vram()`. The function is called from `server.py` via `asyncio.to_thread()`.

Both `server.py` (MCP tools) and `cli.py` (direct local execution) own parallel async orchestration: download audio in a tempdir, call `run_pipeline()`, persist to DB, and clean up the tempdir. Neither imports ML libraries directly.

### VRAM management

ML functions (`transcribe`, `align`, `isolate_vocals`, `diarize`) return `(result, model)` tuples. The caller is responsible for calling `free_vram(model)` to release GPU memory. In `run_pipeline`, each model is freed immediately after use and the reference nulled out. `ensure_cpu()` moves any remaining CUDA tensors in result dicts to CPU before loading the next model.

The Dockerfile sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce CUDA memory fragmentation.

### FastMCP lifespan

DB initialization uses FastMCP's composable lifespan system. The `@lifespan` decorator (from `fastmcp.server.lifespan`) wraps an async generator that yields a context dict. Lifespans are composed with `|` and passed to `FastMCP()` at construction time:

```python
from fastmcp.server.lifespan import lifespan as lifespan_decorator

@lifespan_decorator
async def setup_db(server):
    await db.init_db()
    yield

mcp = FastMCP("youtube-transcription", lifespan=setup_db)
```

**Do NOT** use `@mcp.lifespan` as a decorator — `mcp.lifespan` is the `AggregateProvider.lifespan()` method, not a composable `Lifespan` object. The `|` operator only works between `Lifespan` instances created via the `@lifespan` decorator.

Source: `/home/valec/repos/external/fastmcp/fastmcp_slim/fastmcp/server/lifespan.py`

### Database injection

`db.init_engine(url=None)` initializes the SQLAlchemy engine with an explicit URL (defaults to `DATABASE_URL` from settings). `db.reset_engine()` tears it down. Tests use these to inject a test PostgreSQL URL.

### Settings caching

`settings.get_settings()` uses `@lru_cache` so the same instance is returned across calls. `settings.reset_settings()` clears the cache — tests call this in fixtures to pick up env changes.

### Pipeline flow

`download_audio(url, output_dir)` uses `yt-dlp --extract-audio --audio-format wav` to download only the audio track as WAV. WAV output avoids lossy re-encoding and is natively supported by both whisperx and demucs.

### WhisperX diarization

`diarize()` uses `whisperx.DiarizationPipeline` (pyannote `speaker-diarization-community-1` under the hood) followed by `whisperx.assign_word_speakers()`, which labels both segments and individual words. pyannote handles long-form audio natively (sliding windows with global speaker clustering), so no manual chunking or cross-chunk speaker reconciliation is needed. The pyannote model is gated on HuggingFace — a token is required (settings `hf_token` / `HF_TOKEN` env var).