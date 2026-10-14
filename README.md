# youtube-transcription

CLI + MCP server for downloading YouTube and Twitch audio (Twitch clips and VODs), isolating vocals with Demucs, transcribing and diarizing with WhisperX (pyannote diarization). Transcripts are persisted to PostgreSQL.

## Development

```bash
# Install
uv sync

# Run CLI
uv run youtube-transcription <command> [args]

# Test (requires Docker for PostgreSQL test container)
uv run pytest

# Lint & format
uv run ruff check .
uv run ruff format .
```

### Starting the MCP server

The server requires a running PostgreSQL database. Start both with Docker Compose:

```bash
docker compose up
```

The server listens on `http://localhost:8000`. Tables are auto-created on startup via `db.init_db()`.

### Pre-downloading model weights

Model weights (Whisper, wav2vec2, Demucs, pyannote diarization) are stored in Docker named volumes and persist across container rebuilds. On first run, download them inside the running container:

```bash
docker compose exec server /app/.venv/bin/python scripts/download_models.py
```

The pyannote diarization model is gated on HuggingFace — accept the terms at https://huggingface.co/pyannote/speaker-diarization-community-1 and set `HF_TOKEN` in your environment (compose passes it through to the container).

This only needs to be done once — the volumes `huggingface_cache` and `torch_cache` retain the data across `docker compose up`/`down` cycles. You only need to re-run it if you `docker compose down -v` (which destroys volumes).

## CLI Commands

The CLI runs the pipeline and database access directly — no MCP server required. `transcribe` runs the full ML pipeline locally (GPU required); all commands need PostgreSQL reachable at `DATABASE_URL` (start it with `docker compose up postgres`).

| Command                  | Description                                                 |
|--------------------------|-------------------------------------------------------------|
| `transcribe <url>`       | Synchronous transcription (blocks until done)               |
| `transcribe-async <url>` | Start async job, poll until done                            |
| `status <job_id>`        | Check status of an async job                                |
| `list`                   | List past transcripts                                       |
| `get <id>`               | Retrieve a transcript by ID (`--format json` or `markdown`) |

## MCP Server Tools

| Tool               | Description                                         |
|--------------------|-----------------------------------------------------|
| `transcribe`       | Synchronous: download audio → transcribe → return results (`isolate`, `diarize` optional flags) |
| `list_transcripts` | Paginated transcript listing                        |
| `get_transcript`   | Retrieve a transcript by ID (JSON or markdown)      |

## REST API

The same server also exposes a REST API on the same port (implemented via FastMCP custom routes):

| Endpoint                    | Description                                                                  |
|-----------------------------|------------------------------------------------------------------------------|
| `POST /api/transcribe`      | Blocking; body `{"url", "isolate"?, "diarize"?}` → transcript JSON. 400 bad URL, 422 invalid body, 502 download failure |
| `GET /api/transcripts`      | `?limit=20&offset=0` → `{transcripts, total}`                                |
| `GET /api/transcripts/{id}` | `?format=json\|markdown` (default json). 404 if missing                      |

## Configuration

Settings are loaded from `.env` via pydantic-settings. Extra env vars are ignored, so other tools' config (e.g. `PARASAIL_API_KEY`) can coexist in the same `.env`.

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

## Connecting to Hermes Agent

The MCP server runs over HTTP (Streamable HTTP transport). Start it with `docker compose up` (see above), then add an entry to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  youtube-transcription:
    url: "http://localhost:8000/mcp/"
    timeout: 300    # transcription can take several minutes
```

Then restart Hermes or run `/reload-mcp` in a session. The three tools (`transcribe`, `list_transcripts`, `get_transcript`) will appear prefixed with `mcp_youtube_transcription_`.

If the server is running on a remote host or a different port, update the `url` accordingly. If you use authentication or a reverse proxy, add `headers` for bearer tokens:

```yaml
mcp_servers:
  youtube-transcription:
    url: "https://your-server.example.com/mcp/"
    headers:
      Authorization: "Bearer ${YOUTUBE_TRANSCRIPTION_API_KEY}"
    timeout: 300
```

After adding the config, start a new Hermes session or reload MCP servers. Verify the connection with:

```
hermes mcp list
```

See the [Hermes Agent MCP docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp) for full configuration reference (stdio transport, `include`/`exclude` tool filtering, TLS, etc.).

## Database

- PostgreSQL with async SQLAlchemy
- Tables: `jobs` (async pipeline jobs), `transcripts`, `segments`
- Auto-created on server startup via FastMCP lifespan (`db.init_db()`)

## Project Structure

```
src/youtube_transcription/
  __init__.py
  cli.py        # CLI commands (runs pipeline + DB access directly, no MCP server)
  pipeline.py   # Core pipeline functions (download_audio, isolate_vocals, transcribe, etc.)
  server.py      # FastMCP server (MCP tools + REST API via custom routes)
  runner.py      # Background pipeline runner for async jobs
  db.py          # SQLAlchemy async database operations
  models.py      # SQLAlchemy ORM models (Job, Transcript, Segment)
  settings.py    # Pydantic settings (env-based config)
tests/
  conftest.py       # Docker PostgreSQL fixture, settings reset
  test_pipeline.py  # Pipeline unit tests (diarize, subprocess wrappers, pure functions)
  test_db.py         # Database CRUD tests (real PostgreSQL)
  test_server.py     # MCP tool + REST route tests (mocked pipeline, real DB)
  test_cli.py        # CLI integration tests (CLI → HTTP → REST API → test DB)
  test_runner.py      # Background runner tests (mocked pipeline, real DB)
```

## Docker

The container runs as a non-root `app` user. Model weights are stored in Docker named volumes (`huggingface_cache`, `torch_cache`) — not baked into the image. This keeps the image small and avoids re-downloading models on rebuild.

The image is based on plain `ubuntu:24.04` and includes Python 3.13 (via uv), ffmpeg, and yt-dlp. The pip `nvidia-*-cu12` wheels in the venv already bundle all CUDA userland libraries, so the CUDA toolkit base image is unnecessary — only the host's NVIDIA driver is needed, which nvidia-container-toolkit injects at runtime. GPU access is configured via `deploy.resources.reservations.devices` in `docker-compose.yml`.

## Testing

Tests require Docker for the PostgreSQL test container. The test suite:

- **Pipeline tests** — pure logic (`format_timestamp`, `result_to_dict`, etc.), mocked `diarize`, and subprocess wrappers (`download_audio`, `get_video_metadata`) via `unittest.mock`
- **DB tests** — real PostgreSQL via testcontainers; tests CRUD for jobs, transcripts, segments, and pagination
- **Server tests** — mock all ML/subprocess calls, test MCP tool orchestration end-to-end against real DB
- **Runner tests** — mock the pipeline, test async job lifecycle (queued → downloading → … → completed/failed) against real DB

Key test utilities:

- `db.init_engine(url)` / `db.reset_engine()` — inject/reset test DB connection
- `settings.reset_settings()` — clear cached settings between tests
- `conftest.py` verifies Docker is available and fails fast if not