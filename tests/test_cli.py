"""Integration tests: CLI (click) → HTTP → REST API → test Postgres.

A real uvicorn server runs as a task on the test's event loop, serving the
FastMCP app (MCP endpoint + REST routes) against the testcontainer Postgres
(the db_session fixture injects the test engine before the app starts, so the
app's lifespan init_db() uses the test database). The sync click CLI is driven
via asyncio.to_thread so it doesn't block the loop serving the requests.
Heavy pipeline functions are patched in the server module.
"""

import asyncio
import json
from functools import partial
from pathlib import Path
from unittest.mock import patch

import anyio
import pytest
import pytest_asyncio
import uvicorn
from click.testing import CliRunner

from youtube_transcription import db
from youtube_transcription.cli import cli
from youtube_transcription.pipeline import ProcessedResult, VideoMetadata


@pytest_asyncio.fixture
async def api_server(db_session):
    """Serve the app over HTTP on an ephemeral port; yield its base URL."""
    from youtube_transcription.server import mcp

    server = uvicorn.Server(uvicorn.Config(mcp.http_app(), host="127.0.0.1", port=0, log_level="warning"))
    task = asyncio.create_task(server.serve())
    for _ in range(100):
        if server.started:
            break
        await asyncio.sleep(0.05)
    else:
        raise RuntimeError("test API server did not start")
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    await asyncio.wait_for(task, timeout=5)


async def _invoke(args: list[str], api_url: str):
    """Run the sync CLI in a worker thread so the event loop stays free to serve it.

    Uses anyio.to_thread rather than asyncio.to_thread — tests patch
    asyncio.to_thread (in the server module's namespace) with a synchronous
    side_effect, which would deadlock a sync CLI call against the running loop.
    """
    runner = CliRunner()
    # COLUMNS keeps rich table output untruncated under CliRunner's captured terminal
    env = {"API_URL": api_url, "COLUMNS": "200"}
    return await anyio.to_thread.run_sync(partial(runner.invoke, cli, args, env=env))


def _seed_transcript(**overrides):
    defaults = {
        "video_url": "https://youtube.com/watch?v=abc",
        "video_id": "abc",
        "video_title": "Test Video",
        "channel": "Chan",
        "duration_seconds": 60.0,
        "view_count": 100,
        "language": "en",
        "has_diarization": True,
        "speaker_count": 1,
        "segments": [{"start": 0.0, "end": 5.0, "text": "hello", "speaker": "SPEAKER_00"}],
    }
    return db.create_transcript_with_segments(**(defaults | overrides))


@pytest.mark.asyncio
class TestCliList:
    async def test_list_json(self, api_server):
        await _seed_transcript()
        result = await _invoke(["list", "--json"], api_server)
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["total"] == 1
        assert data["transcripts"][0]["video_id"] == "abc"

    async def test_list_table(self, api_server):
        await _seed_transcript()
        result = await _invoke(["list"], api_server)
        assert result.exit_code == 0, result.output
        assert "Test Video" in result.output

    async def test_list_empty(self, api_server):
        result = await _invoke(["list"], api_server)
        assert result.exit_code == 0, result.output
        assert "No transcripts found." in result.output


@pytest.mark.asyncio
class TestCliGet:
    async def test_get_json(self, api_server, tmp_path):
        created = await _seed_transcript()
        out = tmp_path / "t.json"
        result = await _invoke(["get", str(created.id), "-o", str(out)], api_server)
        assert result.exit_code == 0, result.output
        data = json.loads(out.read_text())
        assert data["video_id"] == "abc"
        assert data["segments"][0]["speaker"] == "SPEAKER_00"

    async def test_get_markdown(self, api_server, tmp_path):
        created = await _seed_transcript()
        out = tmp_path / "t.md"
        result = await _invoke(["get", str(created.id), "--format", "markdown", "-o", str(out)], api_server)
        assert result.exit_code == 0, result.output
        assert "**SPEAKER_00** hello" in out.read_text()

    async def test_get_not_found_exits_1(self, api_server):
        result = await _invoke(["get", "00000000-0000-0000-0000-000000000000"], api_server)
        assert result.exit_code == 1
        assert "not found" in result.output


@pytest.mark.asyncio
class TestCliTranscribe:
    @patch("youtube_transcription.server.pipeline.download_audio")
    @patch("youtube_transcription.server.pipeline.get_video_metadata")
    @patch("youtube_transcription.server.pipeline.run_pipeline")
    @patch("youtube_transcription.server.asyncio.to_thread", side_effect=lambda f, *a, **kw: f(*a, **kw))
    async def test_transcribe_passes_flags_to_pipeline(
        self, mock_to_thread, mock_run_pipeline, mock_metadata, mock_download, api_server
    ):
        mock_metadata.return_value = VideoMetadata(
            video_id="dQw4w9WgXcQ", title="Test", channel="Chan", duration_seconds=60.0, view_count=100
        )
        mock_download.return_value = Path("/tmp/audio.wav")
        mock_run_pipeline.return_value = ProcessedResult(
            language="en",
            segments=[{"start": 0.0, "end": 5.0, "text": "hello", "speaker": "SPEAKER_00"}],
            has_diarization=True,
            speaker_count=1,
        )

        result = await _invoke(
            ["transcribe", "https://youtube.com/watch?v=dQw4w9WgXcQ", "--isolate", "--diarize"],
            api_server,
        )

        assert result.exit_code == 0, result.output
        body = json.loads(result.output[result.output.index("{") :])
        assert body["video_id"] == "dQw4w9WgXcQ"
        assert body["has_diarization"] is True
        _, kwargs = mock_run_pipeline.call_args
        assert kwargs["do_isolate"] is True
        assert kwargs["do_diarize"] is True

    async def test_transcribe_unsupported_url_exits_1(self, api_server):
        result = await _invoke(["transcribe", "https://vimeo.com/123456"], api_server)
        assert result.exit_code == 1
        assert "Unsupported URL" in result.output


@pytest.mark.asyncio
class TestCliServerDown:
    async def test_unreachable_server_exits_1(self):
        # Nothing listens on port 1
        result = await _invoke(["list"], "http://127.0.0.1:1")
        assert result.exit_code == 1
        assert "Cannot reach API" in result.output
