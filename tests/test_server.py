import uuid
from unittest.mock import patch

import pytest

from youtube_transcription import db
from youtube_transcription.pipeline import ProcessedResult


@pytest.mark.asyncio
class TestTranscribeTool:
    @patch("youtube_transcription.server.pipeline.download_audio")
    @patch("youtube_transcription.server.pipeline.get_video_metadata")
    @patch("youtube_transcription.server.pipeline.run_pipeline")
    @patch("youtube_transcription.server.asyncio.to_thread", side_effect=lambda f, *a, **kw: f(*a, **kw))
    async def test_full_pipeline_with_diarization(
        self,
        mock_to_thread,
        mock_run_pipeline,
        mock_metadata,
        mock_download,
        db_session,
    ):
        from pathlib import Path

        from youtube_transcription.pipeline import VideoMetadata
        from youtube_transcription.server import transcribe

        mock_metadata.return_value = VideoMetadata(
            video_id="dQw4w9WgXcQ", title="Test", channel="Chan", duration_seconds=60.0, view_count=100
        )
        mock_download.return_value = Path("/tmp/audio.wav")
        mock_run_pipeline.return_value = ProcessedResult(
            language="en",
            segments=[
                {"start": 0.0, "end": 5.0, "text": "hello", "speaker": "SPEAKER_00"},
                {"start": 5.0, "end": 10.0, "text": "world"},
            ],
            has_diarization=True,
            speaker_count=1,
        )

        result = await transcribe(url="https://youtube.com/watch?v=dQw4w9WgXcQ")

        assert result.id is not None
        assert result.video_id == "dQw4w9WgXcQ"
        assert result.video_title == "Test"
        assert result.has_diarization is True
        assert result.language == "en"
        assert len(result.segments) == 2
        assert result.segments[0].speaker == "SPEAKER_00"

    @patch("youtube_transcription.server.pipeline.download_audio")
    @patch("youtube_transcription.server.pipeline.get_video_metadata")
    @patch("youtube_transcription.server.pipeline.run_pipeline")
    @patch("youtube_transcription.server.asyncio.to_thread", side_effect=lambda f, *a, **kw: f(*a, **kw))
    async def test_twitch_clip_uses_ytdlp_canonical_id(
        self,
        mock_to_thread,
        mock_run_pipeline,
        mock_metadata,
        mock_download,
        db_session,
    ):
        from pathlib import Path

        from youtube_transcription.pipeline import VideoMetadata
        from youtube_transcription.server import transcribe

        # yt-dlp reports the Twitch clip's numeric ID, not the URL slug
        mock_metadata.return_value = VideoMetadata(
            video_id="4270082645", title="Clip title", channel="ninoegyn", duration_seconds=20.0, view_count=55
        )
        mock_download.return_value = Path("/tmp/audio.wav")
        mock_run_pipeline.return_value = ProcessedResult(
            language="en",
            segments=[{"start": 0.0, "end": 5.0, "text": "hello"}],
            has_diarization=True,
            speaker_count=1,
        )

        result = await transcribe(
            url="https://www.twitch.tv/ninoegyn/clip/WanderingGorgeousTireMingLee-udaOxVOvNij79pZS"
        )

        assert result.video_id == "4270082645"
        assert result.channel == "ninoegyn"

    async def test_rejects_unsupported_url(self, db_session):
        from youtube_transcription.server import transcribe

        with pytest.raises(ValueError, match="Unsupported URL"):
            await transcribe(url="https://vimeo.com/123456")


@pytest.mark.asyncio
class TestListTranscriptsTool:
    async def test_returns_paginated_list(self, db_session):
        from youtube_transcription.server import list_transcripts

        await db.create_transcript_with_segments(
            video_url="https://youtube.com/watch?v=abc",
            video_id="abc",
            video_title="Test",
            channel=None,
            duration_seconds=None,
            view_count=None,
            language="en",
            has_diarization=False,
            speaker_count=None,
            segments=[],
        )
        result = await list_transcripts(limit=10, offset=0)
        assert result.total == 1
        assert len(result.transcripts) == 1


@pytest.mark.asyncio
class TestGetTranscriptTool:
    async def test_returns_transcript_json(self, db_session):
        from youtube_transcription.server import get_transcript

        created = await db.create_transcript_with_segments(
            video_url="https://youtube.com/watch?v=abc",
            video_id="abc",
            video_title="Test",
            channel="Chan",
            duration_seconds=60.0,
            view_count=100,
            language="en",
            has_diarization=True,
            speaker_count=2,
            segments=[
                {"start": 0.0, "end": 5.0, "text": "hello", "speaker": "SPEAKER_00"},
                {"start": 5.0, "end": 10.0, "text": "world"},
            ],
        )
        result = await get_transcript(str(created.id), output_format="json")
        assert result.video_id == "abc"
        assert len(result.segments) == 2

    async def test_returns_transcript_markdown(self, db_session):
        from youtube_transcription.server import get_transcript

        created = await db.create_transcript_with_segments(
            video_url="https://youtube.com/watch?v=abc",
            video_id="abc",
            video_title="Test",
            channel=None,
            duration_seconds=None,
            view_count=None,
            language="en",
            has_diarization=False,
            speaker_count=None,
            segments=[{"start": 0.0, "end": 5.0, "text": "hello"}],
        )
        result = await get_transcript(str(created.id), output_format="markdown")
        assert "[00:00:00 - 00:00:05] hello" in result

    async def test_raises_on_missing_transcript(self, db_session):
        from youtube_transcription.server import get_transcript

        with pytest.raises(ValueError, match="not found"):
            await get_transcript(str(uuid.uuid4()))


def _rest_client():
    import httpx

    from youtube_transcription.server import mcp

    transport = httpx.ASGITransport(app=mcp.http_app())
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
class TestRestApi:
    @patch("youtube_transcription.server.pipeline.download_audio")
    @patch("youtube_transcription.server.pipeline.get_video_metadata")
    @patch("youtube_transcription.server.pipeline.run_pipeline")
    @patch("youtube_transcription.server.asyncio.to_thread", side_effect=lambda f, *a, **kw: f(*a, **kw))
    async def test_post_transcribe(self, mock_to_thread, mock_run_pipeline, mock_metadata, mock_download, db_session):
        from pathlib import Path

        from youtube_transcription.pipeline import VideoMetadata

        mock_metadata.return_value = VideoMetadata(
            video_id="dQw4w9WgXcQ", title="Test", channel="Chan", duration_seconds=60.0, view_count=100
        )
        mock_download.return_value = Path("/tmp/audio.wav")
        mock_run_pipeline.return_value = ProcessedResult(
            language="en",
            segments=[{"start": 0.0, "end": 5.0, "text": "hello"}],
            has_diarization=False,
            speaker_count=None,
        )

        async with _rest_client() as client:
            response = await client.post("/api/transcribe", json={"url": "https://youtube.com/watch?v=dQw4w9WgXcQ"})

        assert response.status_code == 200
        body = response.json()
        assert body["video_id"] == "dQw4w9WgXcQ"
        assert body["has_diarization"] is False
        assert len(body["segments"]) == 1
        _, kwargs = mock_run_pipeline.call_args
        assert kwargs["do_isolate"] is False
        assert kwargs["do_diarize"] is False

    async def test_post_transcribe_rejects_unsupported_url(self, db_session):
        async with _rest_client() as client:
            response = await client.post("/api/transcribe", json={"url": "https://vimeo.com/123456"})
        assert response.status_code == 400

    async def test_post_transcribe_rejects_missing_url(self, db_session):
        async with _rest_client() as client:
            response = await client.post("/api/transcribe", json={})
        assert response.status_code == 422

    async def test_list_transcripts(self, db_session):
        await db.create_transcript_with_segments(
            video_url="https://youtube.com/watch?v=abc",
            video_id="abc",
            video_title="Test",
            channel=None,
            duration_seconds=None,
            view_count=None,
            language="en",
            has_diarization=False,
            speaker_count=None,
            segments=[],
        )
        async with _rest_client() as client:
            response = await client.get("/api/transcripts")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert body["transcripts"][0]["video_id"] == "abc"

    async def test_get_transcript_json_and_markdown(self, db_session):
        created = await db.create_transcript_with_segments(
            video_url="https://youtube.com/watch?v=abc",
            video_id="abc",
            video_title="Test",
            channel="Chan",
            duration_seconds=60.0,
            view_count=100,
            language="en",
            has_diarization=True,
            speaker_count=1,
            segments=[{"start": 0.0, "end": 5.0, "text": "hello", "speaker": "SPEAKER_00"}],
        )
        async with _rest_client() as client:
            json_response = await client.get(f"/api/transcripts/{created.id}")
            md_response = await client.get(f"/api/transcripts/{created.id}", params={"format": "markdown"})

        assert json_response.status_code == 200
        assert json_response.json()["segments"][0]["speaker"] == "SPEAKER_00"
        assert md_response.status_code == 200
        assert "**SPEAKER_00**" in md_response.text

    async def test_get_transcript_not_found(self, db_session):
        async with _rest_client() as client:
            response = await client.get(f"/api/transcripts/{uuid.uuid4()}")
        assert response.status_code == 404
