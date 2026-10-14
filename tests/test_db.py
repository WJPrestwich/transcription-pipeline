import uuid

import pytest

from youtube_transcription import db


@pytest.mark.asyncio
class TestTranscriptWithSegments:
    async def test_creates_transcript_with_segments(self, db_session):
        segments = [
            {"start": 0.0, "end": 5.0, "text": "hello", "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 10.0, "text": "world"},
        ]
        transcript = await db.create_transcript_with_segments(
            video_url="https://youtube.com/watch?v=abc",
            video_id="abc",
            video_title="Test",
            channel="Chan",
            duration_seconds=60.0,
            view_count=100,
            language="en",
            has_diarization=True,
            speaker_count=2,
            segments=segments,
        )
        assert transcript.id is not None
        assert transcript.video_id == "abc"
        assert transcript.has_diarization is True
        assert transcript.speaker_count == 2

        db_segments = await db.get_segments(str(transcript.id))
        assert len(db_segments) == 2
        assert db_segments[0].text == "hello"
        assert db_segments[0].speaker == "SPEAKER_00"
        assert db_segments[1].speaker is None
        assert db_segments[0].sort_order == 0
        assert db_segments[1].sort_order == 1

    async def test_segments_ordered_by_sort_order(self, db_session):
        segments = [
            {"start": 5.0, "end": 10.0, "text": "second"},
            {"start": 0.0, "end": 5.0, "text": "first"},
        ]
        transcript = await db.create_transcript_with_segments(
            video_url="https://youtube.com/watch?v=abc",
            video_id="abc",
            video_title="Test",
            channel=None,
            duration_seconds=None,
            view_count=None,
            language="en",
            has_diarization=False,
            speaker_count=None,
            segments=segments,
        )
        db_segments = await db.get_segments(str(transcript.id))
        assert db_segments[0].text == "second"
        assert db_segments[1].text == "first"


@pytest.mark.asyncio
class TestListTranscripts:
    async def test_pagination(self, db_session):
        for i in range(5):
            await db.create_transcript_with_segments(
                video_url=f"https://youtube.com/watch?v=vid{i}",
                video_id=f"vid{i}",
                video_title=f"Video {i}",
                channel=None,
                duration_seconds=None,
                view_count=None,
                language="en",
                has_diarization=False,
                speaker_count=None,
                segments=[],
            )
        page1, total = await db.list_transcripts(limit=2, offset=0)
        assert total == 5
        assert len(page1) == 2

        page2, _ = await db.list_transcripts(limit=2, offset=2)
        assert len(page2) == 2


@pytest.mark.asyncio
class TestGetTranscript:
    async def test_returns_transcript(self, db_session):
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
            segments=[],
        )
        found = await db.get_transcript(str(created.id))
        assert found is not None
        assert found.video_id == "abc"

    async def test_returns_none_for_missing(self, db_session):
        result = await db.get_transcript(str(uuid.uuid4()))
        assert result is None
