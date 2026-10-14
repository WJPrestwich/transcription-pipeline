from collections.abc import AsyncGenerator

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from youtube_transcription.models import Base, Segment, Transcript
from youtube_transcription.settings import get_settings

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None

_POOL_SIZE = 5


def init_engine(url: str | None = None) -> None:
    global _engine, _session_factory
    if url is None:
        url = get_settings().database_url
    _engine = create_async_engine(url, echo=False, pool_size=_POOL_SIZE, max_overflow=10)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)


def reset_engine() -> None:
    global _engine, _session_factory
    _engine = None
    _session_factory = None


def _get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        init_engine()
    return _engine


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        init_engine()
    return _session_factory


async def init_db() -> None:
    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession]:
    factory = _get_session_factory()
    async with factory() as session:
        yield session


async def create_transcript_with_segments(
    video_url: str,
    video_id: str,
    video_title: str | None,
    channel: str | None,
    duration_seconds: float | None,
    view_count: int | None,
    language: str | None,
    has_diarization: bool,
    speaker_count: int | None,
    segments: list[dict],
) -> Transcript:
    factory = _get_session_factory()
    async with factory() as session:
        transcript = Transcript(
            video_url=video_url,
            video_id=video_id,
            video_title=video_title,
            channel=channel,
            duration_seconds=duration_seconds,
            view_count=view_count,
            language=language,
            has_diarization=has_diarization,
            speaker_count=speaker_count,
        )
        session.add(transcript)
        await session.flush()

        for i, seg in enumerate(segments):
            segment = Segment(
                transcript_id=transcript.id,
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                speaker=seg.get("speaker"),
                words=seg.get("words"),
                sort_order=i,
            )
            session.add(segment)

        await session.commit()
        await session.refresh(transcript)
        return transcript


async def list_transcripts(limit: int = 20, offset: int = 0) -> tuple[list[Transcript], int]:
    factory = _get_session_factory()
    async with factory() as session:
        total = (await session.execute(select(func.count()).select_from(Transcript))).scalar_one()
        result = await session.execute(
            select(Transcript).order_by(Transcript.created_at.desc()).limit(limit).offset(offset)
        )
        return list(result.scalars().all()), total


async def get_transcript(transcript_id: str) -> Transcript | None:
    factory = _get_session_factory()
    async with factory() as session:
        return await session.get(Transcript, transcript_id)


async def get_segments(transcript_id: str) -> list[Segment]:
    factory = _get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(Segment).where(Segment.transcript_id == transcript_id).order_by(Segment.sort_order)
        )
        return list(result.scalars().all())
