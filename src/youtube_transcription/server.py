import asyncio
import json
import logging
import shutil
import tempfile
from pathlib import Path

from fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response

from youtube_transcription import db, pipeline
from youtube_transcription.models import Segment, Transcript
from youtube_transcription.schemas import (
    DiarizeParam,
    IsolateParam,
    LimitParam,
    OffsetParam,
    OutputFormatParam,
    TranscribeRequest,
    TranscriptDetailResult,
    TranscriptIdParam,
    TranscriptionResult,
    TranscriptListResult,
    TranscriptSegmentOut,
    TranscriptSummaryOut,
    UrlParam,
)
from youtube_transcription.settings import get_settings

logger = logging.getLogger(__name__)


def _db_lifespan():
    from fastmcp.server.lifespan import lifespan as lifespan_decorator

    @lifespan_decorator
    async def setup_db(server):
        await db.init_db()
        yield

    return setup_db


mcp = FastMCP("youtube-transcription", lifespan=_db_lifespan())


def _segments_from_orm(segments: list[Segment]) -> list[TranscriptSegmentOut]:
    return [TranscriptSegmentOut(start=seg.start, end=seg.end, text=seg.text, speaker=seg.speaker) for seg in segments]


def _transcription_result_from_orm(transcript: Transcript, segments: list[Segment]) -> TranscriptionResult:
    return TranscriptionResult(
        id=str(transcript.id),
        video_url=transcript.video_url,
        video_id=transcript.video_id,
        video_title=transcript.video_title,
        channel=transcript.channel,
        duration_seconds=transcript.duration_seconds,
        view_count=transcript.view_count,
        language=transcript.language,
        has_diarization=transcript.has_diarization,
        speaker_count=transcript.speaker_count,
        created_at=transcript.created_at.isoformat() if transcript.created_at else None,
        segments=_segments_from_orm(segments),
    )


def _segments_to_markdown(segments: list[Segment]) -> str:
    lines = []
    for seg in segments:
        start = pipeline.format_timestamp(seg.start)
        end = pipeline.format_timestamp(seg.end)
        header = f"**{seg.speaker}** " if seg.speaker else ""
        lines.append(f"[{start} - {end}] {header}{seg.text}")
    return "\n".join(lines)


async def _transcribe_and_store(url: str, isolate: bool, diarize: bool) -> TranscriptionResult:
    """Download, transcribe, and persist a video's transcript. Shared by the MCP tool and REST route."""
    settings = get_settings()
    pipeline.validate_url(url)
    logger.info("transcribe: url=%s isolate=%s diarize=%s", url, isolate, diarize)

    metadata = await asyncio.to_thread(pipeline.get_video_metadata, url)
    if metadata is None:
        logger.error("transcribe: failed to retrieve metadata for %s", url)
        raise RuntimeError(f"Could not retrieve metadata for {url}")
    video_id = metadata.video_id
    logger.info(
        "transcribe: metadata id=%s title=%s duration=%ss",
        metadata.video_id,
        metadata.title,
        metadata.duration_seconds,
    )

    tmpdir = tempfile.mkdtemp()
    logger.info("transcribe: tmpdir=%s", tmpdir)

    try:
        audio = await asyncio.to_thread(pipeline.download_audio, url, Path(tmpdir))
        if audio is None:
            logger.error("transcribe: download failed for %s", url)
            raise RuntimeError("Failed to download audio")
        logger.info("transcribe: downloaded audio %s", audio.name)

        result = await asyncio.to_thread(
            pipeline.run_pipeline,
            audio,
            Path(tmpdir),
            settings.device,
            settings.whisper_model,
            settings.compute_type,
            settings.batch_size,
            do_diarize=diarize,
            do_isolate=isolate,
            hf_token=settings.hf_token,
        )
        logger.info(
            "transcribe: pipeline completed — language=%s segments=%d speakers=%d",
            result.language,
            len(result.segments),
            result.speaker_count or 0,
        )

        transcript = await db.create_transcript_with_segments(
            video_url=url,
            video_id=video_id,
            video_title=metadata.title,
            channel=metadata.channel,
            duration_seconds=metadata.duration_seconds,
            view_count=metadata.view_count,
            language=result.language,
            has_diarization=result.has_diarization,
            speaker_count=result.speaker_count,
            segments=result.segments,
        )
        transcript_segments = await db.get_segments(str(transcript.id))
        return _transcription_result_from_orm(transcript, transcript_segments)

    finally:
        logger.info("transcribe: cleaning up tmpdir=%s", tmpdir)
        shutil.rmtree(tmpdir, ignore_errors=True)


@mcp.tool(
    annotations=ToolAnnotations(
        title="Transcribe Video",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    )
)
async def transcribe(
    url: UrlParam, isolate: IsolateParam = False, diarize: DiarizeParam = False
) -> TranscriptionResult:
    """Transcribe a YouTube video or Twitch clip/VOD.

    Downloads the audio track and transcribes speech using whisperx. Set
    diarize=true to identify speakers with WhisperX's pyannote pipeline, and/or
    isolate=true to run Demucs vocal isolation first — only worth it for
    music-heavy audio.

    This is a blocking call — it may take several minutes for long videos.

    Limitations:
    - Only supports YouTube URLs (youtu.be, youtube.com/watch, /shorts/, /embed/)
      and Twitch URLs (clips and VODs)
    - Very long videos (>2h) may exceed GPU memory
    """
    return await _transcribe_and_store(url, isolate, diarize)


async def _list_transcripts(limit: int, offset: int) -> TranscriptListResult:
    transcripts, total = await db.list_transcripts(limit=limit, offset=offset)
    return TranscriptListResult(
        transcripts=[
            TranscriptSummaryOut(
                id=str(t.id),
                video_url=t.video_url,
                video_id=t.video_id,
                video_title=t.video_title,
                channel=t.channel,
                duration_seconds=t.duration_seconds,
                view_count=t.view_count,
                language=t.language,
                has_diarization=t.has_diarization,
                speaker_count=t.speaker_count,
                created_at=t.created_at.isoformat() if t.created_at else None,
            )
            for t in transcripts
        ],
        total=total,
    )


def _rest_error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse({"detail": message}, status_code=status_code)


@mcp.custom_route("/api/transcribe", methods=["POST"])
async def rest_transcribe(request: Request) -> Response:
    """Blocking transcription: POST {"url": ..., "isolate": false, "diarize": false} → transcript JSON."""
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _rest_error(400, "Invalid JSON body")

    try:
        req = TranscribeRequest.model_validate(body)
    except ValidationError as e:
        return JSONResponse({"detail": e.errors(include_context=False, include_url=False)}, status_code=422)

    try:
        result = await _transcribe_and_store(req.url, req.isolate, req.diarize)
    except ValueError as e:  # unsupported URL
        return _rest_error(400, str(e))
    except RuntimeError as e:  # upstream download/metadata failure
        return _rest_error(502, str(e))
    return JSONResponse(result.model_dump(mode="json"))


@mcp.custom_route("/api/transcripts", methods=["GET"])
async def rest_list_transcripts(request: Request) -> Response:
    """Paginated transcript listing: GET /api/transcripts?limit=20&offset=0."""
    try:
        limit = int(request.query_params.get("limit", 20))
        offset = int(request.query_params.get("offset", 0))
    except ValueError:
        return _rest_error(400, "limit and offset must be integers")
    if not (1 <= limit <= 100) or offset < 0:
        return _rest_error(400, "limit must be 1-100 and offset >= 0")

    result = await _list_transcripts(limit, offset)
    return JSONResponse(result.model_dump(mode="json"))


@mcp.custom_route("/api/transcripts/{transcript_id}", methods=["GET"])
async def rest_get_transcript(request: Request) -> Response:
    """Retrieve a transcript: GET /api/transcripts/{id}?format=json|markdown."""
    transcript_id = request.path_params["transcript_id"]
    output_format = request.query_params.get("format", "json")
    if output_format not in ("json", "markdown"):
        return _rest_error(400, "format must be 'json' or 'markdown'")

    try:
        result = await _get_transcript(transcript_id, output_format)
    except ValueError as e:
        return _rest_error(404, str(e))

    if output_format == "markdown":
        return PlainTextResponse(result)
    return JSONResponse(result.model_dump(mode="json"))


@mcp.tool(
    annotations=ToolAnnotations(
        title="List Transcripts",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
async def list_transcripts(limit: LimitParam = 20, offset: OffsetParam = 0) -> TranscriptListResult:
    """List past transcripts with pagination.

    Returns a page of transcript summaries (without segment details).
    Use get_transcript to retrieve the full transcript with segments.
    """
    return await _list_transcripts(limit, offset)


async def _get_transcript(transcript_id: str, output_format: str) -> TranscriptDetailResult | str:
    transcript = await db.get_transcript(transcript_id)
    if transcript is None:
        raise ValueError(f"Transcript {transcript_id} not found")

    segments = await db.get_segments(transcript_id)

    if output_format == "markdown":
        return _segments_to_markdown(segments)

    return TranscriptDetailResult(
        id=str(transcript.id),
        video_url=transcript.video_url,
        video_id=transcript.video_id,
        video_title=transcript.video_title,
        channel=transcript.channel,
        duration_seconds=transcript.duration_seconds,
        view_count=transcript.view_count,
        language=transcript.language,
        has_diarization=transcript.has_diarization,
        speaker_count=transcript.speaker_count,
        created_at=transcript.created_at.isoformat() if transcript.created_at else None,
        segments=_segments_from_orm(segments),
    )


@mcp.tool(
    annotations=ToolAnnotations(
        title="Get Transcript",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
async def get_transcript(
    id: TranscriptIdParam, output_format: OutputFormatParam = "json"
) -> TranscriptDetailResult | str:
    """Retrieve a full transcript by ID.

    Use output_format='json' for structured data with segments, or 'markdown' for human-readable text.
    Raises ValueError if the transcript ID is not found.
    """
    return await _get_transcript(id, output_format)


def main() -> None:
    import copy
    import logging.config

    from uvicorn.config import LOGGING_CONFIG

    settings = get_settings()
    config = copy.deepcopy(LOGGING_CONFIG)
    config["loggers"]["youtube_transcription"] = {
        "handlers": ["default"],
        "level": "INFO",
        "propagate": False,
    }
    config["formatters"]["default"]["fmt"] = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.config.dictConfig(config)

    mcp.run(transport="http", host=settings.server_host, port=settings.server_port)


if __name__ == "__main__":
    main()
