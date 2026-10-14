from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class VideoMetadata(BaseModel):
    """Shared video metadata fields across all transcript-related responses."""

    model_config = ConfigDict(from_attributes=True)

    video_url: str = Field(description="Original video URL (YouTube or Twitch)")
    video_id: str = Field(description="Canonical platform video ID (11-character YouTube ID or numeric Twitch ID)")
    video_title: str | None = Field(default=None, description="Video title from platform metadata")
    channel: str | None = Field(default=None, description="Channel name from platform metadata")
    duration_seconds: float | None = Field(default=None, description="Video duration in seconds")
    view_count: int | None = Field(default=None, description="View count at time of transcription")


class TranscriptSegmentOut(BaseModel):
    start: float = Field(description="Segment start time in seconds")
    end: float = Field(description="Segment end time in seconds")
    text: str = Field(description="Transcribed text for this segment")
    speaker: str | None = Field(
        default=None, description="Speaker label, e.g. 'SPEAKER_00'. Null if diarization was not performed."
    )


class TranscriptionResult(VideoMetadata):
    id: str = Field(description="Transcript UUID")
    language: str | None = Field(default=None, description="Detected language code, e.g. 'en'")
    has_diarization: bool = Field(description="Whether speaker diarization was performed")
    speaker_count: int | None = Field(default=None, description="Number of distinct speakers identified")
    created_at: str | None = Field(default=None, description="ISO 8601 timestamp of transcript creation")
    segments: list[TranscriptSegmentOut] = Field(
        default_factory=list, description="Ordered list of transcript segments"
    )


class TranscriptSummaryOut(VideoMetadata):
    id: str = Field(description="Transcript UUID")
    language: str | None = Field(default=None, description="Detected language code, e.g. 'en'")
    has_diarization: bool = Field(description="Whether speaker diarization was performed")
    speaker_count: int | None = Field(default=None, description="Number of distinct speakers identified")
    created_at: str | None = Field(default=None, description="ISO 8601 timestamp of transcript creation")


class TranscriptListResult(BaseModel):
    transcripts: list[TranscriptSummaryOut] = Field(description="List of transcripts for the current page")
    total: int = Field(description="Total number of transcripts across all pages")


class TranscriptDetailResult(VideoMetadata):
    id: str = Field(description="Transcript UUID")
    language: str | None = Field(default=None, description="Detected language code, e.g. 'en'")
    has_diarization: bool = Field(description="Whether speaker diarization was performed")
    speaker_count: int | None = Field(default=None, description="Number of distinct speakers identified")
    created_at: str | None = Field(default=None, description="ISO 8601 timestamp of transcript creation")
    segments: list[TranscriptSegmentOut] = Field(description="Ordered list of transcript segments")


# Input type aliases with rich Field metadata for MCP tool parameters
UrlParam = Annotated[
    str,
    Field(
        description=(
            "Video URL. Supports YouTube (youtube.com/watch?v=..., youtu.be/..., /shorts/, /embed/) "
            "and Twitch (clips: twitch.tv/<channel>/clip/<slug> or clips.twitch.tv/<slug>; "
            "VODs: twitch.tv/videos/<id>)."
        ),
        examples=[
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.twitch.tv/ninoegyn/clip/WanderingGorgeousTireMingLee-udaOxVOvNij79pZS",
            "https://www.twitch.tv/videos/1234567890",
        ],
    ),
]


DiarizeParam = Annotated[
    bool,
    Field(
        description=(
            "Identify speakers with pyannote diarization (labels segments/words with SPEAKER_XX). "
            "Adds processing time and requires an HF token for the gated pyannote model. Defaults to false."
        ),
    ),
]

IsolateParam = Annotated[
    bool,
    Field(
        description=(
            "Isolate vocals with Demucs before transcription. Can improve accuracy on music-heavy "
            "or noisy audio, but adds significant processing time. Defaults to false."
        ),
    ),
]

LimitParam = Annotated[
    int,
    Field(
        ge=1,
        le=100,
        description="Maximum number of transcripts to return per page.",
    ),
]

OffsetParam = Annotated[
    int,
    Field(
        ge=0,
        description="Number of transcripts to skip for pagination.",
    ),
]

TranscriptIdParam = Annotated[
    str,
    Field(
        description="UUID of the transcript to retrieve.",
    ),
]

OutputFormatParam = Annotated[
    Literal["json", "markdown"],
    Field(
        description=(
            "Output format: 'json' returns structured data with segments, 'markdown' returns human-readable text."
        ),
    ),
]


class TranscribeRequest(BaseModel):
    """Request body for the REST POST /api/transcribe endpoint."""

    url: UrlParam
    isolate: bool = False
    diarize: bool = False
