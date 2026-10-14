import contextlib
import gc
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import whisperx

logger = logging.getLogger(__name__)


@dataclass
class ProcessedResult:
    """Structured output from the ML pipeline, ready for DB storage."""

    language: str | None
    segments: list[dict]
    has_diarization: bool
    speaker_count: int | None = None


@dataclass
class VideoMetadata:
    """Structured metadata from yt-dlp (video_id is the platform's canonical ID)."""

    video_id: str
    title: str | None
    channel: str | None
    duration_seconds: float | None
    view_count: int | None


def free_vram(*models: object) -> None:
    """Free GPU memory held by the given model objects.

    Replaces all parameter/buffer storage with empty CPU tensors, moves model
    to CPU, then forces garbage collection and CUDA cache clearing. The caller
    should set their reference to None after calling this.
    """
    names = [type(m).__name__ for m in models]
    allocated_before = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    for model in models:
        logger.info("free_vram: releasing %s", type(model).__name__)
        # Wrappers like whisperx.DiarizationPipeline hold the actual nn.Module in .model
        if not isinstance(model, torch.nn.Module) and isinstance(getattr(model, "model", None), torch.nn.Module):
            free_vram(model.model)
            continue
        if isinstance(model, torch.nn.Module):
            for param in list(model.parameters()):
                param.data = torch.empty(0, device="cpu")
            for buf in list(model.buffers()):
                buf.data = torch.empty(0, device="cpu")
            with contextlib.suppress(Exception):
                model.cpu()
        elif hasattr(model, "cpu"):
            with contextlib.suppress(Exception):
                model.cpu()
    for _ in range(3):
        gc.collect()
    torch.cuda.empty_cache()
    allocated_after = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    logger.info("free_vram: released %s — VRAM %.2fGB → %.2fGB", names, allocated_before, allocated_after)


def ensure_cpu(result: dict) -> dict:
    """Move any CUDA tensors in a whisperx result dict to CPU."""
    if "segments" in result:
        for seg in result["segments"]:
            for key in list(seg.keys()):
                val = seg[key]
                if isinstance(val, torch.Tensor) and val.is_cuda:
                    seg[key] = val.cpu()
                elif isinstance(val, list):
                    seg[key] = [v.cpu() if isinstance(v, torch.Tensor) and v.is_cuda else v for v in val]
    if "word_segments" in result:
        for ws in result.get("word_segments", []):
            for key in list(ws.keys()):
                val = ws[key]
                if isinstance(val, torch.Tensor) and val.is_cuda:
                    ws[key] = val.cpu()
    return result


def download_audio(url: str, output_dir: Path) -> Path | None:
    """Download the best-quality audio track from a YouTube or Twitch URL as WAV.

    Uses yt-dlp to extract audio without downloading the full video.
    WAV output avoids lossy re-encoding and is natively supported by
    whisperx and demucs.
    """
    logger.info("download_audio: url=%s output_dir=%s", url, output_dir)
    cmd = [
        "yt-dlp",
        "--output",
        "%(channel)s.%(uploader)s.%(title)s.%(webpage_url_domain)s.%(view_count)s.%(id)s.%(ext)s",
        "--print",
        "after_move:filepath",
        "--no-quiet",
        "--restrict-filenames",
        "--js-runtimes",
        "deno",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        url,
    ]
    result = subprocess.run(cmd, cwd=output_dir, text=True, capture_output=True)
    if result.returncode != 0:
        logger.error("download_audio: yt-dlp failed (rc=%d) stderr=%s", result.returncode, result.stderr[:500])
        return None
    for line in result.stdout.strip().splitlines():
        p = output_dir / line.strip()
        if p.exists() and p.is_file():
            logger.info("download_audio: saved %s", p.name)
            return p
    logger.error("download_audio: yt-dlp succeeded but no output file found")
    return None


def get_video_metadata(url: str) -> VideoMetadata | None:
    logger.info("get_video_metadata: url=%s", url)
    cmd = [
        "yt-dlp",
        "--print",
        "%(id)s\n%(title)s\n%(channel)s\n%(duration)s\n%(view_count)s",
        "--no-download",
        "--restrict-filenames",
        "--js-runtimes",
        "deno",
        url,
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        logger.error("get_video_metadata: yt-dlp failed (rc=%d) stderr=%s", result.returncode, result.stderr[:500])
        return None
    lines = result.stdout.strip().splitlines()
    if len(lines) < 5:
        logger.error("get_video_metadata: unexpected output (%d lines): %s", len(lines), result.stdout[:300])
        return None
    metadata = VideoMetadata(
        video_id=lines[0],
        title=lines[1] if lines[1] != "NA" else None,
        channel=lines[2] if lines[2] != "NA" else None,
        duration_seconds=float(lines[3]) if lines[3] != "NA" else None,
        view_count=int(lines[4]) if lines[4] != "NA" else None,
    )
    logger.info(
        "get_video_metadata: id=%s title=%s duration=%ss",
        metadata.video_id,
        metadata.title,
        metadata.duration_seconds,
    )
    return metadata


def transcribe(
    audio: np.ndarray, model_name: str, device: str, compute_type: str, batch_size: int
) -> tuple[dict, object]:
    logger.info(
        "transcribe: model=%s device=%s compute_type=%s batch_size=%d audio_shape=%s",
        model_name,
        device,
        compute_type,
        batch_size,
        audio.shape,
    )
    model = whisperx.load_model(model_name, device=device, compute_type=compute_type)
    result = model.transcribe(audio, batch_size=batch_size)
    lang = result.get("language", "?")
    n_segs = len(result.get("segments", []))
    logger.info("transcribe: detected language=%s segments=%d", lang, n_segs)
    return result, model


def align(segments: list, audio: np.ndarray, language: str, device: str) -> tuple[dict, object]:
    logger.info("align: language=%s device=%s segments=%d", language, device, len(segments))
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(
        segments,
        model_a,
        metadata,
        audio,
        device=device,
        return_char_alignments=False,
    )
    logger.info("align: aligned %d segments", len(result.get("segments", [])))
    return result, model_a


def isolate_vocals(audio_path: Path, output_dir: Path, device: str) -> tuple[Path, object]:
    from demucs.pretrained import get_model
    from demucs.separate import apply_model, load_track, save_audio

    logger.info("isolate_vocals: audio=%s device=%s", audio_path.name, device)
    model = get_model("htdemucs_ft")
    model.to(device)
    model.eval()

    wav = load_track(str(audio_path), model.audio_channels, model.samplerate)
    logger.info("isolate_vocals: loaded track shape=%s sample_rate=%d", wav.shape, model.samplerate)
    ref = wav.mean(0)
    sources = apply_model(model, wav[None], device=device, progress=True)

    vocal_idx = model.sources.index("vocals")
    vocals = sources[0, vocal_idx]
    vocals = vocals * ref.abs().max() / vocals.abs().max()

    output_path = output_dir / f"{audio_path.stem}_vocals.wav"
    save_audio(vocals, str(output_path), model.samplerate)
    size_mb = output_path.stat().st_size / 1024**2
    logger.info("isolate_vocals: saved %s (%.1fMB)", output_path.name, size_mb)

    return output_path, model


def diarize(audio: Path | str, result: dict, device: str, hf_token: str | None = None) -> tuple[dict, object]:
    """Diarize speakers with WhisperX's pyannote pipeline and label segments/words.

    The underlying pyannote model (pyannote/speaker-diarization-community-1) is
    gated on HuggingFace, so a token is required — pass hf_token or set the
    HF_TOKEN env var (huggingface_hub picks it up automatically).
    """
    # Lazy import: whisperx.diarize pulls in pyannote, which is heavy at import time
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers

    n_segs = len(result.get("segments", []))
    logger.info("diarize: audio=%s device=%s input_segments=%d", str(audio), device, n_segs)

    diarizer = DiarizationPipeline(device=device, token=hf_token)
    diarize_df = diarizer(str(audio))
    logger.info("diarize: raw output segments=%d", len(diarize_df))

    result = assign_word_speakers(diarize_df, result)
    speakers = {s.get("speaker") for s in result["segments"] if s.get("speaker")}
    logger.info("diarize: identified %d speakers: %s", len(speakers), sorted(speakers))
    return result, diarizer


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# Supported URL patterns: YouTube videos and Twitch clips/VODs.
# The canonical video ID is taken from yt-dlp metadata, so these patterns only
# gate which URLs we accept — they don't need to capture anything.
SUPPORTED_URL_PATTERNS = [
    # YouTube: watch, youtu.be, /shorts/, /embed/
    r"(?:youtube\.com/watch\?.*v=|youtu\.be/|youtube\.com/(?:shorts|embed)/)[a-zA-Z0-9_-]{11}",
    # Twitch clips: twitch.tv/<channel>/clip/<slug> or clips.twitch.tv/<slug>
    r"twitch\.tv/[^/]+/clip/[a-zA-Z0-9_-]+",
    r"clips\.twitch\.tv/[a-zA-Z0-9_-]+",
    # Twitch VODs: twitch.tv/videos/<id>
    r"twitch\.tv/videos/\d+",
]


def validate_url(url: str) -> None:
    """Raise ValueError unless the URL is a supported YouTube or Twitch video URL."""
    for pattern in SUPPORTED_URL_PATTERNS:
        if re.search(pattern, url):
            return
    raise ValueError(f"Unsupported URL (expected a YouTube video, Twitch clip, or Twitch VOD): {url}")


def result_to_dict(result: dict) -> dict:
    segments = []
    for segment in result["segments"]:
        seg = {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"].strip(),
        }
        if "speaker" in segment:
            seg["speaker"] = segment["speaker"]
        if segment.get("words"):
            seg["words"] = segment["words"]
        segments.append(seg)
    return {"language": result.get("language"), "segments": segments}


def result_to_markdown(result: dict) -> str:
    lines = []
    for segment in result["segments"]:
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        speaker = segment.get("speaker")
        header = f"**{speaker}** " if speaker else ""
        lines.append(f"[{start} - {end}] {header}{segment['text'].strip()}")
    return "\n".join(lines)


def run_pipeline(
    audio_path: Path,
    tmpdir: Path,
    device: str,
    whisper_model: str,
    compute_type: str,
    batch_size: int,
    do_diarize: bool = False,
    do_isolate: bool = False,
    hf_token: str | None = None,
) -> ProcessedResult:
    """Run the full ML pipeline: optionally isolate vocals, transcribe, align, optionally diarize.

    Vocal isolation (Demucs) is off by default — it helps on music-heavy audio but
    costs extra processing time. Each model is freed from GPU memory immediately
    after use to minimize VRAM pressure. Returns a ProcessedResult ready for DB storage.
    """
    import time

    logger.info(
        "run_pipeline: starting pipeline audio=%s device=%s whisper_model=%s compute_type=%s batch_size=%d"
        " diarize=%s isolate=%s",
        audio_path.name,
        device,
        whisper_model,
        compute_type,
        batch_size,
        do_diarize,
        do_isolate,
    )

    t_start = time.monotonic()

    # Vocal isolation (optional)
    source = audio_path
    if do_isolate:
        t0 = time.monotonic()
        source, isolate_model = isolate_vocals(audio_path, tmpdir, device)
        logger.info("run_pipeline: isolate_vocals took %.1fs", time.monotonic() - t0)
        free_vram(isolate_model)
        isolate_model = None

    audio_data = whisperx.load_audio(str(source))

    # Transcription
    t0 = time.monotonic()
    result, transcribe_model = transcribe(audio_data, whisper_model, device, compute_type, batch_size)
    logger.info("run_pipeline: transcribe took %.1fs", time.monotonic() - t0)
    language = result.get("language")

    # Alignment
    try:
        t0 = time.monotonic()
        result, align_model = align(result["segments"], audio_data, result["language"], device)
        # whisperx.align's result drops the top-level language key — restore it
        result["language"] = language
        logger.info("run_pipeline: align took %.1fs", time.monotonic() - t0)
    finally:
        free_vram(transcribe_model)
        transcribe_model = None
    free_vram(align_model)
    align_model = None
    result = ensure_cpu(result)
    del audio_data

    # Diarization
    if do_diarize:
        t0 = time.monotonic()
        result, diarizer = diarize(source, result, device, hf_token)
        logger.info("run_pipeline: diarize took %.1fs", time.monotonic() - t0)
        free_vram(diarizer)
        diarizer = None

    processed = result_to_dict(result)
    segs = processed["segments"]
    speakers = {s["speaker"] for s in segs if "speaker" in s}

    elapsed = time.monotonic() - t_start
    logger.info(
        "run_pipeline: completed in %.1fs — %d segments, %d speakers, language=%s",
        elapsed,
        len(segs),
        len(speakers) if speakers else 0,
        processed.get("language"),
    )

    return ProcessedResult(
        language=processed.get("language"),
        segments=segs,
        has_diarization=do_diarize,
        speaker_count=len(speakers) if speakers else None,
    )
