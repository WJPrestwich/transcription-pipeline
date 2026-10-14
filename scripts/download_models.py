"""Pre-download all ML models needed by the pipeline.

Run this script inside the server container to cache models to the volume-mounted
cache directories. Models are persisted across container restarts via Docker volumes:

    docker compose exec server python scripts/download_models.py

Models cached:
  1. WhisperX/faster-whisper      -> ~/.cache/huggingface/hub/
  2. Wav2Vec2 alignment           -> ~/.cache/huggingface/hub/
  3. Demucs htdemucs_ft           -> ~/.cache/torch/hub/checkpoints/
  4. Pyannote diarization (gated) -> ~/.cache/huggingface/hub/

The pyannote diarization model requires a HuggingFace token (accept the model's
terms at https://huggingface.co/pyannote/speaker-diarization-community-1 first).
Set HF_TOKEN in the environment before running.
"""

import os

DEVICE = os.environ.get("DEVICE", "cpu")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "int8")


def download_whisper_model() -> None:
    import whisperx

    print(f"Downloading whisper model: {WHISPER_MODEL} (compute_type={COMPUTE_TYPE}, device={DEVICE})")
    model = whisperx.load_model(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
    del model

    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    print("Whisper model downloaded and cached.")


def download_align_model() -> None:
    """Pre-download alignment models for common languages."""
    from whisperx import load_align_model

    languages = ["en", "fr", "de", "es", "it", "pt", "ja", "zh", "ko", "ru"]
    for lang in languages:
        print(f"Downloading alignment model for: {lang}")
        try:
            model_a, metadata = load_align_model(language_code=lang, device=DEVICE)
            del model_a
        except Exception as e:
            print(f"  Skipped {lang}: {e}")

    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    print("Alignment models downloaded and cached.")


def download_demucs_model() -> None:
    from demucs.pretrained import get_model

    print("Downloading demucs model: htdemucs_ft")
    model = get_model("htdemucs_ft")
    del model
    print("Demucs model downloaded and cached.")


def download_diarize_model() -> None:
    from whisperx.diarize import DiarizationPipeline

    # Instantiating the pipeline downloads the gated pyannote model;
    # huggingface_hub picks up HF_TOKEN from the environment.
    print("Downloading pyannote diarization model: pyannote/speaker-diarization-community-1")
    diarizer = DiarizationPipeline(device=DEVICE, token=os.environ.get("HF_TOKEN"))
    del diarizer

    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    print("Diarization model downloaded and cached.")


def main() -> None:
    print("=" * 60)
    print("Pre-downloading ML models...")
    print("=" * 60)

    download_whisper_model()
    download_align_model()
    download_demucs_model()
    download_diarize_model()

    print("=" * 60)
    print("All models downloaded and cached.")
    print("=" * 60)


if __name__ == "__main__":
    main()
