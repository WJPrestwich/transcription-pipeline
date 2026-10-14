# Plain Ubuntu base: the pip nvidia-*-cu12 wheels in the venv already bundle all
# CUDA userland libs (cudnn, cublas, ...). Only the host's NVIDIA driver is
# needed, which nvidia-container-toolkit injects at runtime. Using the
# nvidia/cuda base would duplicate ~3GB of CUDA libraries.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    # Standalone binary: the base image has no system python for the zipapp variant
    && curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_linux -o /usr/local/bin/yt-dlp \
    && chmod a+rx /usr/local/bin/yt-dlp

RUN useradd --create-home app

# Pre-create cache dirs owned by app so named volumes inherit correct ownership
RUN mkdir -p /home/app/.cache/huggingface /home/app/.cache/torch \
    && chown -R app:app /home/app/.cache

USER app

ENV PATH="/home/app/.local/bin:${PATH}"
ENV HF_HOME=/home/app/.cache/huggingface

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

# --- Dependency install (cached unless pyproject.toml or uv.lock change) ---
COPY --chown=app:app pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --python 3.13 --no-install-project

# --- Application source (invalidates only when code changes) ---
COPY --chown=app:app src/ src/
COPY --chown=app:app scripts/ scripts/
RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "youtube_transcription.server"]