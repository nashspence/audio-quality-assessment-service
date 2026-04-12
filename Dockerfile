FROM python:3.11-slim-bookworm

ARG NISQA_REF=fe84f0f252abec382b24367d5b22498a7ce34dbb
ARG PYTORCH_VERSION=2.9.1
ARG TORCHVISION_VERSION=0.24.1
ARG TORCHAUDIO_VERSION=2.9.1
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu130

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    PIP_EXTRA_INDEX_URL=https://pypi.org/simple

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libgomp1 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/gabrielmittag/NISQA.git /opt/vendor/NISQA \
    && git -C /opt/vendor/NISQA checkout "${NISQA_REF}"

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        --index-url "${PYTORCH_INDEX_URL}" \
        "torch==${PYTORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
    && python -m pip install /app

EXPOSE 8000

CMD ["python", "-m", "audio_quality_service.main"]
