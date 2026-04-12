from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
import json
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.config import get_settings
from app.models import AnalyzeRequest, AnalyzeResponse
from app.service import analyze_audio


def detect_gpu() -> bool:
    return any(os.path.exists(path) for path in ("/dev/nvidia0", "/dev/dxg"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.ready = False
    app.state.initialized_at = datetime.now(UTC).isoformat()
    app.state.runtime = {
        "gpu_available": detect_gpu(),
        "optional_gpu_models_enabled": settings.use_optional_gpu_models,
    }
    app.state.ready = True
    yield


app = FastAPI(
    title="Unified Audio Quality Assessment API",
    version=get_settings().service_version,
    description=(
        "Analytical-only HTTP service for direct audio quality measurements, "
        "interpreted segment quality assessment, and transparent rules-based confidence fusion."
    ),
    lifespan=lifespan,
)


@app.get("/healthz")
async def healthz() -> dict[str, object]:
    settings = get_settings()
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "ready": bool(app.state.ready),
        "gpu_available": app.state.runtime["gpu_available"],
        "optional_gpu_models_enabled": app.state.runtime["optional_gpu_models_enabled"],
        "initialized_at": app.state.initialized_at,
    }


@app.get("/v1/schema")
async def schema() -> dict[str, object]:
    return {
        "transport": {
            "content_type": "multipart/form-data",
            "file_field": "file",
            "json_field": "request_json",
            "behavior": (
                "Send audio in `file` and the structured request payload as a JSON string in `request_json`."
            ),
        },
        "request_payload_json_schema": AnalyzeRequest.model_json_schema(),
        "response_json_schema": AnalyzeResponse.model_json_schema(),
        "optional_upstream_behavior": {
            "diarization_segments": "Optional. When absent, overlap risk falls back to audio-only heuristics.",
            "speaker_id_segments": "Optional. When absent, speaker consistency is returned with a note and null score.",
            "known_speaker_ids": "Optional. Used only as supporting context when speaker-id metadata is present.",
            "asr_words": "Optional. When absent, ASR usability uses audio-only evidence.",
            "emotion_segments": "Optional. When absent, neighboring-window instability is null unless ASR metadata is present.",
            "sed_events": "Optional. When absent, interfering-event penalties are not applied.",
            "enhancement": "Optional. When absent, no enhancement penalties or boosts are applied.",
        },
    }


@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    request_json: str = Form(default="{}"),
) -> AnalyzeResponse:
    settings = get_settings()
    payload = await file.read()
    max_audio_bytes = settings.max_audio_mb * 1024 * 1024
    if len(payload) > max_audio_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file exceeds the configured limit of {settings.max_audio_mb} MB.",
        )
    try:
        request_data = json.loads(request_json or "{}")
        options = request_data.setdefault("options", {})
        options.setdefault(
            "boundary_margin_seconds", settings.default_boundary_margin_seconds
        )
        options.setdefault("clip_threshold", settings.clip_threshold)
        request = AnalyzeRequest.model_validate(request_data)
    except (
        Exception
    ) as exc:  # pragma: no cover - FastAPI returns structured error payload
        raise HTTPException(
            status_code=422, detail=f"Invalid request_json payload: {exc}"
        ) from exc
    try:
        return analyze_audio(payload, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
