from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Response

from .config import get_settings
from .schemas import AnalyzeFileRequest, AnalyzeResponse, AnalyzeSegmentsRequest, HealthResponse
from .service import QualityService


settings = get_settings()
service = QualityService(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load()
    yield


app = FastAPI(
    title="Audio Segment Quality Assessment",
    version=settings.schema_version,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> Response | HealthResponse:
    payload = service.health()
    if payload.ready:
        return payload
    return Response(
        content=payload.model_dump_json(),
        media_type="application/json",
        status_code=503,
    )


@app.post("/analyze_file", response_model=AnalyzeResponse)
async def analyze_file(request: AnalyzeFileRequest) -> AnalyzeResponse:
    return service.analyze_file(request)


@app.post("/analyze_segments", response_model=AnalyzeResponse)
async def analyze_segments(request: AnalyzeSegmentsRequest) -> AnalyzeResponse:
    return service.analyze_segments(request)


def main() -> None:
    uvicorn.run(
        "audio_quality_service.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        workers=1,
        reload=False,
    )


if __name__ == "__main__":
    main()

