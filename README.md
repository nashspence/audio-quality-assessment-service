# Unified Audio Segment Quality Assessment

Analytical-only HTTP service for long-form wearable audio quality assessment. The service does not run ASR, diarization, speaker identification, emotion inference, sound event detection, or enhancement itself. Instead, it combines optional upstream outputs from those systems with direct audio-derived measurements and transparent rules-based fusion.

## What it returns

For each analyzed segment, the JSON response contains three top-level sections:

- `quality_metrics`: direct waveform and spectral measurements
- `quality_assessment`: interpreted risks and task usability scores
- `confidence`: fused confidence values, reason codes, and contributing factors

When the upload is treated as a file, the response also includes `file_summary`.

## API

- `GET /healthz`
- `GET /v1/schema`
- `POST /v1/analyze`

`POST /v1/analyze` expects `multipart/form-data`:

- `file`: uploaded audio file
- `request_json`: JSON string matching the request schema in [docs/api-schema.md](/workspaces/audio-segment-quality-assessment/docs/api-schema.md)

### Quick start

```bash
cp .env.example .env
docker compose up --build
```

Open `http://localhost:8000/docs` for the interactive FastAPI docs.

### Analyze example audio

```bash
python3 scripts/generate_sample_audio.py
curl -fsS \
  -F "file=@examples/smoke-sample.wav;type=audio/wav" \
  -F "request_json=$(jq -c . examples/smoke-request.json)" \
  http://127.0.0.1:8000/v1/analyze | jq
```

## Runtime configuration

Runtime settings live in `.env`.

- `SERVICE_PORT`
- `API_BASE_URL`
- `LOG_LEVEL`
- `MAX_AUDIO_MB`
- `DEFAULT_BOUNDARY_MARGIN_SECONDS`
- `CLIP_THRESHOLD`
- `USE_OPTIONAL_GPU_MODELS`

The default stack is CPU-safe. If you have a local GPU and want the container to expose it for future optional learned quality models, start with:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The current implementation remains fully functional without a GPU using DSP and rules-based fusion alone.

## Smoke test

```bash
bash scripts/smoke_test.sh
```

The smoke test:

1. Generates a sample WAV file.
2. Starts the compose stack.
3. Waits for the health check to report readiness.
4. Posts representative synthetic upstream metadata.
5. Verifies the response shape with `jq`.
6. Shuts the stack down cleanly.

## Notes on missing upstream metadata

- Missing upstream inputs never prevent analysis.
- Direct audio metrics are always computed.
- Upstream-dependent interpreted fields remain present in the response.
- If a field cannot be estimated responsibly without upstream evidence, its score is `null` and the response includes a note.
