# Audio Segment Quality Assessment

This repository packages a single HTTP service for deterministic `v1` scoring of noisy wearable-audio quality, segment integrity, and confidence. It does not run upstream ASR, diarization, speaker ID, emotion, SED, spelling correction, audio understanding, or enhancement; it only consumes their supplied outputs and fuses them with raw-audio measurements.

## Endpoints

- `GET /health`
- `POST /analyze_file`
- `POST /analyze_segments`

Both POST endpoints accept JSON with `schema_version: "v1"` and an `audio` object that can use either a local container-visible `path` or base64-encoded bytes. `analyze_segments` also accepts optional `enhanced_audio`.

## Run

```bash
docker compose up --build
```

The service binds to `http://localhost:8000` by default. Required quality-model artifacts are downloaded to the repo-local `./.cache/models` directory on startup if missing. Health only becomes ready after NISQA, optional DNSMOS, and the API are fully loaded.

The container is pinned to a Blackwell-capable PyTorch wheel stack: `torch 2.9.1`, `torchvision 0.24.1`, and `torchaudio 2.9.1` from the official `cu130` index. With the default `.env` values, NISQA is expected to load on `cuda:0`; if CUDA is unavailable, `/health` stays unready instead of silently falling back to CPU.

## Smoke Test

```bash
bash scripts/smoke_test.sh
```

The smoke test builds the stack, waits for health readiness, exercises both POST endpoints with the included `test.opus` clip plus synthetic upstream payloads spanning the named services, verifies response shape, and shuts the stack down cleanly.
