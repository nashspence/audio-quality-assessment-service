#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"

cleanup() {
  local status="$?"
  if [[ "$status" -ne 0 ]]; then
    docker compose logs --no-color || true
  fi
  docker compose down -v --remove-orphans >/dev/null 2>&1 || true
  exit "$status"
}

trap cleanup EXIT

docker compose down -v --remove-orphans >/dev/null 2>&1 || true
docker compose up -d --build

python - <<'PY'
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import os

BASE_URL = os.environ.get(
    "SMOKE_BASE_URL",
    f"http://host.docker.internal:{os.environ.get('SERVICE_PORT', '8000')}",
)
EXPECTED_SEGMENT_KEYS = {
    "segment_id",
    "start",
    "end",
    "speaker_id",
    "source_flags",
    "quality_metrics",
    "quality_assessment",
    "task_usability",
    "reason_codes",
}
EXPECTED_SERVICE_KEYS = {
    "parakeet_tdt",
    "diarizen_wavlm_large_s80_md_v2",
    "titanet",
    "emotion_categorical_3loi",
    "emotion_multi_attributes_3loi",
    "atst_sed",
    "qwen3_spelling_correction",
    "qwen2_5_omni",
    "mossformergan_se_16k",
}


def wait_for_health(timeout_seconds: int = 1200) -> dict:
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=15) as response:
                payload = json.load(response)
                if response.status == 200 and payload.get("ready") is True:
                    return payload
                last_error = payload
        except urllib.error.HTTPError as exc:
            last_error = exc.read().decode()
        except Exception as exc:  # pragma: no cover - smoke path only
            last_error = str(exc)
        time.sleep(5)
    raise SystemExit(f"health never became ready: {last_error}")


def post_json(path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def verify_response_shape(payload: dict) -> None:
    assert payload["schema_version"] == "v1"
    assert set(payload["service_inputs_used"].keys()) == EXPECTED_SERVICE_KEYS
    assert "file_summary" in payload
    assert "aggregates" in payload["file_summary"]
    assert "flags" in payload["file_summary"]
    assert payload["segments"]
    for segment in payload["segments"]:
        assert set(segment.keys()) == EXPECTED_SEGMENT_KEYS
        assert "source_flags" in segment
        assert "quality_metrics" in segment
        assert "quality_assessment" in segment
        assert "task_usability" in segment
        assert isinstance(segment["reason_codes"], list)


audio_b64 = base64.b64encode(Path("test.opus").read_bytes()).decode("ascii")

health_payload = wait_for_health()
assert str(health_payload["device"]).startswith("cuda"), health_payload
assert "cuda" in health_payload["backends"]["nisqa"]["detail"], health_payload

file_request = {
    "schema_version": "v1",
    "audio": {
        "base64": audio_b64,
        "filename": "test.opus",
        "mime_type": "audio/ogg",
    },
    "rolling_windows": {
        "enabled": True,
        "window_seconds": 4.0,
        "hop_seconds": 2.0,
    },
    "upstream_file_level": {
        "parakeet_tdt": {
            "transcript": "hello from the wearable microphone",
            "tokens": [
                {"text": "hello", "start": 0.1, "end": 0.4, "confidence": 0.93},
                {"text": "from", "start": 0.45, "end": 0.62, "confidence": 0.91},
                {"text": "the", "start": 0.67, "end": 0.78, "confidence": 0.88},
                {"text": "wearable", "start": 0.83, "end": 1.2, "confidence": 0.79},
                {"text": "microphone", "start": 1.32, "end": 1.86, "confidence": 0.83},
            ],
        },
        "diarizen_wavlm_large_s80_md_v2": {
            "turns": [
                {"speaker_id": "wearer_a", "start": 0.0, "end": 1.5, "overlap": False},
                {"speaker_id": "wearer_a", "start": 1.45, "end": 2.1, "overlap": True},
                {"speaker_id": "wearer_b", "start": 2.1, "end": 3.4, "overlap": False},
            ]
        },
        "titanet": {
            "windows": [
                {"speaker_id": "wearer_a", "start": 0.0, "end": 1.6, "confidence": 0.86},
                {"speaker_id": "wearer_b", "start": 2.0, "end": 3.4, "confidence": 0.72},
            ]
        },
        "emotion_categorical_3loi": {
            "windows": [
                {"start": 0.0, "end": 1.5, "top_label": "neutral", "label_scores": {"neutral": 0.72, "sad": 0.14}},
                {"start": 1.5, "end": 3.0, "top_label": "neutral", "label_scores": {"neutral": 0.63, "angry": 0.19}},
            ]
        },
        "emotion_multi_attributes_3loi": {
            "windows": [
                {"start": 0.0, "end": 1.5, "attributes": {"valence": 0.08, "arousal": 0.21}},
                {"start": 1.5, "end": 3.0, "attributes": {"valence": 0.05, "arousal": 0.26}},
            ]
        },
        "atst_sed": {
            "events": [
                {"label": "speech", "start": 0.0, "end": 1.8, "score": 0.88},
                {"label": "walking", "start": 1.8, "end": 2.6, "score": 0.54},
            ]
        },
        "qwen3_spelling_correction": {
            "raw_text": "helo from the wearable microfone",
            "corrected_text": "hello from the wearable microphone",
        },
        "qwen2_5_omni": {
            "windows": [
                {"start": 0.0, "end": 1.8, "scene": "indoor_office", "tags": ["speech", "desk"], "confidence": 0.81},
                {"start": 1.8, "end": 3.2, "scene": "hallway", "tags": ["footsteps", "speech"], "confidence": 0.63},
            ]
        },
    },
}

file_response = post_json("/analyze_file", file_request)
verify_response_shape(file_response)

full_file_segment = next(segment for segment in file_response["segments"] if segment["segment_id"] == "full_file")
duration = float(full_file_segment["end"])
half = round(max(duration / 2.0, 1.0), 3)
segments = [
    {
        "segment_id": "seg_a",
        "start": 0.0,
        "end": min(half, duration),
        "speaker_id": "wearer_a",
    }
]
if duration - half > 0.75:
    segments.append(
        {
            "segment_id": "seg_b",
            "start": half,
            "end": duration,
            "speaker_id": "wearer_b",
        }
    )

segment_request = {
    "schema_version": "v1",
    "audio": {
        "base64": audio_b64,
        "filename": "test.opus",
        "mime_type": "audio/ogg",
    },
    "enhanced_audio": {
        "base64": audio_b64,
        "filename": "test.opus",
        "mime_type": "audio/ogg",
    },
    "segments": segments,
    "upstream_by_segment": {
        "seg_a": {
            "parakeet_tdt": {
                "transcript": "hello from the wearable microphone",
                "avg_token_confidence": 0.89,
                "tokens": [
                    {"text": "hello", "start": 0.1, "end": 0.4, "confidence": 0.93},
                    {"text": "wearable", "start": 0.83, "end": 1.2, "confidence": 0.79},
                ],
            },
            "diarizen_wavlm_large_s80_md_v2": {
                "dominant_speaker_id": "wearer_a",
                "nearest_boundary_distance_s": 0.18,
                "overlap_fraction": 0.08,
                "overlap_flag": False,
            },
            "titanet": {
                "speaker_id": "wearer_a",
                "confidence": 0.87,
                "alternatives": [{"speaker_id": "wearer_b", "confidence": 0.09}],
            },
            "emotion_categorical_3loi": {
                "top_label": "neutral",
                "label_scores": {"neutral": 0.71, "sad": 0.12},
                "adjacent_window_labels": ["neutral", "neutral"],
            },
            "emotion_multi_attributes_3loi": {
                "attributes": {"valence": 0.05, "arousal": 0.24},
                "adjacent_window_attributes": [{"valence": 0.04, "arousal": 0.23}],
            },
            "atst_sed": {
                "dominant_label": "speech",
                "dominant_score": 0.88,
                "events": [{"label": "speech", "start": 0.0, "end": min(half, duration), "score": 0.88}],
            },
            "qwen3_spelling_correction": {
                "raw_text": "helo from the wearable microfone",
                "corrected_text": "hello from the wearable microphone",
            },
            "qwen2_5_omni": {
                "scene": "indoor_office",
                "tags": ["speech", "desk"],
                "confidence": 0.81,
                "adjacent_scenes": ["indoor_office"],
            },
            "mossformergan_se_16k": {
                "model_name": "MossFormerGAN_SE_16K",
                "notes": ["synthetic-smoke-input"],
            },
        }
    },
}
if len(segments) > 1:
    segment_request["upstream_by_segment"]["seg_b"] = {
        "parakeet_tdt": {
            "transcript": "background movement",
            "avg_token_confidence": 0.67,
        },
        "diarizen_wavlm_large_s80_md_v2": {
            "dominant_speaker_id": "wearer_b",
            "nearest_boundary_distance_s": 0.11,
            "overlap_fraction": 0.22,
            "overlap_flag": True,
        },
        "titanet": {
            "speaker_id": "wearer_b",
            "confidence": 0.71,
        },
        "emotion_categorical_3loi": {
            "top_label": "neutral",
            "label_scores": {"neutral": 0.52, "angry": 0.24},
            "adjacent_window_labels": ["neutral", "angry"],
        },
        "emotion_multi_attributes_3loi": {
            "attributes": {"valence": -0.08, "arousal": 0.42},
            "adjacent_window_attributes": [{"valence": -0.04, "arousal": 0.36}],
        },
        "atst_sed": {
            "dominant_label": "walking",
            "dominant_score": 0.63,
            "events": [{"label": "walking", "start": half, "end": duration, "score": 0.63}],
        },
        "qwen3_spelling_correction": {
            "raw_text": "bakground movement",
            "corrected_text": "background movement",
        },
        "qwen2_5_omni": {
            "scene": "hallway",
            "tags": ["footsteps", "speech"],
            "confidence": 0.63,
            "adjacent_scenes": ["indoor_office", "hallway"],
        },
        "mossformergan_se_16k": {
            "model_name": "MossFormerGAN_SE_16K",
            "notes": ["synthetic-smoke-input"],
        },
    }

segment_response = post_json("/analyze_segments", segment_request)
verify_response_shape(segment_response)

for payload in (file_response, segment_response):
    for segment in payload["segments"]:
        assert set(segment["source_flags"].keys()) == {
            "speaker_id",
            "quality_metrics",
            "learned_speech_quality",
            "boundary_risk",
            "overlap_risk",
            "speaker_consistency",
            "asr_consistency",
            "spelling_correction_delta",
            "sed_context_dominance",
            "emotion_instability",
            "enhancement_delta",
            "general_context_consistency",
            "segment_integrity",
            "confidence_components",
            "task_usability",
        }
        assert set(segment["quality_metrics"].keys()) == {
            "duration_s",
            "speech_ratio",
            "silence_ratio",
            "rms_dbfs",
            "peak_dbfs",
            "dynamic_range_db",
            "crest_factor_db",
            "clipping_fraction",
            "zero_crossing_rate",
            "spectral_centroid_hz",
            "spectral_rolloff_hz",
            "spectral_bandwidth_hz",
            "low_freq_energy_ratio",
            "high_freq_energy_ratio",
            "blind_snr_db",
        }
        assert set(segment["task_usability"].keys()) == {
            "asr",
            "emotion",
            "speaker_id",
            "sed",
            "general_audio_understanding",
        }
        assert "overall_confidence" in segment["quality_assessment"]
        assert "confidence_components" in segment["quality_assessment"]

print("Smoke test passed.")
PY
