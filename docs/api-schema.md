# API Schema

The service exposes a single analysis endpoint:

- `POST /v1/analyze`
  - Transport: `multipart/form-data`
  - Binary audio field: `file`
  - Structured JSON field: `request_json`

The `request_json` payload uses the schema below.

## Request

```json
{
  "request_id": "optional-string",
  "analysis_target": "file",
  "segments": [
    {
      "segment_id": "seg_001",
      "start_time": 0.0,
      "end_time": 12.5,
      "label": "optional",
      "channel": 0
    }
  ],
  "upstream": {
    "diarization_segments": [
      {
        "start_time": 0.0,
        "end_time": 3.2,
        "speaker_id": "wearer",
        "confidence": 0.91,
        "overlap": false
      }
    ],
    "speaker_id_segments": [
      {
        "start_time": 0.0,
        "end_time": 3.2,
        "speaker_id": "wearer",
        "confidence": 0.94
      }
    ],
    "known_speaker_ids": ["wearer", "companion"],
    "asr_words": [
      {
        "start_time": 0.12,
        "end_time": 0.34,
        "word": "hello",
        "confidence": 0.95,
        "confidence_proxy": 0.95
      }
    ],
    "emotion_segments": [
      {
        "start_time": 0.0,
        "end_time": 1.5,
        "label": "neutral",
        "score": 0.7,
        "uncertainty": 0.1
      }
    ],
    "sed_events": [
      {
        "start_time": 2.0,
        "end_time": 3.0,
        "event_label": "vehicle",
        "score": 0.8
      }
    ],
    "enhancement": {
      "applied": true,
      "provider": "upstream-denoiser",
      "noise_reduction_db": 6.0,
      "artifacts_risk": 0.2,
      "residual_noise_risk": 0.3,
      "clipping_repaired": false,
      "notes": "optional"
    }
  },
  "options": {
    "include_file_summary": true,
    "boundary_margin_seconds": 0.25,
    "clip_threshold": 0.99
  }
}
```

### Optional upstream behavior

- `diarization_segments` is optional. If absent, overlap risk uses audio-only heuristics.
- `speaker_id_segments` is optional. If absent, `speaker_consistency.score` is `null`.
- `known_speaker_ids` is optional. It only provides context when speaker metadata exists.
- `asr_words` is optional. If absent, ASR usability uses audio-only evidence.
- `emotion_segments` is optional. If absent, neighboring-window instability is `null` unless ASR confidence proxies are present.
- `sed_events` is optional. If absent, no interfering-event penalty is applied.
- `enhancement` is optional. If absent, no enhancement penalty or boost is applied.

### Segment handling

- If `segments` is omitted, the uploaded audio is treated as one segment.
- If `analysis_target` is `segment`, the upload is treated as a single audio segment and the response still returns one item in `segments`.
- If `analysis_target` is `file`, the response includes `file_summary` plus the per-segment list.

## Response

Each segment contains exactly three top-level analytical sections:

- `quality_metrics`: direct measurements from the audio waveform
- `quality_assessment`: interpreted risks and task-specific usability
- `confidence`: transparent fused confidence values, reason codes, and contributing factors

```json
{
  "schema_version": "1.0.0",
  "request_id": "optional-string",
  "analysis_target": "file",
  "file_summary": {
    "duration_seconds": 120.0,
    "sample_rate_hz": 16000,
    "channel_count": 1,
    "segment_count": 3,
    "analyzed_duration_seconds": 120.0,
    "quality_metrics_summary": {
      "mean_speech_ratio": 0.61,
      "mean_silence_ratio": 0.17,
      "mean_blind_snr_estimate_db": 14.2,
      "mean_clipping_fraction": 0.0
    },
    "quality_assessment_summary": {
      "mean_overlap_risk": 0.19,
      "mean_boundary_proximity_risk": 0.27,
      "mean_general_usability": 0.74
    },
    "confidence_summary": {
      "overall_confidence_mean": 0.78,
      "segment_integrity_mean": 0.8,
      "speech_usability_for_asr_mean": 0.76,
      "top_reason_codes": ["strong_speech_coverage", "high_estimated_snr"]
    }
  },
  "segments": [
    {
      "segment_id": "seg_001",
      "start_time": 0.0,
      "end_time": 12.5,
      "label": "optional",
      "quality_metrics": {
        "duration_seconds": 12.5,
        "sample_rate_hz": 16000,
        "channels_original": 1,
        "rms_level_dbfs": -22.3,
        "peak_level_dbfs": -1.5,
        "dynamic_range_db": 18.2,
        "crest_factor": 3.7,
        "clipping_fraction": 0.0,
        "zero_crossing_rate": 0.11,
        "speech_ratio": 0.67,
        "silence_ratio": 0.14,
        "blind_snr_estimate_db": 15.1,
        "spectral_centroid_hz": 1360.0,
        "spectral_bandwidth_hz": 1720.0,
        "spectral_rolloff_hz": 3120.0,
        "spectral_flatness": 0.18,
        "spectral_flux": 0.03,
        "low_band_energy_ratio": 0.24,
        "speech_band_energy_ratio": 0.61,
        "high_band_energy_ratio": 0.15,
        "energy_entropy": 0.81,
        "dc_offset": 0.0002,
        "speech_frame_count": 811,
        "total_frame_count": 1204
      },
      "quality_assessment": {
        "overlap_risk": {
          "score": 0.14,
          "label": "low",
          "evidence": ["audio_only_heuristic"]
        },
        "boundary_proximity_risk": {
          "score": 0.28,
          "label": "low",
          "evidence": ["edge_speech_activity"]
        },
        "speech_dominance": {
          "score": 0.67,
          "label": "good",
          "evidence": ["direct_audio_measurement"]
        },
        "non_speech_dominance": {
          "score": 0.33,
          "label": "poor",
          "evidence": ["direct_audio_measurement"]
        },
        "speaker_consistency": {
          "score": 0.92,
          "label": "excellent",
          "evidence": ["speaker_id_segments", "known_speaker_ids"],
          "note": null
        },
        "neighboring_window_instability": {
          "score": 0.21,
          "label": "low",
          "evidence": ["emotion_segments"],
          "note": null
        },
        "usability_for_asr": {
          "score": 0.85,
          "label": "excellent",
          "evidence": ["audio_metrics", "asr_word_confidence_proxy"]
        },
        "usability_for_emotion": {
          "score": 0.78,
          "label": "good",
          "evidence": ["audio_metrics", "emotion_segments"]
        },
        "usability_for_speaker_id": {
          "score": 0.84,
          "label": "good",
          "evidence": ["audio_metrics", "speaker_identity_metadata"]
        },
        "usability_for_general_downstream": {
          "score": 0.8,
          "label": "good",
          "evidence": ["audio_metrics", "rules_based_fusion"]
        }
      },
      "confidence": {
        "overall_confidence": 0.82,
        "segment_integrity": 0.84,
        "speech_usability_for_asr": 0.85,
        "speech_usability_for_emotion": 0.78,
        "speech_usability_for_speaker_id": 0.84,
        "reason_codes": [
          "strong_speech_coverage",
          "high_estimated_snr",
          "speaker_identity_stable"
        ],
        "contributing_factors": [
          {
            "name": "Speech coverage",
            "direction": "boost",
            "source": "audio",
            "impact": 0.08,
            "value": 0.67,
            "explanation": "The segment contains a strong share of speech-dominant frames."
          }
        ]
      }
    }
  ]
}
```
