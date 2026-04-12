from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from app.audio_features import compute_quality_metrics, load_audio_from_bytes
from app.fusion import build_confidence, build_quality_assessment
from app.models import AnalyzeRequest, AnalyzeResponse, FileSummary, SegmentResult


def _clip_time(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(float(value), maximum))


def _to_plain_dicts(items: list[Any]) -> list[dict[str, Any]]:
    return [
        item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
        for item in items
    ]


def _overlapping_items(
    items: list[dict[str, Any]],
    segment_start: float,
    segment_end: float,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for item in items:
        if (
            float(item["end_time"]) <= segment_start
            or float(item["start_time"]) >= segment_end
        ):
            continue
        selected.append(item)
    return selected


def _default_segments(
    request: AnalyzeRequest, duration_seconds: float
) -> list[dict[str, Any]]:
    if request.segments:
        segments: list[dict[str, Any]] = []
        for index, segment in enumerate(request.segments):
            start_time = _clip_time(segment.start_time, 0.0, duration_seconds)
            end_time = _clip_time(segment.end_time, 0.0, duration_seconds)
            if end_time <= start_time:
                continue
            segments.append(
                {
                    "segment_id": segment.segment_id or f"segment_{index:03d}",
                    "start_time": round(start_time, 6),
                    "end_time": round(end_time, 6),
                    "label": segment.label,
                    "channel": segment.channel,
                }
            )
        if segments:
            return segments
    return [
        {
            "segment_id": "segment_000",
            "start_time": 0.0,
            "end_time": round(duration_seconds, 6),
            "label": "full_upload"
            if request.analysis_target != "segment"
            else "input_segment",
            "channel": None,
        }
    ]


def _weighted_mean(
    segment_results: list[dict[str, Any]], getter: callable
) -> float | None:
    weighted_values: list[float] = []
    weights: list[float] = []
    for result in segment_results:
        duration = float(result["quality_metrics"]["duration_seconds"])
        value = getter(result)
        if value is None:
            continue
        weighted_values.append(float(value) * duration)
        weights.append(duration)
    if not weights:
        return None
    return round(sum(weighted_values) / sum(weights), 6)


def build_file_summary(
    segment_results: list[dict[str, Any]],
    duration_seconds: float,
    sample_rate: int,
    channel_count: int,
) -> FileSummary:
    analyzed_duration_seconds = round(
        sum(
            float(item["quality_metrics"]["duration_seconds"])
            for item in segment_results
        ),
        6,
    )
    reason_counter = Counter(
        code for item in segment_results for code in item["confidence"]["reason_codes"]
    )
    top_reason_codes = [code for code, _ in reason_counter.most_common(5)]

    return FileSummary(
        duration_seconds=round(duration_seconds, 6),
        sample_rate_hz=sample_rate,
        channel_count=channel_count,
        segment_count=len(segment_results),
        analyzed_duration_seconds=analyzed_duration_seconds,
        quality_metrics_summary={
            "mean_speech_ratio": _weighted_mean(
                segment_results, lambda item: item["quality_metrics"]["speech_ratio"]
            ),
            "mean_silence_ratio": _weighted_mean(
                segment_results, lambda item: item["quality_metrics"]["silence_ratio"]
            ),
            "mean_blind_snr_estimate_db": _weighted_mean(
                segment_results,
                lambda item: item["quality_metrics"]["blind_snr_estimate_db"],
            ),
            "mean_clipping_fraction": _weighted_mean(
                segment_results,
                lambda item: item["quality_metrics"]["clipping_fraction"],
            ),
        },
        quality_assessment_summary={
            "mean_overlap_risk": _weighted_mean(
                segment_results,
                lambda item: item["quality_assessment"]["overlap_risk"]["score"],
            ),
            "mean_boundary_proximity_risk": _weighted_mean(
                segment_results,
                lambda item: item["quality_assessment"]["boundary_proximity_risk"][
                    "score"
                ],
            ),
            "mean_general_usability": _weighted_mean(
                segment_results,
                lambda item: item["quality_assessment"][
                    "usability_for_general_downstream"
                ]["score"],
            ),
        },
        confidence_summary={
            "overall_confidence_mean": _weighted_mean(
                segment_results, lambda item: item["confidence"]["overall_confidence"]
            ),
            "segment_integrity_mean": _weighted_mean(
                segment_results, lambda item: item["confidence"]["segment_integrity"]
            ),
            "speech_usability_for_asr_mean": _weighted_mean(
                segment_results,
                lambda item: item["confidence"]["speech_usability_for_asr"],
            ),
            "top_reason_codes": top_reason_codes,
        },
    )


def analyze_audio(file_bytes: bytes, request: AnalyzeRequest) -> AnalyzeResponse:
    audio, sample_rate = load_audio_from_bytes(file_bytes)
    channel_count = int(audio.shape[1])
    duration_seconds = float(audio.shape[0]) / float(sample_rate)
    segments = _default_segments(request, duration_seconds)
    analysis_target = request.analysis_target or "file"

    upstream = request.upstream or type("EmptyUpstream", (), {})()
    diarization = _to_plain_dicts(getattr(upstream, "diarization_segments", []))
    speaker_segments = _to_plain_dicts(getattr(upstream, "speaker_id_segments", []))
    asr_words = _to_plain_dicts(getattr(upstream, "asr_words", []))
    emotion_segments = _to_plain_dicts(getattr(upstream, "emotion_segments", []))
    sed_events = _to_plain_dicts(getattr(upstream, "sed_events", []))
    known_speaker_ids = list(getattr(upstream, "known_speaker_ids", []))
    enhancement = (
        upstream.enhancement.model_dump(mode="json")
        if getattr(upstream, "enhancement", None) is not None
        else None
    )

    segment_results: list[dict[str, Any]] = []
    for segment in segments:
        start_sample = int(round(float(segment["start_time"]) * sample_rate))
        end_sample = int(round(float(segment["end_time"]) * sample_rate))
        if end_sample <= start_sample:
            continue
        channel = segment.get("channel")
        if channel is not None:
            if channel >= channel_count:
                raise ValueError(
                    f"Segment {segment['segment_id']} requested channel {channel}, "
                    f"but the upload has only {channel_count} channel(s)."
                )
            segment_audio = audio[start_sample:end_sample, channel].astype(np.float32)
        else:
            segment_audio = np.mean(audio[start_sample:end_sample], axis=1).astype(
                np.float32
            )
        metrics = compute_quality_metrics(
            signal=segment_audio,
            sample_rate=sample_rate,
            channels_original=channel_count,
            clip_threshold=request.options.clip_threshold,
        )
        public_metrics = {
            key: value for key, value in metrics.items() if not key.startswith("edge_")
        }
        segment_context = {
            "start_time": float(segment["start_time"]),
            "end_time": float(segment["end_time"]),
            "diarization_segments": _overlapping_items(
                diarization, float(segment["start_time"]), float(segment["end_time"])
            ),
            "speaker_id_segments": _overlapping_items(
                speaker_segments,
                float(segment["start_time"]),
                float(segment["end_time"]),
            ),
            "asr_words": _overlapping_items(
                asr_words, float(segment["start_time"]), float(segment["end_time"])
            ),
            "emotion_segments": _overlapping_items(
                emotion_segments,
                float(segment["start_time"]),
                float(segment["end_time"]),
            ),
            "sed_events": _overlapping_items(
                sed_events, float(segment["start_time"]), float(segment["end_time"])
            ),
            "known_speaker_ids": known_speaker_ids,
            "enhancement": enhancement,
        }
        assessment = build_quality_assessment(
            metrics=metrics,
            segment_context=segment_context,
            boundary_margin=request.options.boundary_margin_seconds,
        )
        confidence = build_confidence(
            metrics=metrics, assessment=assessment, segment_context=segment_context
        )
        segment_result = SegmentResult(
            segment_id=segment["segment_id"],
            start_time=segment["start_time"],
            end_time=segment["end_time"],
            label=segment["label"],
            quality_metrics=public_metrics,
            quality_assessment=assessment,
            confidence=confidence,
        )
        segment_results.append(segment_result.model_dump(mode="json"))

    file_summary = None
    if request.options.include_file_summary:
        file_summary = build_file_summary(
            segment_results=segment_results,
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            channel_count=channel_count,
        )

    response = AnalyzeResponse(
        request_id=request.request_id,
        analysis_target=analysis_target,
        file_summary=file_summary,
        segments=[SegmentResult.model_validate(item) for item in segment_results],
    )
    return response
