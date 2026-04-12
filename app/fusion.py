from __future__ import annotations

from collections import Counter
from typing import Any


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def ratio_score(value: float, good_min: float, good_max: float) -> float:
    if value < good_min:
        return clamp01(value / good_min) if good_min else 0.0
    if value > good_max:
        width = max(1e-6, 1.0 - good_max)
        return clamp01(1.0 - ((value - good_max) / width))
    return 1.0


def banded_label(score: float | None, meaning: str) -> str | None:
    if score is None:
        return None
    if meaning in {"risk", "level"}:
        if score >= 0.7:
            return "high"
        if score >= 0.4:
            return "moderate"
        return "low"
    if score >= 0.85:
        return "excellent"
    if score >= 0.65:
        return "good"
    if score >= 0.4:
        return "fair"
    return "poor"


def overlap_duration(
    segment_start: float, segment_end: float, item_start: float, item_end: float
) -> float:
    return max(0.0, min(segment_end, item_end) - max(segment_start, item_start))


def average_confidence(
    items: list[dict[str, Any]], key: str = "confidence"
) -> float | None:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def build_quality_assessment(
    metrics: dict[str, Any],
    segment_context: dict[str, Any],
    boundary_margin: float,
) -> dict[str, Any]:
    duration = float(metrics["duration_seconds"])
    speech_ratio = float(metrics["speech_ratio"])
    silence_ratio = float(metrics["silence_ratio"])
    snr = float(metrics["blind_snr_estimate_db"])
    clipping_fraction = float(metrics["clipping_fraction"])
    dynamic_range = float(metrics["dynamic_range_db"])
    edge_start = float(metrics["edge_speech_start_ratio"])
    edge_end = float(metrics["edge_speech_end_ratio"])
    spectral_flatness = float(metrics["spectral_flatness"])
    high_band_ratio = float(metrics["high_band_energy_ratio"])

    diarization = segment_context["diarization_segments"]
    speaker_segments = segment_context["speaker_id_segments"]
    asr_words = segment_context["asr_words"]
    emotion_segments = segment_context["emotion_segments"]
    sed_events = segment_context["sed_events"]
    enhancement = segment_context["enhancement"]
    start_time = segment_context["start_time"]
    end_time = segment_context["end_time"]

    overlap_evidence: list[str] = []
    overlap_ratio = 0.0
    diarization_overlap = [item for item in diarization if item.get("overlap") is True]
    if diarization_overlap:
        overlap_ratio = sum(
            overlap_duration(start_time, end_time, item["start_time"], item["end_time"])
            for item in diarization_overlap
        ) / max(duration, 1e-6)
        overlap_evidence.append("diarization_overlap_flags")

    interfering_events = {
        "music",
        "crowd",
        "vehicle",
        "engine",
        "television",
        "alarm",
        "dog",
        "noise",
    }
    interfering_score = 0.0
    if sed_events:
        interfering_score = max(
            (
                float(event.get("score", 0.5))
                for event in sed_events
                if event.get("event_label", "").lower() in interfering_events
            ),
            default=0.0,
        )
        if interfering_score > 0.0:
            overlap_evidence.append("sed_interfering_events")

    heuristic_overlap = clamp01(
        0.12
        + max(0.0, high_band_ratio - 0.18) * 0.9
        + max(0.0, spectral_flatness - 0.28) * 0.6
    )
    if not overlap_evidence:
        overlap_evidence.append("audio_only_heuristic")
    overlap_risk = clamp01(
        max(overlap_ratio * 0.95, interfering_score * 0.55, heuristic_overlap * 0.45)
    )

    boundary_evidence: list[str] = []
    boundary_risk = clamp01(max(edge_start, edge_end) * 0.45)
    if edge_start > 0.3 or edge_end > 0.3:
        boundary_evidence.append("edge_speech_activity")
    if duration < 1.0:
        boundary_risk = clamp01(boundary_risk + 0.18)
        boundary_evidence.append("short_segment")

    boundary_hits = 0
    for item in [*asr_words, *diarization, *emotion_segments]:
        item_start = float(item["start_time"])
        item_end = float(item["end_time"])
        if (
            abs(item_start - start_time) <= boundary_margin
            or abs(item_end - end_time) <= boundary_margin
        ):
            boundary_hits += 1
    if boundary_hits:
        boundary_risk = clamp01(boundary_risk + min(0.35, 0.08 * boundary_hits))
        boundary_evidence.append("upstream_boundary_near_segment_edge")

    speech_dominance = clamp01(speech_ratio)
    non_speech_dominance = clamp01(max(silence_ratio, 1.0 - speech_ratio))

    speaker_consistency_score: float | None = None
    speaker_evidence: list[str] = []
    speaker_note: str | None = None
    if speaker_segments:
        weighted_durations: Counter[str] = Counter()
        total_weight = 0.0
        for item in speaker_segments:
            confidence = float(item.get("confidence") or 0.6)
            span = overlap_duration(
                start_time, end_time, item["start_time"], item["end_time"]
            )
            if span <= 0.0:
                continue
            weight = span * confidence
            weighted_durations[item["speaker_id"]] += weight
            total_weight += weight
        if total_weight > 0.0:
            dominant_weight = weighted_durations.most_common(1)[0][1]
            distinct_speakers = len(weighted_durations)
            speaker_consistency_score = clamp01(
                (dominant_weight / total_weight) - max(0.0, distinct_speakers - 1) * 0.1
            )
            speaker_evidence.append("speaker_id_segments")
            if segment_context["known_speaker_ids"]:
                speaker_evidence.append("known_speaker_ids")
        else:
            speaker_note = "Speaker metadata overlapped the segment only marginally."
    elif diarization and any(item.get("speaker_id") for item in diarization):
        durations: Counter[str] = Counter()
        for item in diarization:
            if not item.get("speaker_id"):
                continue
            durations[item["speaker_id"]] += overlap_duration(
                start_time, end_time, item["start_time"], item["end_time"]
            )
        if durations:
            dominant = durations.most_common(1)[0][1]
            total = sum(durations.values())
            speaker_consistency_score = clamp01(dominant / max(total, 1e-6))
            speaker_evidence.append("diarization_speaker_labels")
        else:
            speaker_note = "No labeled speaker identity overlapped the segment."
    else:
        speaker_note = "Speaker consistency is unavailable without diarization or speaker-id metadata."

    instability_score: float | None = None
    instability_evidence: list[str] = []
    instability_note: str | None = None
    if emotion_segments:
        labels = [item.get("label") or "unknown" for item in emotion_segments]
        changes = sum(
            1 for index in range(1, len(labels)) if labels[index] != labels[index - 1]
        )
        uncertainty = average_confidence(
            [
                {"confidence": 1.0 - float(item.get("uncertainty", 0.0))}
                for item in emotion_segments
            ],
        )
        uncertainty_penalty = 0.0 if uncertainty is None else 1.0 - uncertainty
        instability_score = clamp01(
            (changes / max(1, len(labels) - 1)) * 0.6 + uncertainty_penalty * 0.4
        )
        instability_evidence.append("emotion_segments")
    if asr_words:
        proxy_values = [
            float(item.get("confidence_proxy") or item.get("confidence"))
            for item in asr_words
            if item.get("confidence_proxy") is not None
            or item.get("confidence") is not None
        ]
        if proxy_values:
            low_proxy_penalty = (
                max(0.0, 0.75 - (sum(proxy_values) / len(proxy_values))) / 0.75
            )
            word_durations = [
                max(0.0, float(item["end_time"]) - float(item["start_time"]))
                for item in asr_words
            ]
            duration_spread = 0.0
            if word_durations:
                mean_duration = sum(word_durations) / len(word_durations)
                duration_spread = (
                    (
                        sum((value - mean_duration) ** 2 for value in word_durations)
                        / len(word_durations)
                    )
                    ** 0.5
                ) / max(mean_duration, 1e-6)
            asr_instability = clamp01(
                low_proxy_penalty * 0.7 + min(duration_spread, 1.0) * 0.3
            )
            instability_score = (
                asr_instability
                if instability_score is None
                else max(instability_score, asr_instability)
            )
            instability_evidence.append("asr_word_confidence_proxy")
    if instability_score is None:
        instability_note = "Neighboring-window instability is unavailable without emotion or ASR metadata."

    snr_score = clamp01((snr - 3.0) / 17.0)
    clipping_score = clamp01(1.0 - min(1.0, clipping_fraction / 0.02))
    boundary_score = clamp01(1.0 - boundary_risk)
    overlap_score = clamp01(1.0 - overlap_risk)
    speech_presence_score = ratio_score(speech_ratio, 0.35, 0.95)
    dynamic_range_score = clamp01(dynamic_range / 24.0)
    enhancement_artifacts = float((enhancement or {}).get("artifacts_risk") or 0.0)
    enhancement_score = clamp01(1.0 - enhancement_artifacts)

    asr_proxy = average_confidence(
        [
            {"confidence": item.get("confidence_proxy") or item.get("confidence")}
            for item in asr_words
        ]
    )
    emotion_stability_score = (
        0.6 if instability_score is None else clamp01(1.0 - instability_score)
    )
    speaker_support_score = (
        0.7 if speaker_consistency_score is None else speaker_consistency_score
    )
    duration_for_emotion = clamp01(
        min(duration / 3.0, 1.0) * min(1.0, 20.0 / max(duration, 1.0))
    )
    duration_for_speaker = clamp01(min(duration / 2.0, 1.0))

    usability_for_asr = clamp01(
        0.27 * speech_presence_score
        + 0.22 * snr_score
        + 0.16 * overlap_score
        + 0.12 * clipping_score
        + 0.10 * boundary_score
        + 0.07 * dynamic_range_score
        + 0.06 * (0.65 if asr_proxy is None else asr_proxy)
    )
    usability_for_emotion = clamp01(
        0.24 * speech_presence_score
        + 0.18 * snr_score
        + 0.16 * duration_for_emotion
        + 0.14 * emotion_stability_score
        + 0.10 * clipping_score
        + 0.10 * overlap_score
        + 0.08 * enhancement_score
    )
    usability_for_speaker_id = clamp01(
        0.25 * speech_presence_score
        + 0.21 * snr_score
        + 0.20 * overlap_score
        + 0.14 * duration_for_speaker
        + 0.12 * speaker_support_score
        + 0.08 * clipping_score
    )
    usability_for_general = clamp01(
        0.30 * speech_presence_score
        + 0.20 * snr_score
        + 0.15 * clipping_score
        + 0.15 * boundary_score
        + 0.10 * overlap_score
        + 0.10 * enhancement_score
    )

    return {
        "overlap_risk": {
            "score": round(overlap_risk, 6),
            "label": banded_label(overlap_risk, "risk"),
            "evidence": overlap_evidence,
        },
        "boundary_proximity_risk": {
            "score": round(boundary_risk, 6),
            "label": banded_label(boundary_risk, "risk"),
            "evidence": boundary_evidence,
        },
        "speech_dominance": {
            "score": round(speech_dominance, 6),
            "label": banded_label(speech_dominance, "level"),
            "evidence": ["direct_audio_measurement"],
        },
        "non_speech_dominance": {
            "score": round(non_speech_dominance, 6),
            "label": banded_label(non_speech_dominance, "level"),
            "evidence": ["direct_audio_measurement"],
        },
        "speaker_consistency": {
            "score": None
            if speaker_consistency_score is None
            else round(speaker_consistency_score, 6),
            "label": banded_label(speaker_consistency_score, "score"),
            "evidence": speaker_evidence,
            "note": speaker_note,
        },
        "neighboring_window_instability": {
            "score": None if instability_score is None else round(instability_score, 6),
            "label": banded_label(instability_score, "risk")
            if instability_score is not None
            else None,
            "evidence": instability_evidence,
            "note": instability_note,
        },
        "usability_for_asr": {
            "score": round(usability_for_asr, 6),
            "label": banded_label(usability_for_asr, "score"),
            "evidence": ["audio_metrics"]
            + (["asr_word_confidence_proxy"] if asr_proxy is not None else []),
        },
        "usability_for_emotion": {
            "score": round(usability_for_emotion, 6),
            "label": banded_label(usability_for_emotion, "score"),
            "evidence": ["audio_metrics"]
            + (["emotion_segments"] if emotion_segments else []),
        },
        "usability_for_speaker_id": {
            "score": round(usability_for_speaker_id, 6),
            "label": banded_label(usability_for_speaker_id, "score"),
            "evidence": ["audio_metrics"]
            + (
                ["speaker_identity_metadata"]
                if speaker_consistency_score is not None
                else []
            ),
        },
        "usability_for_general_downstream": {
            "score": round(usability_for_general, 6),
            "label": banded_label(usability_for_general, "score"),
            "evidence": ["audio_metrics", "rules_based_fusion"],
        },
    }


def build_confidence(
    metrics: dict[str, Any],
    assessment: dict[str, Any],
    segment_context: dict[str, Any],
) -> dict[str, Any]:
    duration = float(metrics["duration_seconds"])
    snr = float(metrics["blind_snr_estimate_db"])
    speech_ratio = float(metrics["speech_ratio"])
    clipping_fraction = float(metrics["clipping_fraction"])
    overlap_risk = float(assessment["overlap_risk"]["score"] or 0.0)
    boundary_risk = float(assessment["boundary_proximity_risk"]["score"] or 0.0)
    instability = assessment["neighboring_window_instability"]["score"]
    speaker_consistency = assessment["speaker_consistency"]["score"]
    enhancement = segment_context["enhancement"] or {}
    enhancement_artifacts = float(enhancement.get("artifacts_risk") or 0.0)

    factors: list[dict[str, Any]] = []
    reason_codes: list[str] = []

    def add_factor(
        *,
        code: str,
        name: str,
        direction: str,
        source: str,
        impact: float,
        value: Any,
        explanation: str,
    ) -> None:
        factors.append(
            {
                "name": name,
                "direction": direction,
                "source": source,
                "impact": round(impact, 6),
                "value": value,
                "explanation": explanation,
            }
        )
        if direction != "neutral":
            reason_codes.append(code)

    if speech_ratio >= 0.55:
        add_factor(
            code="strong_speech_coverage",
            name="Speech coverage",
            direction="boost",
            source="audio",
            impact=0.08,
            value=round(speech_ratio, 6),
            explanation="The segment contains a strong share of speech-dominant frames.",
        )
    elif speech_ratio < 0.25:
        add_factor(
            code="limited_speech_content",
            name="Limited speech content",
            direction="penalty",
            source="audio",
            impact=-0.18,
            value=round(speech_ratio, 6),
            explanation="Low speech coverage reduces confidence for speech-centric downstream tasks.",
        )

    if snr >= 14.0:
        add_factor(
            code="high_estimated_snr",
            name="Estimated SNR",
            direction="boost",
            source="audio",
            impact=0.08,
            value=round(snr, 6),
            explanation="The blind SNR estimate indicates relatively clean speech content.",
        )
    elif snr < 6.0:
        add_factor(
            code="low_estimated_snr",
            name="Estimated SNR",
            direction="penalty",
            source="audio",
            impact=-0.22,
            value=round(snr, 6),
            explanation="Low blind SNR reduces intelligibility and downstream reliability.",
        )

    if clipping_fraction > 0.002:
        add_factor(
            code="clipping_detected",
            name="Clipping",
            direction="penalty",
            source="audio",
            impact=-0.2,
            value=round(clipping_fraction, 6),
            explanation="Detected clipping suggests waveform distortion and reduced fidelity.",
        )

    if overlap_risk >= 0.55:
        add_factor(
            code="overlap_risk_high",
            name="Speaker overlap risk",
            direction="penalty",
            source="fusion",
            impact=-0.16,
            value=round(overlap_risk, 6),
            explanation="Likely overlap or interference lowers segment integrity and task usability.",
        )

    if boundary_risk >= 0.55:
        add_factor(
            code="boundary_truncation_risk",
            name="Boundary risk",
            direction="penalty",
            source="fusion",
            impact=-0.14,
            value=round(boundary_risk, 6),
            explanation="Active speech close to the segment edge raises truncation risk.",
        )

    if duration < 0.8:
        add_factor(
            code="segment_too_short",
            name="Short duration",
            direction="penalty",
            source="audio",
            impact=-0.15,
            value=round(duration, 6),
            explanation="Very short segments provide limited evidence for robust quality estimation.",
        )

    if speaker_consistency is not None and speaker_consistency >= 0.8:
        add_factor(
            code="speaker_identity_stable",
            name="Speaker consistency",
            direction="boost",
            source="speaker_id",
            impact=0.06,
            value=round(float(speaker_consistency), 6),
            explanation="Upstream speaker metadata indicates stable identity within the segment.",
        )
    elif speaker_consistency is not None and speaker_consistency < 0.45:
        add_factor(
            code="speaker_identity_unstable",
            name="Speaker consistency",
            direction="penalty",
            source="speaker_id",
            impact=-0.12,
            value=round(float(speaker_consistency), 6),
            explanation="Competing speaker labels reduce trust in speaker-specific downstream use.",
        )
    else:
        add_factor(
            code="speaker_metadata_absent",
            name="Speaker metadata availability",
            direction="neutral",
            source="speaker_id",
            impact=0.0,
            value=None,
            explanation="Speaker identity metadata did not contribute a strong boost or penalty for this segment.",
        )

    if instability is not None and instability >= 0.55:
        add_factor(
            code="neighboring_instability_high",
            name="Neighboring-window instability",
            direction="penalty",
            source="asr_or_emotion",
            impact=-0.11,
            value=round(float(instability), 6),
            explanation="Unstable adjacent emotion or ASR windows reduce confidence in higher-level interpretation.",
        )

    if enhancement_artifacts >= 0.45:
        add_factor(
            code="enhancement_artifacts_risk",
            name="Enhancement artifacts",
            direction="penalty",
            source="enhancement",
            impact=-0.08,
            value=round(enhancement_artifacts, 6),
            explanation="Upstream enhancement metadata reports a meaningful artifact risk.",
        )

    base_integrity = 0.82
    segment_integrity = clamp01(base_integrity + sum(f["impact"] for f in factors))
    speech_usability_for_asr = clamp01(float(assessment["usability_for_asr"]["score"]))
    speech_usability_for_emotion = clamp01(
        float(assessment["usability_for_emotion"]["score"])
    )
    speech_usability_for_speaker_id = clamp01(
        float(assessment["usability_for_speaker_id"]["score"])
    )
    overall_confidence = clamp01(
        0.30 * segment_integrity
        + 0.24 * speech_usability_for_asr
        + 0.18 * speech_usability_for_emotion
        + 0.18 * speech_usability_for_speaker_id
        + 0.10 * float(assessment["usability_for_general_downstream"]["score"])
    )

    sorted_factors = sorted(factors, key=lambda item: abs(item["impact"]), reverse=True)
    sorted_codes = [
        code
        for code, _ in sorted(
            Counter(reason_codes).items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    return {
        "overall_confidence": round(overall_confidence, 6),
        "segment_integrity": round(segment_integrity, 6),
        "speech_usability_for_asr": round(speech_usability_for_asr, 6),
        "speech_usability_for_emotion": round(speech_usability_for_emotion, 6),
        "speech_usability_for_speaker_id": round(speech_usability_for_speaker_id, 6),
        "reason_codes": sorted_codes,
        "contributing_factors": sorted_factors,
    }
