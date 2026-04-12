from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .schemas import (
    ASRFileMetadata,
    ASRSegmentMetadata,
    ContextWindow,
    DiarizationFileMetadata,
    DiarizationSegmentMetadata,
    EmotionAttributesFileMetadata,
    EmotionAttributesSegmentMetadata,
    EmotionCategoricalFileMetadata,
    EmotionCategoricalSegmentMetadata,
    GeneralContextFileMetadata,
    GeneralContextSegmentMetadata,
    SEDFileMetadata,
    SEDSegmentMetadata,
    SegmentSpec,
    SoundEvent,
    SpellingCorrectionFileMetadata,
    SpellingCorrectionSegmentMetadata,
    TitanetFileMetadata,
    TitanetSegmentMetadata,
    UpstreamFilePayload,
    UpstreamSegmentPayload,
)


def _overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _weighted_average_dict(items: Iterable[tuple[dict[str, float], float]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    weights: dict[str, float] = {}
    for values, weight in items:
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value * weight
            weights[key] = weights.get(key, 0.0) + weight
    return {key: totals[key] / max(weights[key], 1e-9) for key in totals}


@dataclass(slots=True)
class ResolvedSegmentUpstream:
    asr: ASRSegmentMetadata | None = None
    diarization: DiarizationSegmentMetadata | None = None
    titanet: TitanetSegmentMetadata | None = None
    emotion_categorical: EmotionCategoricalSegmentMetadata | None = None
    emotion_attributes: EmotionAttributesSegmentMetadata | None = None
    sed: SEDSegmentMetadata | None = None
    spelling: SpellingCorrectionSegmentMetadata | None = None
    context: GeneralContextSegmentMetadata | None = None
    enhancement_provided: bool = False
    internal_boundary_count: int | None = None


def _resolve_asr(
    segment: SegmentSpec,
    file_level: ASRFileMetadata | None,
    direct: ASRSegmentMetadata | None,
) -> ASRSegmentMetadata | None:
    if direct:
        return direct
    if not file_level:
        return None

    tokens = [
        token
        for token in file_level.tokens
        if _overlap(segment.start, segment.end, token.start, token.end) > 0.0
    ]
    utterances = [
        utterance
        for utterance in file_level.utterances
        if _overlap(segment.start, segment.end, utterance.start, utterance.end) > 0.0
    ]
    transcript = " ".join(item.text for item in utterances).strip()
    if not transcript and tokens:
        transcript = " ".join(token.text for token in tokens).strip()
    if not transcript and segment.segment_id == "full_file":
        transcript = file_level.transcript
    avg_conf = None
    confidences = [token.confidence for token in tokens if token.confidence is not None]
    if confidences:
        avg_conf = float(np.mean(confidences))
    if not transcript and not tokens:
        return None
    return ASRSegmentMetadata(transcript=transcript or None, tokens=tokens, avg_token_confidence=avg_conf)


def _resolve_diarization(
    segment: SegmentSpec,
    file_level: DiarizationFileMetadata | None,
    direct: DiarizationSegmentMetadata | None,
) -> tuple[DiarizationSegmentMetadata | None, int | None]:
    if direct:
        return direct, None
    if not file_level:
        return None, None
    overlaps = []
    speaker_durations: dict[str, float] = {}
    boundaries: list[float] = []
    for turn in file_level.turns:
        duration = _overlap(segment.start, segment.end, turn.start, turn.end)
        if duration <= 0.0:
            continue
        overlaps.append((turn, duration))
        if turn.speaker_id:
            speaker_durations[turn.speaker_id] = speaker_durations.get(turn.speaker_id, 0.0) + duration
        boundaries.extend([turn.start, turn.end])

    internal_boundary_count = 0
    nearest_boundary = None
    if boundaries:
        internal_boundary_count = sum(1 for boundary in boundaries if segment.start < boundary < segment.end)
        nearest_boundary = min(
            min(abs(boundary - segment.start), abs(boundary - segment.end)) for boundary in boundaries
        )

    if not overlaps:
        return None, internal_boundary_count
    dominant_speaker = None
    if speaker_durations:
        dominant_speaker = max(speaker_durations.items(), key=lambda item: item[1])[0]
    overlap_fraction = float(
        sum(duration for turn, duration in overlaps if turn.overlap) / max(segment.end - segment.start, 1e-9)
    )
    return (
        DiarizationSegmentMetadata(
            dominant_speaker_id=dominant_speaker,
            nearest_boundary_distance_s=nearest_boundary,
            overlap_fraction=overlap_fraction,
            overlap_flag=overlap_fraction > 0.0,
        ),
        internal_boundary_count,
    )


def _resolve_titanet(
    segment: SegmentSpec,
    file_level: TitanetFileMetadata | None,
    direct: TitanetSegmentMetadata | None,
) -> TitanetSegmentMetadata | None:
    if direct:
        return direct
    if not file_level:
        return None
    speaker_weights: dict[str, float] = {}
    speaker_confidences: dict[str, list[float]] = {}
    for window in file_level.windows:
        duration = _overlap(segment.start, segment.end, window.start, window.end)
        if duration <= 0.0:
            continue
        speaker_weights[window.speaker_id] = speaker_weights.get(window.speaker_id, 0.0) + duration * window.confidence
        speaker_confidences.setdefault(window.speaker_id, []).append(window.confidence)
    if not speaker_weights:
        return None
    speaker_id = max(speaker_weights.items(), key=lambda item: item[1])[0]
    confidence = float(np.mean(speaker_confidences[speaker_id]))
    alternatives = [
        {"speaker_id": other_id, "confidence": float(np.mean(confs))}
        for other_id, confs in speaker_confidences.items()
        if other_id != speaker_id
    ]
    return TitanetSegmentMetadata(speaker_id=speaker_id, confidence=confidence, alternatives=alternatives)


def _resolve_emotion_categorical(
    segment: SegmentSpec,
    file_level: EmotionCategoricalFileMetadata | None,
    direct: EmotionCategoricalSegmentMetadata | None,
) -> EmotionCategoricalSegmentMetadata | None:
    if direct:
        return direct
    if not file_level:
        return None
    items = []
    labels: list[str] = []
    for window in file_level.windows:
        duration = _overlap(segment.start, segment.end, window.start, window.end)
        if duration <= 0.0:
            continue
        items.append((window.label_scores, duration))
        if window.top_label:
            labels.append(window.top_label)
    if not items:
        return None
    label_scores = _weighted_average_dict(items)
    top_label = max(label_scores.items(), key=lambda item: item[1])[0] if label_scores else None
    return EmotionCategoricalSegmentMetadata(
        top_label=top_label,
        label_scores=label_scores,
        adjacent_window_labels=labels,
    )


def _resolve_emotion_attributes(
    segment: SegmentSpec,
    file_level: EmotionAttributesFileMetadata | None,
    direct: EmotionAttributesSegmentMetadata | None,
) -> EmotionAttributesSegmentMetadata | None:
    if direct:
        return direct
    if not file_level:
        return None
    items = []
    adjacent = []
    for window in file_level.windows:
        duration = _overlap(segment.start, segment.end, window.start, window.end)
        if duration <= 0.0:
            continue
        items.append((window.attributes, duration))
        adjacent.append(window.attributes)
    if not items:
        return None
    attrs = _weighted_average_dict(items)
    return EmotionAttributesSegmentMetadata(
        attributes=attrs,
        adjacent_window_attributes=adjacent,
    )


def _resolve_sed(
    segment: SegmentSpec,
    file_level: SEDFileMetadata | None,
    direct: SEDSegmentMetadata | None,
) -> SEDSegmentMetadata | None:
    if direct:
        return direct
    if not file_level:
        return None
    events: list[SoundEvent] = []
    label_scores: dict[str, float] = {}
    for event in file_level.events:
        duration = _overlap(segment.start, segment.end, event.start, event.end)
        if duration <= 0.0:
            continue
        events.append(event)
        label_scores[event.label] = label_scores.get(event.label, 0.0) + duration * event.score
    if not label_scores:
        return None
    dominant_label, weighted_score = max(label_scores.items(), key=lambda item: item[1])
    return SEDSegmentMetadata(
        dominant_label=dominant_label,
        dominant_score=float(weighted_score / max(segment.end - segment.start, 1e-9)),
        events=events,
    )


def _resolve_spelling(
    segment: SegmentSpec,
    file_level: SpellingCorrectionFileMetadata | None,
    direct: SpellingCorrectionSegmentMetadata | None,
) -> SpellingCorrectionSegmentMetadata | None:
    if direct:
        return direct
    if not file_level:
        return None
    for item in file_level.segments:
        if _overlap(segment.start, segment.end, item.start, item.end) > 0.0:
            return SpellingCorrectionSegmentMetadata(
                raw_text=item.raw_text,
                corrected_text=item.corrected_text,
                edit_count=item.edit_count,
            )
    if segment.segment_id == "full_file" and (file_level.raw_text or file_level.corrected_text):
        return SpellingCorrectionSegmentMetadata(
            raw_text=file_level.raw_text,
            corrected_text=file_level.corrected_text,
        )
    return None


def _resolve_context(
    segment: SegmentSpec,
    file_level: GeneralContextFileMetadata | None,
    direct: GeneralContextSegmentMetadata | None,
) -> GeneralContextSegmentMetadata | None:
    if direct:
        return direct
    if not file_level:
        return None
    windows: list[ContextWindow] = []
    items = []
    for window in file_level.windows:
        duration = _overlap(segment.start, segment.end, window.start, window.end)
        if duration <= 0.0:
            continue
        windows.append(window)
        items.append(({"__confidence__": window.confidence or 0.5}, duration))
    if not windows:
        return None
    scene_weights: dict[str, float] = {}
    tag_weights: dict[str, float] = {}
    for window in windows:
        weight = max(_overlap(segment.start, segment.end, window.start, window.end), 1e-6)
        if window.scene:
            scene_weights[window.scene] = scene_weights.get(window.scene, 0.0) + weight * (window.confidence or 0.5)
        for tag in window.tags:
            tag_weights[tag] = tag_weights.get(tag, 0.0) + weight
    scene = max(scene_weights.items(), key=lambda item: item[1])[0] if scene_weights else None
    tags = [tag for tag, _weight in sorted(tag_weights.items(), key=lambda item: item[1], reverse=True)[:5]]
    confidence_values = [window.confidence for window in windows if window.confidence is not None]
    confidence = float(np.mean(confidence_values)) if confidence_values else None
    return GeneralContextSegmentMetadata(
        scene=scene,
        tags=tags,
        confidence=confidence,
        adjacent_scenes=[window.scene for window in windows if window.scene],
    )


def resolve_segment_upstream(
    segment: SegmentSpec,
    upstream_file_level: UpstreamFilePayload | None,
    upstream_by_segment: dict[str, UpstreamSegmentPayload],
    enhancement_provided: bool,
) -> ResolvedSegmentUpstream:
    direct = upstream_by_segment.get(segment.segment_id, UpstreamSegmentPayload())
    diarization, internal_boundary_count = _resolve_diarization(
        segment,
        upstream_file_level.diarizen_wavlm_large_s80_md_v2 if upstream_file_level else None,
        direct.diarizen_wavlm_large_s80_md_v2,
    )
    return ResolvedSegmentUpstream(
        asr=_resolve_asr(
            segment,
            upstream_file_level.parakeet_tdt if upstream_file_level else None,
            direct.parakeet_tdt,
        ),
        diarization=diarization,
        titanet=_resolve_titanet(
            segment,
            upstream_file_level.titanet if upstream_file_level else None,
            direct.titanet,
        ),
        emotion_categorical=_resolve_emotion_categorical(
            segment,
            upstream_file_level.emotion_categorical_3loi if upstream_file_level else None,
            direct.emotion_categorical_3loi,
        ),
        emotion_attributes=_resolve_emotion_attributes(
            segment,
            upstream_file_level.emotion_multi_attributes_3loi if upstream_file_level else None,
            direct.emotion_multi_attributes_3loi,
        ),
        sed=_resolve_sed(
            segment,
            upstream_file_level.atst_sed if upstream_file_level else None,
            direct.atst_sed,
        ),
        spelling=_resolve_spelling(
            segment,
            upstream_file_level.qwen3_spelling_correction if upstream_file_level else None,
            direct.qwen3_spelling_correction,
        ),
        context=_resolve_context(
            segment,
            upstream_file_level.qwen2_5_omni if upstream_file_level else None,
            direct.qwen2_5_omni,
        ),
        enhancement_provided=enhancement_provided or direct.mossformergan_se_16k is not None,
        internal_boundary_count=internal_boundary_count,
    )

