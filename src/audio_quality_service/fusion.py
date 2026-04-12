from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import Settings
from .quality_backends import DNSMOSScores, NISQAScores
from .schemas import (
    AggregateStats,
    ASRConsistencyAssessment,
    BoundaryRiskAssessment,
    ConfidenceComponent,
    ConfidenceComponents,
    DNSMOSResult,
    EmotionInstabilityAssessment,
    EnhancementDeltaAssessment,
    FileSummary,
    FlagSummary,
    GeneralContextConsistencyAssessment,
    LearnedSpeechQuality,
    NISQAResult,
    OverlapRiskAssessment,
    QualityAssessment,
    QualityMetrics,
    SEDContextDominanceAssessment,
    SegmentIntegrityAssessment,
    SegmentResult,
    SourceFlags,
    SourceGroupFlag,
    SpeakerConsistencyAssessment,
    SpellingCorrectionDeltaAssessment,
    TaskUsability,
)
from .schemas import SegmentSpec
from .upstream import ResolvedSegmentUpstream


@dataclass(slots=True)
class ComponentScore:
    available: bool
    score: float | None
    weight: float
    evidence: list[str]


def clamp01(value: float | None) -> float:
    if value is None:
        return 0.0
    return float(max(0.0, min(1.0, value)))


def normalize(value: float, bad: float, good: float) -> float:
    if good == bad:
        return 0.0
    return clamp01((value - bad) / (good - bad))


def inverse_normalize(value: float, good: float, bad: float) -> float:
    if bad == good:
        return 0.0
    return clamp01((bad - value) / (bad - good))


def average_scores(items: Iterable[tuple[float, float]]) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for score, weight in items:
        weighted_sum += score * weight
        total_weight += weight
    if total_weight <= 0.0:
        return 0.0
    return float(weighted_sum / total_weight)


def density_score(value: float, low: float, high: float) -> float:
    if value <= 0.0:
        return 0.0
    if low <= value <= high:
        return 1.0
    if value < low:
        return clamp01(value / max(low, 1e-9))
    return clamp01(high / value)


def levenshtein_distance(text_a: str, text_b: str) -> int:
    if text_a == text_b:
        return 0
    if not text_a:
        return len(text_b)
    if not text_b:
        return len(text_a)
    prev = list(range(len(text_b) + 1))
    for i, char_a in enumerate(text_a, start=1):
        curr = [i]
        for j, char_b in enumerate(text_b, start=1):
            insert_cost = curr[j - 1] + 1
            delete_cost = prev[j] + 1
            replace_cost = prev[j - 1] + (char_a != char_b)
            curr.append(min(insert_cost, delete_cost, replace_cost))
        prev = curr
    return prev[-1]


def build_learned_speech_quality(
    nisqa_scores: NISQAScores,
    dnsmos_scores: DNSMOSScores | None,
) -> LearnedSpeechQuality:
    return LearnedSpeechQuality(
        nisqa=NISQAResult(
            available=nisqa_scores.mos is not None,
            mos=nisqa_scores.mos,
            noisiness=nisqa_scores.noisiness,
            coloration=nisqa_scores.coloration,
            discontinuity=nisqa_scores.discontinuity,
            loudness=nisqa_scores.loudness,
        ),
        dnsmos=DNSMOSResult(
            available=dnsmos_scores is not None and dnsmos_scores.ovrl is not None,
            ovrl=None if dnsmos_scores is None else dnsmos_scores.ovrl,
            sig=None if dnsmos_scores is None else dnsmos_scores.sig,
            bak=None if dnsmos_scores is None else dnsmos_scores.bak,
            p808_mos=None if dnsmos_scores is None else dnsmos_scores.p808_mos,
        ),
    )


def _audio_quality_component(
    metrics: QualityMetrics,
    learned: LearnedSpeechQuality,
    settings: Settings,
) -> ComponentScore:
    snr_score = normalize(metrics.blind_snr_db, settings.blind_snr_bad_db, settings.blind_snr_good_db)
    speech_score = normalize(metrics.speech_ratio, 0.15, 0.85)
    clipping_score = inverse_normalize(
        metrics.clipping_fraction,
        settings.clipping_warn_fraction,
        settings.clipping_bad_fraction,
    )
    dynamic_score = normalize(metrics.dynamic_range_db, 8.0, 30.0)
    nisqa_score = normalize(learned.nisqa.mos or 3.0, 1.5, 4.5)
    items = [
        (snr_score, 2.0),
        (speech_score, 1.0),
        (clipping_score, 2.0),
        (dynamic_score, 1.0),
        (nisqa_score, 2.0),
    ]
    evidence = [
        f"blind_snr_db={metrics.blind_snr_db:.2f}",
        f"speech_ratio={metrics.speech_ratio:.2f}",
        f"nisqa_mos={learned.nisqa.mos:.2f}" if learned.nisqa.mos is not None else "nisqa_mos=na",
    ]
    if learned.dnsmos.available and learned.dnsmos.ovrl is not None:
        items.append((normalize(learned.dnsmos.ovrl, 1.5, 4.5), 1.0))
        evidence.append(f"dnsmos_ovrl={learned.dnsmos.ovrl:.2f}")
    return ComponentScore(
        available=True,
        score=average_scores(items),
        weight=settings.weight_audio_quality,
        evidence=evidence,
    )


def _boundary_assessment(
    upstream: ResolvedSegmentUpstream,
    settings: Settings,
) -> tuple[BoundaryRiskAssessment, ComponentScore]:
    diar = upstream.diarization
    if diar is None:
        return (
            BoundaryRiskAssessment(available=False),
            ComponentScore(False, None, settings.weight_boundary, ["missing_diarization"]),
        )
    edge_risk = 0.0
    if diar.nearest_boundary_distance_s is not None:
        edge_risk = inverse_normalize(
            diar.nearest_boundary_distance_s,
            settings.boundary_safe_seconds,
            settings.boundary_near_seconds,
        )
    count_risk = clamp01((upstream.internal_boundary_count or 0) / 2.0)
    risk = clamp01(max(edge_risk, count_risk))
    assessment = BoundaryRiskAssessment(
        available=True,
        risk=risk,
        nearest_boundary_distance_s=diar.nearest_boundary_distance_s,
        internal_boundary_count=upstream.internal_boundary_count,
    )
    component = ComponentScore(
        available=True,
        score=1.0 - risk,
        weight=settings.weight_boundary,
        evidence=[
            f"nearest_boundary_distance_s={diar.nearest_boundary_distance_s}",
            f"internal_boundary_count={upstream.internal_boundary_count}",
        ],
    )
    return assessment, component


def _overlap_assessment(
    upstream: ResolvedSegmentUpstream,
    settings: Settings,
) -> tuple[OverlapRiskAssessment, ComponentScore]:
    diar = upstream.diarization
    if diar is None or diar.overlap_fraction is None:
        return (
            OverlapRiskAssessment(available=False),
            ComponentScore(False, None, settings.weight_overlap, ["missing_diarization"]),
        )
    risk = normalize(
        diar.overlap_fraction,
        settings.overlap_warn_fraction,
        settings.overlap_heavy_fraction,
    )
    assessment = OverlapRiskAssessment(
        available=True,
        risk=risk,
        overlap_fraction=diar.overlap_fraction,
    )
    component = ComponentScore(
        available=True,
        score=1.0 - risk,
        weight=settings.weight_overlap,
        evidence=[f"overlap_fraction={diar.overlap_fraction:.3f}"],
    )
    return assessment, component


def _speaker_assessment(
    speaker_id: str | None,
    upstream: ResolvedSegmentUpstream,
    settings: Settings,
) -> tuple[SpeakerConsistencyAssessment, ComponentScore]:
    titanet = upstream.titanet
    if titanet is None:
        return (
            SpeakerConsistencyAssessment(available=False),
            ComponentScore(False, None, settings.weight_speaker, ["missing_titanet"]),
        )
    confidence = titanet.confidence or 0.0
    conf_score = normalize(confidence, settings.speaker_confidence_low, settings.speaker_confidence_high)
    mismatch = bool(speaker_id and titanet.speaker_id and speaker_id != titanet.speaker_id)
    score = clamp01(0.75 * conf_score + 0.25 * (0.0 if mismatch else 1.0))
    assessment = SpeakerConsistencyAssessment(
        available=True,
        score=score,
        dominant_speaker_id=speaker_id,
        predicted_speaker_id=titanet.speaker_id,
        confidence=confidence,
        mismatch=mismatch,
    )
    component = ComponentScore(
        available=True,
        score=score,
        weight=settings.weight_speaker,
        evidence=[
            f"speaker_id={speaker_id}",
            f"predicted={titanet.speaker_id}",
            f"confidence={confidence:.2f}",
        ],
    )
    return assessment, component


def _asr_assessment(
    segment: SegmentSpec,
    upstream: ResolvedSegmentUpstream,
    settings: Settings,
) -> tuple[ASRConsistencyAssessment, ComponentScore]:
    asr = upstream.asr
    if asr is None:
        return (
            ASRConsistencyAssessment(available=False),
            ComponentScore(False, None, settings.weight_asr, ["missing_asr"]),
        )
    transcript = asr.transcript or ""
    token_count = len(asr.tokens) or len(transcript.split())
    avg_conf = asr.avg_token_confidence
    if avg_conf is None:
        confs = [token.confidence for token in asr.tokens if token.confidence is not None]
        avg_conf = float(np.mean(confs)) if confs else None
    sorted_tokens = sorted(asr.tokens, key=lambda token: token.start)
    gaps = []
    for prev, curr in zip(sorted_tokens, sorted_tokens[1:]):
        gap = curr.start - prev.end
        if gap > 0.0:
            gaps.append(gap)
    gap_rate = float(sum(gap > 0.4 for gap in gaps) / max(len(sorted_tokens) - 1, 1)) if sorted_tokens else None
    density = float(token_count / max(segment.end - segment.start, 1e-9))
    items = [(density_score(density, settings.asr_density_low_wps, settings.asr_density_high_wps), 0.2)]
    evidence = [f"transcript_density_wps={density:.2f}"]
    if avg_conf is not None:
        items.append((normalize(avg_conf, settings.asr_confidence_low, settings.asr_confidence_high), 0.5))
        evidence.append(f"avg_token_confidence={avg_conf:.2f}")
    if gap_rate is not None:
        items.append((inverse_normalize(gap_rate, settings.asr_gap_good_rate, settings.asr_gap_bad_rate), 0.3))
        evidence.append(f"gap_rate={gap_rate:.2f}")
    score = average_scores(items)
    assessment = ASRConsistencyAssessment(
        available=True,
        score=score,
        avg_token_confidence=avg_conf,
        gap_rate=gap_rate,
        transcript_density_wps=density,
        token_count=token_count,
    )
    component = ComponentScore(True, score, settings.weight_asr, evidence)
    return assessment, component


def _spelling_assessment(
    upstream: ResolvedSegmentUpstream,
    settings: Settings,
) -> tuple[SpellingCorrectionDeltaAssessment, ComponentScore]:
    spelling = upstream.spelling
    if spelling is None or not spelling.raw_text or not spelling.corrected_text:
        return (
            SpellingCorrectionDeltaAssessment(available=False),
            ComponentScore(False, None, settings.weight_spelling, ["missing_spelling_correction"]),
        )
    edit_count = spelling.edit_count
    if edit_count is None:
        edit_count = levenshtein_distance(spelling.raw_text, spelling.corrected_text)
    denom = max(len(spelling.raw_text.split()), 1)
    normalized_edit_rate = spelling.normalized_edit_rate
    if normalized_edit_rate is None:
        normalized_edit_rate = float(edit_count / denom)
    score = inverse_normalize(normalized_edit_rate, 0.02, settings.spelling_edit_warn_rate)
    assessment = SpellingCorrectionDeltaAssessment(
        available=True,
        score=score,
        edit_count=edit_count,
        normalized_edit_rate=normalized_edit_rate,
    )
    component = ComponentScore(
        True,
        score,
        settings.weight_spelling,
        [f"normalized_edit_rate={normalized_edit_rate:.3f}", f"edit_count={edit_count}"],
    )
    return assessment, component


def _sed_assessment(
    upstream: ResolvedSegmentUpstream,
    settings: Settings,
) -> tuple[SEDContextDominanceAssessment, ComponentScore]:
    sed = upstream.sed
    if sed is None:
        return (
            SEDContextDominanceAssessment(available=False),
            ComponentScore(False, None, settings.weight_sed, ["missing_sed"]),
        )
    dominant = sed.dominant_label
    non_speech_fraction = 0.0
    if sed.events:
        total = 0.0
        non_speech_total = 0.0
        for event in sed.events:
            duration = max(event.end - event.start, 0.0)
            weight = duration * event.score
            total += weight
            label = event.label.lower()
            if not any(token in label for token in settings.speech_like_labels):
                non_speech_total += weight
        if total > 0.0:
            non_speech_fraction = float(non_speech_total / total)
    elif dominant:
        label = dominant.lower()
        non_speech_fraction = 0.0 if any(token in label for token in settings.speech_like_labels) else clamp01(sed.dominant_score or 1.0)
    score = 1.0 - clamp01(non_speech_fraction)
    assessment = SEDContextDominanceAssessment(
        available=True,
        score=score,
        dominant_label=dominant,
        dominant_non_speech_fraction=non_speech_fraction,
    )
    component = ComponentScore(
        True,
        score,
        settings.weight_sed,
        [f"dominant_label={dominant}", f"non_speech_fraction={non_speech_fraction:.2f}"],
    )
    return assessment, component


def _emotion_assessment(
    upstream: ResolvedSegmentUpstream,
    settings: Settings,
) -> tuple[EmotionInstabilityAssessment, ComponentScore]:
    cat = upstream.emotion_categorical
    attrs = upstream.emotion_attributes
    if cat is None and attrs is None:
        return (
            EmotionInstabilityAssessment(available=False),
            ComponentScore(False, None, settings.weight_emotion, ["missing_emotion"]),
        )
    label_instability = 0.0
    dominant_label = None
    if cat is not None:
        labels = [label for label in [cat.top_label, *cat.adjacent_window_labels] if label]
        dominant_label = cat.top_label
        if labels:
            counts: dict[str, int] = {}
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
            label_instability = 1.0 - max(counts.values()) / len(labels)
    attr_instability = 0.0
    if attrs is not None:
        rows = [attrs.attributes, *attrs.adjacent_window_attributes]
        keys = sorted({key for row in rows for key in row})
        if keys:
            stdevs = []
            for key in keys:
                values = [row[key] for row in rows if key in row]
                if len(values) > 1:
                    stdevs.append(float(np.std(values)))
            if stdevs:
                attr_instability = clamp01(float(np.mean(stdevs)) / max(settings.emotion_instability_warn, 1e-9))
    instability = average_scores([(label_instability, 0.5), (attr_instability, 0.5)])
    score = 1.0 - instability
    assessment = EmotionInstabilityAssessment(
        available=True,
        score=score,
        instability=instability,
        dominant_label=dominant_label,
    )
    component = ComponentScore(
        True,
        score,
        settings.weight_emotion,
        [f"instability={instability:.2f}", f"dominant_label={dominant_label}"],
    )
    return assessment, component


def _enhancement_assessment(
    metrics: QualityMetrics,
    learned: LearnedSpeechQuality,
    enhanced_metrics: QualityMetrics | None,
    enhanced_learned: LearnedSpeechQuality | None,
    settings: Settings,
) -> tuple[EnhancementDeltaAssessment, ComponentScore]:
    if enhanced_metrics is None or enhanced_learned is None:
        return (
            EnhancementDeltaAssessment(available=False),
            ComponentScore(False, None, settings.weight_enhancement, ["missing_enhanced_audio"]),
        )
    nisqa_delta = None
    if learned.nisqa.mos is not None and enhanced_learned.nisqa.mos is not None:
        nisqa_delta = float(enhanced_learned.nisqa.mos - learned.nisqa.mos)
    snr_delta = float(enhanced_metrics.blind_snr_db - metrics.blind_snr_db)
    spectral_shift = abs(enhanced_metrics.spectral_centroid_hz - metrics.spectral_centroid_hz) / max(
        metrics.spectral_centroid_hz,
        1.0,
    )
    score = clamp01(
        0.5
        + 0.15 * (nisqa_delta or 0.0)
        + 0.03 * snr_delta
        - max(0.0, spectral_shift - settings.enhancement_shift_warn)
    )
    assessment = EnhancementDeltaAssessment(
        available=True,
        score=score,
        nisqa_mos_delta=nisqa_delta,
        blind_snr_delta_db=snr_delta,
        spectral_shift_ratio=float(spectral_shift),
    )
    component = ComponentScore(
        True,
        score,
        settings.weight_enhancement,
        [
            f"nisqa_mos_delta={nisqa_delta}",
            f"blind_snr_delta_db={snr_delta:.2f}",
            f"spectral_shift_ratio={spectral_shift:.2f}",
        ],
    )
    return assessment, component


def _context_assessment(
    upstream: ResolvedSegmentUpstream,
    settings: Settings,
) -> tuple[GeneralContextConsistencyAssessment, ComponentScore]:
    context = upstream.context
    if context is None:
        return (
            GeneralContextConsistencyAssessment(available=False),
            ComponentScore(False, None, settings.weight_context, ["missing_general_context"]),
        )
    scenes = [scene for scene in [context.scene, *context.adjacent_scenes] if scene]
    scene_consistency = 0.5
    if scenes:
        counts: dict[str, int] = {}
        for scene in scenes:
            counts[scene] = counts.get(scene, 0) + 1
        scene_consistency = max(counts.values()) / len(scenes)
    score = average_scores(
        [
            (scene_consistency, 0.7),
            (context.confidence if context.confidence is not None else 0.5, 0.3),
        ]
    )
    assessment = GeneralContextConsistencyAssessment(
        available=True,
        score=score,
        dominant_scene=context.scene,
        scene_consistency=scene_consistency,
    )
    component = ComponentScore(
        True,
        score,
        settings.weight_context,
        [f"scene={context.scene}", f"scene_consistency={scene_consistency:.2f}"],
    )
    return assessment, component


def _integrity_assessment(
    metrics: QualityMetrics,
    settings: Settings,
) -> tuple[SegmentIntegrityAssessment, ComponentScore]:
    duration = metrics.duration_s
    if duration < settings.short_segment_seconds:
        duration_score = clamp01(duration / max(settings.short_segment_seconds, 1e-9))
    elif duration > settings.long_segment_seconds:
        duration_score = clamp01(settings.long_segment_seconds / duration)
    else:
        duration_score = 1.0

    clipping_score = inverse_normalize(
        metrics.clipping_fraction,
        settings.clipping_warn_fraction,
        settings.clipping_bad_fraction,
    )
    speech_presence_score = normalize(metrics.speech_ratio, 0.1, 0.6)
    peak_headroom_score = normalize(-metrics.peak_dbfs, 0.5, 6.0)
    rms_score = normalize(metrics.rms_dbfs, -45.0, -18.0)
    headroom_score = average_scores([(peak_headroom_score, 0.6), (rms_score, 0.4)])
    score = average_scores(
        [
            (duration_score, 0.25),
            (clipping_score, 0.25),
            (speech_presence_score, 0.25),
            (headroom_score, 0.25),
        ]
    )
    assessment = SegmentIntegrityAssessment(
        available=True,
        score=score,
        duration_score=duration_score,
        clipping_score=clipping_score,
        speech_presence_score=speech_presence_score,
        headroom_score=headroom_score,
    )
    component = ComponentScore(
        True,
        score,
        settings.weight_integrity,
        [
            f"duration_score={duration_score:.2f}",
            f"clipping_score={clipping_score:.2f}",
            f"speech_presence_score={speech_presence_score:.2f}",
            f"headroom_score={headroom_score:.2f}",
        ],
    )
    return assessment, component


def _component_to_model(component: ComponentScore) -> ConfidenceComponent:
    contribution = component.weight * (component.score if component.available and component.score is not None else 0.0)
    return ConfidenceComponent(
        available=component.available,
        score=component.score,
        weight=component.weight,
        contribution=contribution,
        evidence=component.evidence,
    )


def _task_usability(
    components: dict[str, ComponentScore],
) -> TaskUsability:
    def select(*items: tuple[str, float]) -> float:
        weighted = []
        for name, weight in items:
            component = components[name]
            if component.available and component.score is not None:
                weighted.append((component.score, weight))
        return average_scores(weighted)

    return TaskUsability(
        asr=select(
            ("audio_quality", 0.35),
            ("integrity", 0.2),
            ("boundary", 0.1),
            ("overlap", 0.1),
            ("asr", 0.2),
            ("spelling", 0.05),
        ),
        emotion=select(
            ("audio_quality", 0.4),
            ("integrity", 0.2),
            ("emotion", 0.25),
            ("overlap", 0.1),
            ("boundary", 0.05),
        ),
        speaker_id=select(
            ("audio_quality", 0.2),
            ("integrity", 0.2),
            ("speaker", 0.4),
            ("boundary", 0.1),
            ("overlap", 0.1),
        ),
        sed=select(
            ("audio_quality", 0.25),
            ("integrity", 0.15),
            ("sed", 0.6),
        ),
        general_audio_understanding=select(
            ("audio_quality", 0.35),
            ("integrity", 0.25),
            ("context", 0.4),
        ),
    )


def _reason_codes(
    metrics: QualityMetrics,
    boundary: BoundaryRiskAssessment,
    overlap: OverlapRiskAssessment,
    speaker: SpeakerConsistencyAssessment,
    asr: ASRConsistencyAssessment,
    spelling: SpellingCorrectionDeltaAssessment,
    sed: SEDContextDominanceAssessment,
    emotion: EmotionInstabilityAssessment,
    enhancement: EnhancementDeltaAssessment,
    context: GeneralContextConsistencyAssessment,
    settings: Settings,
) -> list[str]:
    penalties: list[tuple[str, float]] = []
    penalties.append(("low_snr", 1.0 - normalize(metrics.blind_snr_db, settings.blind_snr_bad_db, settings.blind_snr_good_db)))
    penalties.append(
        (
            "clipping",
            normalize(metrics.clipping_fraction, settings.clipping_warn_fraction, settings.clipping_bad_fraction),
        )
    )
    if overlap.available and overlap.risk is not None:
        penalties.append(("overlap", overlap.risk))
    if boundary.available and boundary.risk is not None:
        penalties.append(("near_boundary", boundary.risk))
    if speaker.available and speaker.mismatch:
        penalties.append(("speaker_mismatch", 1.0))
    if asr.available:
        if asr.avg_token_confidence is not None:
            penalties.append(("asr_low_confidence", 1.0 - normalize(asr.avg_token_confidence, settings.asr_confidence_low, settings.asr_confidence_high)))
        if asr.gap_rate is not None:
            penalties.append(("high_asr_gap_rate", normalize(asr.gap_rate, settings.asr_gap_good_rate, settings.asr_gap_bad_rate)))
    if spelling.available and spelling.normalized_edit_rate is not None:
        penalties.append(("spelling_delta", normalize(spelling.normalized_edit_rate, 0.02, settings.spelling_edit_warn_rate)))
    if sed.available and sed.dominant_non_speech_fraction is not None:
        penalties.append(("non_speech_dominant", sed.dominant_non_speech_fraction))
    if emotion.available and emotion.instability is not None:
        penalties.append(("emotion_unstable", emotion.instability))
    if enhancement.available and enhancement.score is not None:
        penalties.append(("heavy_enhancement_change", 1.0 - enhancement.score))
    if context.available and context.score is not None:
        penalties.append(("context_mismatch", 1.0 - context.score))

    penalties = [(code, value) for code, value in penalties if value >= 0.2]
    penalties.sort(key=lambda item: item[1], reverse=True)
    reason_codes: list[str] = []
    for code, _value in penalties:
        if code not in reason_codes:
            reason_codes.append(code)
        if len(reason_codes) >= settings.max_reason_codes:
            break
    return reason_codes


def _source_flags(
    speaker_id: str | None,
    upstream: ResolvedSegmentUpstream,
    dnsmos_enabled: bool,
    enhanced_available: bool,
    components: dict[str, ComponentScore],
) -> SourceFlags:
    any_upstream = any(
        item is not None
        for item in (
            upstream.asr,
            upstream.diarization,
            upstream.titanet,
            upstream.emotion_categorical,
            upstream.emotion_attributes,
            upstream.sed,
            upstream.spelling,
            upstream.context,
        )
    )
    return SourceFlags(
        speaker_id=SourceGroupFlag(
            from_raw_audio=False,
            from_upstream_metadata=speaker_id is not None,
            missing_inputs=[] if speaker_id is not None else ["speaker_id"],
        ),
        quality_metrics=SourceGroupFlag(
            from_raw_audio=True,
            from_upstream_metadata=False,
            missing_inputs=[],
        ),
        learned_speech_quality=SourceGroupFlag(
            from_raw_audio=True,
            from_upstream_metadata=False,
            missing_inputs=[] if dnsmos_enabled else ["dnsmos_disabled"],
        ),
        boundary_risk=SourceGroupFlag(
            from_raw_audio=False,
            from_upstream_metadata=upstream.diarization is not None,
            missing_inputs=[] if upstream.diarization else ["diarization"],
        ),
        overlap_risk=SourceGroupFlag(
            from_raw_audio=False,
            from_upstream_metadata=upstream.diarization is not None,
            missing_inputs=[] if upstream.diarization else ["diarization"],
        ),
        speaker_consistency=SourceGroupFlag(
            from_raw_audio=False,
            from_upstream_metadata=upstream.titanet is not None,
            missing_inputs=[] if upstream.titanet else ["titanet"],
        ),
        asr_consistency=SourceGroupFlag(
            from_raw_audio=False,
            from_upstream_metadata=upstream.asr is not None,
            missing_inputs=[] if upstream.asr else ["parakeet_tdt"],
        ),
        spelling_correction_delta=SourceGroupFlag(
            from_raw_audio=False,
            from_upstream_metadata=upstream.spelling is not None,
            missing_inputs=[] if upstream.spelling else ["qwen3_spelling_correction"],
        ),
        sed_context_dominance=SourceGroupFlag(
            from_raw_audio=False,
            from_upstream_metadata=upstream.sed is not None,
            missing_inputs=[] if upstream.sed else ["atst_sed"],
        ),
        emotion_instability=SourceGroupFlag(
            from_raw_audio=False,
            from_upstream_metadata=upstream.emotion_categorical is not None or upstream.emotion_attributes is not None,
            missing_inputs=[]
            if (upstream.emotion_categorical is not None or upstream.emotion_attributes is not None)
            else ["emotion_outputs"],
        ),
        enhancement_delta=SourceGroupFlag(
            from_raw_audio=True,
            from_upstream_metadata=False,
            missing_inputs=[] if enhanced_available else ["enhanced_audio"],
        ),
        general_context_consistency=SourceGroupFlag(
            from_raw_audio=False,
            from_upstream_metadata=upstream.context is not None,
            missing_inputs=[] if upstream.context else ["qwen2_5_omni"],
        ),
        segment_integrity=SourceGroupFlag(
            from_raw_audio=True,
            from_upstream_metadata=False,
            missing_inputs=[],
        ),
        confidence_components=SourceGroupFlag(
            from_raw_audio=True,
            from_upstream_metadata=any_upstream,
            missing_inputs=[
                name
                for name, component in components.items()
                if not component.available and name not in {"audio_quality", "integrity"}
            ],
        ),
        task_usability=SourceGroupFlag(
            from_raw_audio=True,
            from_upstream_metadata=any_upstream,
            missing_inputs=[],
        ),
    )


def score_segment(
    segment: SegmentSpec,
    metrics: QualityMetrics,
    learned: LearnedSpeechQuality,
    upstream: ResolvedSegmentUpstream,
    settings: Settings,
    enhanced_metrics: QualityMetrics | None = None,
    enhanced_learned: LearnedSpeechQuality | None = None,
) -> SegmentResult:
    speaker_id = segment.speaker_id or (
        upstream.titanet.speaker_id
        if upstream.titanet and upstream.titanet.speaker_id
        else upstream.diarization.dominant_speaker_id
        if upstream.diarization and upstream.diarization.dominant_speaker_id
        else None
    )

    audio_component = _audio_quality_component(metrics, learned, settings)
    boundary, boundary_component = _boundary_assessment(upstream, settings)
    overlap, overlap_component = _overlap_assessment(upstream, settings)
    speaker, speaker_component = _speaker_assessment(speaker_id, upstream, settings)
    asr, asr_component = _asr_assessment(segment, upstream, settings)
    spelling, spelling_component = _spelling_assessment(upstream, settings)
    sed, sed_component = _sed_assessment(upstream, settings)
    emotion, emotion_component = _emotion_assessment(upstream, settings)
    enhancement, enhancement_component = _enhancement_assessment(
        metrics,
        learned,
        enhanced_metrics,
        enhanced_learned,
        settings,
    )
    context, context_component = _context_assessment(upstream, settings)
    integrity, integrity_component = _integrity_assessment(metrics, settings)

    component_map = {
        "audio_quality": audio_component,
        "boundary": boundary_component,
        "overlap": overlap_component,
        "speaker": speaker_component,
        "asr": asr_component,
        "spelling": spelling_component,
        "sed": sed_component,
        "emotion": emotion_component,
        "enhancement": enhancement_component,
        "context": context_component,
        "integrity": integrity_component,
    }
    available = [
        (component.score or 0.0, component.weight)
        for component in component_map.values()
        if component.available and component.score is not None
    ]
    overall_confidence = average_scores(available)
    source_flags = _source_flags(
        speaker_id=speaker_id,
        upstream=upstream,
        dnsmos_enabled=learned.dnsmos.available,
        enhanced_available=enhanced_metrics is not None,
        components=component_map,
    )
    quality_assessment = QualityAssessment(
        learned_speech_quality=learned,
        boundary_risk=boundary,
        overlap_risk=overlap,
        speaker_consistency=speaker,
        asr_consistency=asr,
        spelling_correction_delta=spelling,
        sed_context_dominance=sed,
        emotion_instability=emotion,
        enhancement_delta=enhancement,
        general_context_consistency=context,
        segment_integrity=integrity,
        confidence_components=ConfidenceComponents(
            audio_quality=_component_to_model(audio_component),
            boundary=_component_to_model(boundary_component),
            overlap=_component_to_model(overlap_component),
            speaker=_component_to_model(speaker_component),
            asr=_component_to_model(asr_component),
            spelling=_component_to_model(spelling_component),
            sed=_component_to_model(sed_component),
            emotion=_component_to_model(emotion_component),
            enhancement=_component_to_model(enhancement_component),
            context=_component_to_model(context_component),
            integrity=_component_to_model(integrity_component),
        ),
        overall_confidence=overall_confidence,
    )
    task_usability = _task_usability(component_map)
    reason_codes = _reason_codes(
        metrics=metrics,
        boundary=boundary,
        overlap=overlap,
        speaker=speaker,
        asr=asr,
        spelling=spelling,
        sed=sed,
        emotion=emotion,
        enhancement=enhancement,
        context=context,
        settings=settings,
    )

    return SegmentResult(
        segment_id=segment.segment_id,
        start=segment.start,
        end=segment.end,
        speaker_id=speaker_id,
        source_flags=source_flags,
        quality_metrics=metrics,
        quality_assessment=quality_assessment,
        task_usability=task_usability,
        reason_codes=reason_codes,
    )


def build_file_summary(results: list[SegmentResult], settings: Settings) -> FileSummary:
    aggregates: dict[str, AggregateStats] = {}
    metric_extractors = {
        "blind_snr_db": lambda item: item.quality_metrics.blind_snr_db,
        "speech_ratio": lambda item: item.quality_metrics.speech_ratio,
        "clipping_fraction": lambda item: item.quality_metrics.clipping_fraction,
        "nisqa_mos": lambda item: item.quality_assessment.learned_speech_quality.nisqa.mos or 0.0,
        "overall_confidence": lambda item: item.quality_assessment.overall_confidence,
        "task_usability_asr": lambda item: item.task_usability.asr,
        "task_usability_emotion": lambda item: item.task_usability.emotion,
        "task_usability_speaker_id": lambda item: item.task_usability.speaker_id,
        "task_usability_sed": lambda item: item.task_usability.sed,
        "task_usability_general_audio_understanding": lambda item: item.task_usability.general_audio_understanding,
    }
    for name, extractor in metric_extractors.items():
        values = np.array([extractor(result) for result in results], dtype=np.float64)
        aggregates[name] = AggregateStats(
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            p05=float(np.percentile(values, settings.file_summary_percentiles[0])),
            p50=float(np.percentile(values, settings.file_summary_percentiles[1])),
            p95=float(np.percentile(values, settings.file_summary_percentiles[2])),
        )

    def summarize(predicate) -> FlagSummary:
        selected = [result for result in results if predicate(result)]
        return FlagSummary(
            count=len(selected),
            total_duration_s=float(sum(result.end - result.start for result in selected)),
        )

    flags = {
        "low_confidence": summarize(lambda item: item.quality_assessment.overall_confidence < settings.low_confidence_threshold),
        "low_snr": summarize(lambda item: item.quality_metrics.blind_snr_db < settings.blind_snr_bad_db),
        "clipped": summarize(lambda item: item.quality_metrics.clipping_fraction >= settings.clipping_warn_fraction),
        "overlap_heavy": summarize(
            lambda item: item.quality_assessment.overlap_risk.available
            and (item.quality_assessment.overlap_risk.overlap_fraction or 0.0) >= settings.overlap_heavy_fraction
        ),
        "near_boundary": summarize(
            lambda item: item.quality_assessment.boundary_risk.available
            and (item.quality_assessment.boundary_risk.nearest_boundary_distance_s or settings.boundary_safe_seconds)
            <= settings.boundary_near_seconds
        ),
        "asr_poor": summarize(
            lambda item: item.quality_assessment.asr_consistency.available
            and (item.quality_assessment.asr_consistency.score or 1.0) < settings.low_confidence_threshold
        ),
        "non_speech_dominant": summarize(
            lambda item: item.quality_assessment.sed_context_dominance.available
            and (item.quality_assessment.sed_context_dominance.dominant_non_speech_fraction or 0.0)
            >= settings.non_speech_dominance_warn
        ),
        "emotion_unstable": summarize(
            lambda item: item.quality_assessment.emotion_instability.available
            and (item.quality_assessment.emotion_instability.instability or 0.0) >= settings.emotion_instability_warn
        ),
    }

    return FileSummary(
        num_segments=len(results),
        total_duration_s=float(sum(result.end - result.start for result in results)),
        aggregates=aggregates,
        flags=flags,
    )
