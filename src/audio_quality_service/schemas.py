from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class AudioInput(StrictModel):
    path: str | None = Field(default=None, description="Absolute container-visible file path.")
    base64: str | None = Field(
        default=None,
        description="Base64-encoded audio bytes.",
    )
    filename: str | None = Field(
        default=None,
        description="Optional filename used to infer a suffix for base64 inputs.",
    )
    mime_type: str | None = Field(default=None, description="Optional MIME type.")

    @model_validator(mode="after")
    def validate_source(self) -> "AudioInput":
        if bool(self.path) == bool(self.base64):
            raise ValueError("Exactly one of 'path' or 'base64' must be supplied.")
        return self


class SegmentSpec(StrictModel):
    segment_id: str
    start: float = Field(ge=0.0)
    end: float = Field(gt=0.0)
    speaker_id: str | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> "SegmentSpec":
        if self.end <= self.start:
            raise ValueError("'end' must be greater than 'start'.")
        return self


class RollingWindowConfig(StrictModel):
    enabled: bool = True
    window_seconds: float | None = Field(default=None, gt=0.0)
    hop_seconds: float | None = Field(default=None, gt=0.0)


class TimedToken(StrictModel):
    text: str
    start: float = Field(ge=0.0)
    end: float = Field(gt=0.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class TimedTranscriptSpan(StrictModel):
    start: float = Field(ge=0.0)
    end: float = Field(gt=0.0)
    text: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ASRFileMetadata(StrictModel):
    transcript: str | None = None
    tokens: list[TimedToken] = Field(default_factory=list)
    utterances: list[TimedTranscriptSpan] = Field(default_factory=list)


class ASRSegmentMetadata(StrictModel):
    transcript: str | None = None
    tokens: list[TimedToken] = Field(default_factory=list)
    avg_token_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class DiarizationTurn(StrictModel):
    speaker_id: str | None = None
    start: float = Field(ge=0.0)
    end: float = Field(gt=0.0)
    overlap: bool = False


class DiarizationFileMetadata(StrictModel):
    turns: list[DiarizationTurn] = Field(default_factory=list)


class DiarizationSegmentMetadata(StrictModel):
    dominant_speaker_id: str | None = None
    nearest_boundary_distance_s: float | None = Field(default=None, ge=0.0)
    overlap_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    overlap_flag: bool | None = None


class SpeakerCandidate(StrictModel):
    speaker_id: str
    confidence: float = Field(ge=0.0, le=1.0)


class TimedSpeakerPrediction(StrictModel):
    start: float = Field(ge=0.0)
    end: float = Field(gt=0.0)
    speaker_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    alternatives: list[SpeakerCandidate] = Field(default_factory=list)


class TitanetFileMetadata(StrictModel):
    windows: list[TimedSpeakerPrediction] = Field(default_factory=list)


class TitanetSegmentMetadata(StrictModel):
    speaker_id: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    alternatives: list[SpeakerCandidate] = Field(default_factory=list)


class EmotionCategoricalWindow(StrictModel):
    start: float = Field(ge=0.0)
    end: float = Field(gt=0.0)
    top_label: str | None = None
    label_scores: dict[str, float] = Field(default_factory=dict)


class EmotionCategoricalFileMetadata(StrictModel):
    windows: list[EmotionCategoricalWindow] = Field(default_factory=list)


class EmotionCategoricalSegmentMetadata(StrictModel):
    top_label: str | None = None
    label_scores: dict[str, float] = Field(default_factory=dict)
    adjacent_window_labels: list[str] = Field(default_factory=list)


class EmotionAttributesWindow(StrictModel):
    start: float = Field(ge=0.0)
    end: float = Field(gt=0.0)
    attributes: dict[str, float] = Field(default_factory=dict)


class EmotionAttributesFileMetadata(StrictModel):
    windows: list[EmotionAttributesWindow] = Field(default_factory=list)


class EmotionAttributesSegmentMetadata(StrictModel):
    attributes: dict[str, float] = Field(default_factory=dict)
    adjacent_window_attributes: list[dict[str, float]] = Field(default_factory=list)


class SoundEvent(StrictModel):
    label: str
    start: float = Field(ge=0.0)
    end: float = Field(gt=0.0)
    score: float = Field(ge=0.0, le=1.0)


class SEDFileMetadata(StrictModel):
    events: list[SoundEvent] = Field(default_factory=list)


class SEDSegmentMetadata(StrictModel):
    dominant_label: str | None = None
    dominant_score: float | None = Field(default=None, ge=0.0, le=1.0)
    events: list[SoundEvent] = Field(default_factory=list)


class TimedCorrectionSpan(StrictModel):
    start: float = Field(ge=0.0)
    end: float = Field(gt=0.0)
    raw_text: str
    corrected_text: str
    edit_count: int | None = Field(default=None, ge=0)


class SpellingCorrectionFileMetadata(StrictModel):
    raw_text: str | None = None
    corrected_text: str | None = None
    segments: list[TimedCorrectionSpan] = Field(default_factory=list)


class SpellingCorrectionSegmentMetadata(StrictModel):
    raw_text: str | None = None
    corrected_text: str | None = None
    edit_count: int | None = Field(default=None, ge=0)
    normalized_edit_rate: float | None = Field(default=None, ge=0.0)


class ContextWindow(StrictModel):
    start: float = Field(ge=0.0)
    end: float = Field(gt=0.0)
    scene: str | None = None
    tags: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class GeneralContextFileMetadata(StrictModel):
    windows: list[ContextWindow] = Field(default_factory=list)


class GeneralContextSegmentMetadata(StrictModel):
    scene: str | None = None
    tags: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    adjacent_scenes: list[str] = Field(default_factory=list)


class EnhancementSegmentMetadata(StrictModel):
    model_name: str | None = None
    notes: list[str] = Field(default_factory=list)


class UpstreamFilePayload(StrictModel):
    parakeet_tdt: ASRFileMetadata | None = None
    diarizen_wavlm_large_s80_md_v2: DiarizationFileMetadata | None = None
    titanet: TitanetFileMetadata | None = None
    emotion_categorical_3loi: EmotionCategoricalFileMetadata | None = None
    emotion_multi_attributes_3loi: EmotionAttributesFileMetadata | None = None
    atst_sed: SEDFileMetadata | None = None
    qwen3_spelling_correction: SpellingCorrectionFileMetadata | None = None
    qwen2_5_omni: GeneralContextFileMetadata | None = None


class UpstreamSegmentPayload(StrictModel):
    parakeet_tdt: ASRSegmentMetadata | None = None
    diarizen_wavlm_large_s80_md_v2: DiarizationSegmentMetadata | None = None
    titanet: TitanetSegmentMetadata | None = None
    emotion_categorical_3loi: EmotionCategoricalSegmentMetadata | None = None
    emotion_multi_attributes_3loi: EmotionAttributesSegmentMetadata | None = None
    atst_sed: SEDSegmentMetadata | None = None
    qwen3_spelling_correction: SpellingCorrectionSegmentMetadata | None = None
    qwen2_5_omni: GeneralContextSegmentMetadata | None = None
    mossformergan_se_16k: EnhancementSegmentMetadata | None = None


class AnalyzeFileRequest(StrictModel):
    schema_version: Literal["v1"] = "v1"
    audio: AudioInput
    segments: list[SegmentSpec] | None = None
    rolling_windows: RollingWindowConfig | None = None
    upstream_file_level: UpstreamFilePayload | None = None
    upstream_by_segment: dict[str, UpstreamSegmentPayload] = Field(default_factory=dict)


class AnalyzeSegmentsRequest(StrictModel):
    schema_version: Literal["v1"] = "v1"
    audio: AudioInput
    enhanced_audio: AudioInput | None = None
    segments: list[SegmentSpec]
    upstream_file_level: UpstreamFilePayload | None = None
    upstream_by_segment: dict[str, UpstreamSegmentPayload] = Field(default_factory=dict)


class SourceGroupFlag(StrictModel):
    from_raw_audio: bool
    from_upstream_metadata: bool
    missing_inputs: list[str] = Field(default_factory=list)


class SourceFlags(StrictModel):
    speaker_id: SourceGroupFlag
    quality_metrics: SourceGroupFlag
    learned_speech_quality: SourceGroupFlag
    boundary_risk: SourceGroupFlag
    overlap_risk: SourceGroupFlag
    speaker_consistency: SourceGroupFlag
    asr_consistency: SourceGroupFlag
    spelling_correction_delta: SourceGroupFlag
    sed_context_dominance: SourceGroupFlag
    emotion_instability: SourceGroupFlag
    enhancement_delta: SourceGroupFlag
    general_context_consistency: SourceGroupFlag
    segment_integrity: SourceGroupFlag
    confidence_components: SourceGroupFlag
    task_usability: SourceGroupFlag


class QualityMetrics(StrictModel):
    duration_s: float
    speech_ratio: float
    silence_ratio: float
    rms_dbfs: float
    peak_dbfs: float
    dynamic_range_db: float
    crest_factor_db: float
    clipping_fraction: float
    zero_crossing_rate: float
    spectral_centroid_hz: float
    spectral_rolloff_hz: float
    spectral_bandwidth_hz: float
    low_freq_energy_ratio: float
    high_freq_energy_ratio: float
    blind_snr_db: float


class NISQAResult(StrictModel):
    available: bool
    mos: float | None = None
    noisiness: float | None = None
    coloration: float | None = None
    discontinuity: float | None = None
    loudness: float | None = None


class DNSMOSResult(StrictModel):
    available: bool
    ovrl: float | None = None
    sig: float | None = None
    bak: float | None = None
    p808_mos: float | None = None


class LearnedSpeechQuality(StrictModel):
    nisqa: NISQAResult
    dnsmos: DNSMOSResult


class BoundaryRiskAssessment(StrictModel):
    available: bool
    risk: float | None = None
    nearest_boundary_distance_s: float | None = None
    internal_boundary_count: int | None = None


class OverlapRiskAssessment(StrictModel):
    available: bool
    risk: float | None = None
    overlap_fraction: float | None = None


class SpeakerConsistencyAssessment(StrictModel):
    available: bool
    score: float | None = None
    dominant_speaker_id: str | None = None
    predicted_speaker_id: str | None = None
    confidence: float | None = None
    mismatch: bool | None = None


class ASRConsistencyAssessment(StrictModel):
    available: bool
    score: float | None = None
    avg_token_confidence: float | None = None
    gap_rate: float | None = None
    transcript_density_wps: float | None = None
    token_count: int | None = None


class SpellingCorrectionDeltaAssessment(StrictModel):
    available: bool
    score: float | None = None
    edit_count: int | None = None
    normalized_edit_rate: float | None = None


class SEDContextDominanceAssessment(StrictModel):
    available: bool
    score: float | None = None
    dominant_label: str | None = None
    dominant_non_speech_fraction: float | None = None


class EmotionInstabilityAssessment(StrictModel):
    available: bool
    score: float | None = None
    instability: float | None = None
    dominant_label: str | None = None


class EnhancementDeltaAssessment(StrictModel):
    available: bool
    score: float | None = None
    nisqa_mos_delta: float | None = None
    blind_snr_delta_db: float | None = None
    spectral_shift_ratio: float | None = None


class GeneralContextConsistencyAssessment(StrictModel):
    available: bool
    score: float | None = None
    dominant_scene: str | None = None
    scene_consistency: float | None = None


class SegmentIntegrityAssessment(StrictModel):
    available: bool
    score: float
    duration_score: float
    clipping_score: float
    speech_presence_score: float
    headroom_score: float


class ConfidenceComponent(StrictModel):
    available: bool
    score: float | None = None
    weight: float
    contribution: float
    evidence: list[str] = Field(default_factory=list)


class ConfidenceComponents(StrictModel):
    audio_quality: ConfidenceComponent
    boundary: ConfidenceComponent
    overlap: ConfidenceComponent
    speaker: ConfidenceComponent
    asr: ConfidenceComponent
    spelling: ConfidenceComponent
    sed: ConfidenceComponent
    emotion: ConfidenceComponent
    enhancement: ConfidenceComponent
    context: ConfidenceComponent
    integrity: ConfidenceComponent


class QualityAssessment(StrictModel):
    learned_speech_quality: LearnedSpeechQuality
    boundary_risk: BoundaryRiskAssessment
    overlap_risk: OverlapRiskAssessment
    speaker_consistency: SpeakerConsistencyAssessment
    asr_consistency: ASRConsistencyAssessment
    spelling_correction_delta: SpellingCorrectionDeltaAssessment
    sed_context_dominance: SEDContextDominanceAssessment
    emotion_instability: EmotionInstabilityAssessment
    enhancement_delta: EnhancementDeltaAssessment
    general_context_consistency: GeneralContextConsistencyAssessment
    segment_integrity: SegmentIntegrityAssessment
    confidence_components: ConfidenceComponents
    overall_confidence: float


class TaskUsability(StrictModel):
    asr: float
    emotion: float
    speaker_id: float
    sed: float
    general_audio_understanding: float


class SegmentResult(StrictModel):
    segment_id: str
    start: float
    end: float
    speaker_id: str | None = None
    source_flags: SourceFlags
    quality_metrics: QualityMetrics
    quality_assessment: QualityAssessment
    task_usability: TaskUsability
    reason_codes: list[str]


class AggregateStats(StrictModel):
    mean: float
    std: float
    p05: float
    p50: float
    p95: float


class FlagSummary(StrictModel):
    count: int
    total_duration_s: float


class FileSummary(StrictModel):
    num_segments: int
    total_duration_s: float
    aggregates: dict[str, AggregateStats]
    flags: dict[str, FlagSummary]


class ServiceInputUsage(StrictModel):
    provided: bool
    contributed_segments: int


class ServiceInputsUsed(StrictModel):
    parakeet_tdt: ServiceInputUsage
    diarizen_wavlm_large_s80_md_v2: ServiceInputUsage
    titanet: ServiceInputUsage
    emotion_categorical_3loi: ServiceInputUsage
    emotion_multi_attributes_3loi: ServiceInputUsage
    atst_sed: ServiceInputUsage
    qwen3_spelling_correction: ServiceInputUsage
    qwen2_5_omni: ServiceInputUsage
    mossformergan_se_16k: ServiceInputUsage


class AnalyzeResponse(StrictModel):
    schema_version: Literal["v1"] = "v1"
    segments: list[SegmentResult]
    file_summary: FileSummary
    service_inputs_used: ServiceInputsUsed


class BackendHealth(StrictModel):
    ready: bool
    detail: str


class HealthResponse(StrictModel):
    schema_version: Literal["v1"] = "v1"
    ready: bool
    device: str
    backends: dict[str, BackendHealth]

