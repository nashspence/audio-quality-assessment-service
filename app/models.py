from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def _round_float(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


class TimeWindow(BaseModel):
    start_time: float = Field(ge=0.0)
    end_time: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_times(self) -> "TimeWindow":
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be greater than start_time")
        return self

    @field_validator("start_time", "end_time")
    @classmethod
    def round_times(cls, value: float) -> float:
        return _round_float(value) or 0.0


class SegmentDefinition(TimeWindow):
    segment_id: str | None = None
    label: str | None = None
    channel: int | None = Field(default=None, ge=0)


class DiarizationSegment(TimeWindow):
    speaker_id: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    overlap: bool | None = None


class SpeakerIdSegment(TimeWindow):
    speaker_id: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class AsrWord(TimeWindow):
    word: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_proxy: float | None = Field(default=None, ge=0.0, le=1.0)


class EmotionSegment(TimeWindow):
    label: str | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    uncertainty: float | None = Field(default=None, ge=0.0, le=1.0)


class SedEvent(TimeWindow):
    event_label: str
    score: float | None = Field(default=None, ge=0.0, le=1.0)


class EnhancementMetadata(BaseModel):
    applied: bool = False
    provider: str | None = None
    noise_reduction_db: float | None = None
    artifacts_risk: float | None = Field(default=None, ge=0.0, le=1.0)
    residual_noise_risk: float | None = Field(default=None, ge=0.0, le=1.0)
    clipping_repaired: bool | None = None
    notes: str | None = None


class UpstreamMetadata(BaseModel):
    diarization_segments: list[DiarizationSegment] = Field(default_factory=list)
    speaker_id_segments: list[SpeakerIdSegment] = Field(default_factory=list)
    known_speaker_ids: list[str] = Field(default_factory=list)
    asr_words: list[AsrWord] = Field(default_factory=list)
    emotion_segments: list[EmotionSegment] = Field(default_factory=list)
    sed_events: list[SedEvent] = Field(default_factory=list)
    enhancement: EnhancementMetadata | None = None


class AnalysisOptions(BaseModel):
    include_file_summary: bool = True
    boundary_margin_seconds: float = Field(default=0.25, gt=0.0, le=2.0)
    clip_threshold: float = Field(default=0.99, gt=0.8, lt=1.0)


class AnalyzeRequest(BaseModel):
    request_id: str | None = None
    analysis_target: Literal["file", "segment"] | None = None
    segments: list[SegmentDefinition] = Field(default_factory=list)
    upstream: UpstreamMetadata | None = None
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)


class QualitySignal(BaseModel):
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    label: str | None = None
    evidence: list[str] = Field(default_factory=list)
    note: str | None = None


class QualityMetrics(BaseModel):
    duration_seconds: float = Field(ge=0.0)
    sample_rate_hz: int = Field(gt=0)
    channels_original: int = Field(gt=0)
    rms_level_dbfs: float
    peak_level_dbfs: float
    dynamic_range_db: float = Field(ge=0.0)
    crest_factor: float = Field(ge=0.0)
    clipping_fraction: float = Field(ge=0.0, le=1.0)
    zero_crossing_rate: float = Field(ge=0.0)
    speech_ratio: float = Field(ge=0.0, le=1.0)
    silence_ratio: float = Field(ge=0.0, le=1.0)
    blind_snr_estimate_db: float
    spectral_centroid_hz: float = Field(ge=0.0)
    spectral_bandwidth_hz: float = Field(ge=0.0)
    spectral_rolloff_hz: float = Field(ge=0.0)
    spectral_flatness: float = Field(ge=0.0)
    spectral_flux: float = Field(ge=0.0)
    low_band_energy_ratio: float = Field(ge=0.0, le=1.0)
    speech_band_energy_ratio: float = Field(ge=0.0, le=1.0)
    high_band_energy_ratio: float = Field(ge=0.0, le=1.0)
    energy_entropy: float = Field(ge=0.0, le=1.0)
    dc_offset: float
    speech_frame_count: int = Field(ge=0)
    total_frame_count: int = Field(gt=0)


class QualityAssessment(BaseModel):
    overlap_risk: QualitySignal
    boundary_proximity_risk: QualitySignal
    speech_dominance: QualitySignal
    non_speech_dominance: QualitySignal
    speaker_consistency: QualitySignal
    neighboring_window_instability: QualitySignal
    usability_for_asr: QualitySignal
    usability_for_emotion: QualitySignal
    usability_for_speaker_id: QualitySignal
    usability_for_general_downstream: QualitySignal


class ContributingFactor(BaseModel):
    name: str
    direction: Literal["boost", "penalty", "neutral"]
    source: str
    impact: float
    value: Any = None
    explanation: str


class ConfidenceSection(BaseModel):
    overall_confidence: float = Field(ge=0.0, le=1.0)
    segment_integrity: float = Field(ge=0.0, le=1.0)
    speech_usability_for_asr: float = Field(ge=0.0, le=1.0)
    speech_usability_for_emotion: float = Field(ge=0.0, le=1.0)
    speech_usability_for_speaker_id: float = Field(ge=0.0, le=1.0)
    reason_codes: list[str] = Field(default_factory=list)
    contributing_factors: list[ContributingFactor] = Field(default_factory=list)


class SegmentResult(BaseModel):
    segment_id: str
    start_time: float = Field(ge=0.0)
    end_time: float = Field(gt=0.0)
    label: str | None = None
    quality_metrics: QualityMetrics
    quality_assessment: QualityAssessment
    confidence: ConfidenceSection


class FileSummary(BaseModel):
    duration_seconds: float = Field(ge=0.0)
    sample_rate_hz: int = Field(gt=0)
    channel_count: int = Field(gt=0)
    segment_count: int = Field(ge=0)
    analyzed_duration_seconds: float = Field(ge=0.0)
    quality_metrics_summary: dict[str, Any]
    quality_assessment_summary: dict[str, Any]
    confidence_summary: dict[str, Any]


class AnalyzeResponse(BaseModel):
    schema_version: str = "1.0.0"
    request_id: str | None = None
    analysis_target: Literal["file", "segment"]
    file_summary: FileSummary | None = None
    segments: list[SegmentResult]
