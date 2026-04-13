from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    schema_version: str = "v1"
    service_name: str = "audio-quality-assessment-service"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    workspace_dir: Path = Path("/workspace")
    model_cache_dir: Path = Path("/models/cache")
    temp_dir: Path = Path("/tmp/audio-quality-service")

    prefer_gpu: bool = True
    require_gpu_for_nisqa: bool = True
    enable_dnsmos: bool = False
    nisqa_source_dir: Path = Path("/opt/vendor/NISQA")
    nisqa_model_url: str = (
        "https://raw.githubusercontent.com/gabrielmittag/NISQA/master/weights/nisqa.tar"
    )
    nisqa_model_filename: str = "nisqa.tar"
    nisqa_batch_size: int = 16
    nisqa_warmup_seconds: float = 1.2
    nisqa_min_segment_seconds: float = 1.0

    dnsmos_primary_model_url: str = (
        "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
    )
    dnsmos_p808_model_url: str = (
        "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/model_v8.onnx"
    )
    dnsmos_personalized_model_url: str = (
        "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/pDNSMOS/sig_bak_ovr.onnx"
    )

    rolling_window_seconds: float = 6.0
    rolling_hop_seconds: float = 3.0
    min_window_seconds: float = 2.0

    frame_length_ms: float = 30.0
    frame_hop_ms: float = 10.0
    silence_floor_offset_db: float = 6.0
    speech_threshold_below_peak_db: float = 35.0
    clipping_threshold: float = 0.995
    low_freq_cutoff_hz: float = 300.0
    high_freq_cutoff_hz: float = 3000.0

    short_segment_seconds: float = 1.0
    long_segment_seconds: float = 20.0
    blind_snr_bad_db: float = 5.0
    blind_snr_good_db: float = 25.0
    clipping_warn_fraction: float = 0.001
    clipping_bad_fraction: float = 0.01
    boundary_near_seconds: float = 0.35
    boundary_safe_seconds: float = 1.0
    overlap_warn_fraction: float = 0.1
    overlap_heavy_fraction: float = 0.35
    speaker_confidence_low: float = 0.5
    speaker_confidence_high: float = 0.85
    asr_confidence_low: float = 0.6
    asr_confidence_high: float = 0.9
    asr_gap_good_rate: float = 0.05
    asr_gap_bad_rate: float = 0.25
    asr_density_low_wps: float = 0.8
    asr_density_high_wps: float = 3.5
    spelling_edit_warn_rate: float = 0.12
    non_speech_dominance_warn: float = 0.6
    emotion_instability_warn: float = 0.45
    enhancement_shift_warn: float = 0.3
    context_consistency_low: float = 0.4
    low_confidence_threshold: float = 0.5

    weight_audio_quality: float = 3.0
    weight_boundary: float = 1.0
    weight_overlap: float = 1.0
    weight_speaker: float = 1.0
    weight_asr: float = 2.0
    weight_spelling: float = 1.0
    weight_sed: float = 1.0
    weight_emotion: float = 1.0
    weight_enhancement: float = 1.0
    weight_context: float = 1.0
    weight_integrity: float = 2.0

    max_reason_codes: int = 6

    @property
    def nisqa_model_path(self) -> Path:
        return self.model_cache_dir / self.nisqa_model_filename

    @property
    def dnsmos_primary_model_path(self) -> Path:
        return self.model_cache_dir / "dnsmos_sig_bak_ovr.onnx"

    @property
    def dnsmos_p808_model_path(self) -> Path:
        return self.model_cache_dir / "dnsmos_model_v8.onnx"

    @property
    def dnsmos_personalized_model_path(self) -> Path:
        return self.model_cache_dir / "pdnsmos_sig_bak_ovr.onnx"

    @property
    def file_summary_percentiles(self) -> tuple[int, int, int]:
        return (5, 50, 95)

    @property
    def speech_like_labels(self) -> tuple[str, ...]:
        return ("speech", "talking", "conversation", "voice", "human_voice", "speaker")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
