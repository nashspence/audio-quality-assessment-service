from __future__ import annotations

from dataclasses import dataclass

from fastapi import HTTPException

from .audio import LoadedAudio, build_default_segments, load_audio_input, slice_audio, validate_segments
from .config import Settings
from .fusion import build_file_summary, build_learned_speech_quality, score_segment
from .metrics import compute_quality_metrics
from .quality_backends import DNSMOSBackend, DNSMOSScores, NISQABackend
from .schemas import (
    AnalyzeFileRequest,
    AnalyzeResponse,
    AnalyzeSegmentsRequest,
    BackendHealth,
    HealthResponse,
    ServiceInputUsage,
    ServiceInputsUsed,
)
from .upstream import resolve_segment_upstream


SERVICE_NAMES = (
    "parakeet_tdt",
    "diarizen_wavlm_large_s80_md_v2",
    "titanet",
    "emotion_categorical_3loi",
    "emotion_multi_attributes_3loi",
    "atst_sed",
    "qwen3_spelling_correction",
    "qwen2_5_omni",
    "mossformergan_se_16k",
)


@dataclass(slots=True)
class RuntimeState:
    ready: bool
    device: str
    backend_status: dict[str, BackendHealth]


class QualityService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.nisqa = NISQABackend(settings)
        self.dnsmos = DNSMOSBackend(settings)
        self.state = RuntimeState(
            ready=False,
            device="cpu",
            backend_status={
                "nisqa": BackendHealth(ready=False, detail="not_loaded"),
                "dnsmos": BackendHealth(ready=not settings.enable_dnsmos, detail="disabled" if not settings.enable_dnsmos else "not_loaded"),
            },
        )

    def load(self) -> None:
        self.settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.settings.temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.nisqa.load()
            self.state.backend_status["nisqa"] = BackendHealth(ready=True, detail=self.nisqa.detail)
            self.state.device = self.nisqa.device_name
        except Exception as exc:
            self.state.backend_status["nisqa"] = BackendHealth(ready=False, detail=str(exc))

        if self.settings.enable_dnsmos:
            try:
                self.dnsmos.load()
                self.state.backend_status["dnsmos"] = BackendHealth(ready=True, detail=self.dnsmos.detail)
            except Exception as exc:
                self.state.backend_status["dnsmos"] = BackendHealth(ready=False, detail=str(exc))

        self.state.ready = self.state.backend_status["nisqa"].ready and self.state.backend_status["dnsmos"].ready

    def health(self) -> HealthResponse:
        return HealthResponse(
            ready=self.state.ready,
            device=self.state.device,
            backends=self.state.backend_status,
        )

    def _ensure_ready(self) -> None:
        if not self.state.ready:
            raise HTTPException(status_code=503, detail=self.health().model_dump())

    def analyze_file(self, request: AnalyzeFileRequest) -> AnalyzeResponse:
        self._ensure_ready()
        audio = load_audio_input(request.audio, self.settings)
        segments = request.segments or build_default_segments(
            audio.duration_s,
            request.rolling_windows,
            self.settings,
        )
        validate_segments(segments, audio.duration_s)
        return self._analyze(
            audio=audio,
            segments=segments,
            upstream_file_level=request.upstream_file_level,
            upstream_by_segment=request.upstream_by_segment,
            enhanced_audio=None,
        )

    def analyze_segments(self, request: AnalyzeSegmentsRequest) -> AnalyzeResponse:
        self._ensure_ready()
        audio = load_audio_input(request.audio, self.settings)
        enhanced = load_audio_input(request.enhanced_audio, self.settings) if request.enhanced_audio else None
        validate_segments(request.segments, audio.duration_s)
        return self._analyze(
            audio=audio,
            segments=request.segments,
            upstream_file_level=request.upstream_file_level,
            upstream_by_segment=request.upstream_by_segment,
            enhanced_audio=enhanced,
        )

    def _analyze(
        self,
        audio: LoadedAudio,
        segments,
        upstream_file_level,
        upstream_by_segment,
        enhanced_audio: LoadedAudio | None,
    ) -> AnalyzeResponse:
        raw_segment_audio = []
        enhanced_segment_audio = []
        raw_metrics = {}
        enhanced_metrics = {}
        resolved_upstreams = {}
        contribution_counts = {name: 0 for name in SERVICE_NAMES}

        for segment in segments:
            segment_samples = slice_audio(audio.samples, audio.sample_rate, segment.start, segment.end)
            raw_segment_audio.append((segment.segment_id, segment_samples, audio.sample_rate))
            raw_metrics[segment.segment_id] = compute_quality_metrics(segment_samples, audio.sample_rate, self.settings)

            resolved = resolve_segment_upstream(
                segment,
                upstream_file_level=upstream_file_level,
                upstream_by_segment=upstream_by_segment,
                enhancement_provided=enhanced_audio is not None,
            )
            resolved_upstreams[segment.segment_id] = resolved

            if resolved.asr:
                contribution_counts["parakeet_tdt"] += 1
            if resolved.diarization:
                contribution_counts["diarizen_wavlm_large_s80_md_v2"] += 1
            if resolved.titanet:
                contribution_counts["titanet"] += 1
            if resolved.emotion_categorical:
                contribution_counts["emotion_categorical_3loi"] += 1
            if resolved.emotion_attributes:
                contribution_counts["emotion_multi_attributes_3loi"] += 1
            if resolved.sed:
                contribution_counts["atst_sed"] += 1
            if resolved.spelling:
                contribution_counts["qwen3_spelling_correction"] += 1
            if resolved.context:
                contribution_counts["qwen2_5_omni"] += 1

            if enhanced_audio and segment.end <= enhanced_audio.duration_s + 1e-6:
                enhanced_samples = slice_audio(
                    enhanced_audio.samples,
                    enhanced_audio.sample_rate,
                    segment.start,
                    segment.end,
                )
                enhanced_segment_audio.append((segment.segment_id, enhanced_samples, enhanced_audio.sample_rate))
                enhanced_metrics[segment.segment_id] = compute_quality_metrics(
                    enhanced_samples,
                    enhanced_audio.sample_rate,
                    self.settings,
                )
                contribution_counts["mossformergan_se_16k"] += 1

        raw_nisqa = self.nisqa.score_segments(raw_segment_audio)
        enhanced_nisqa = self.nisqa.score_segments(enhanced_segment_audio) if enhanced_segment_audio else {}

        raw_dnsmos = {}
        enhanced_dnsmos = {}
        if self.settings.enable_dnsmos:
            raw_dnsmos = {
                segment_id: self.dnsmos.score_segment(samples, sample_rate)
                for segment_id, samples, sample_rate in raw_segment_audio
            }
            enhanced_dnsmos = {
                segment_id: self.dnsmos.score_segment(samples, sample_rate)
                for segment_id, samples, sample_rate in enhanced_segment_audio
            }

        results = []
        for segment in segments:
            segment_id = segment.segment_id
            learned = build_learned_speech_quality(
                raw_nisqa[segment_id],
                raw_dnsmos.get(segment_id),
            )
            enhanced_learned = None
            if segment_id in enhanced_nisqa:
                enhanced_learned = build_learned_speech_quality(
                    enhanced_nisqa[segment_id],
                    enhanced_dnsmos.get(segment_id),
                )
            results.append(
                score_segment(
                    segment=segment,
                    metrics=raw_metrics[segment_id],
                    learned=learned,
                    upstream=resolved_upstreams[segment_id],
                    settings=self.settings,
                    enhanced_metrics=enhanced_metrics.get(segment_id),
                    enhanced_learned=enhanced_learned,
                )
            )

        file_summary = build_file_summary(results, self.settings)
        service_inputs_used = self._service_inputs_used(
            upstream_file_level=upstream_file_level,
            upstream_by_segment=upstream_by_segment,
            contribution_counts=contribution_counts,
        )
        return AnalyzeResponse(
            segments=results,
            file_summary=file_summary,
            service_inputs_used=service_inputs_used,
        )

    def _service_inputs_used(
        self,
        upstream_file_level,
        upstream_by_segment,
        contribution_counts: dict[str, int],
    ) -> ServiceInputsUsed:
        direct_payloads = list(upstream_by_segment.values())
        return ServiceInputsUsed(
            parakeet_tdt=ServiceInputUsage(
                provided=bool(
                    (upstream_file_level and upstream_file_level.parakeet_tdt)
                    or any(payload.parakeet_tdt for payload in direct_payloads)
                ),
                contributed_segments=contribution_counts["parakeet_tdt"],
            ),
            diarizen_wavlm_large_s80_md_v2=ServiceInputUsage(
                provided=bool(
                    (upstream_file_level and upstream_file_level.diarizen_wavlm_large_s80_md_v2)
                    or any(payload.diarizen_wavlm_large_s80_md_v2 for payload in direct_payloads)
                ),
                contributed_segments=contribution_counts["diarizen_wavlm_large_s80_md_v2"],
            ),
            titanet=ServiceInputUsage(
                provided=bool(
                    (upstream_file_level and upstream_file_level.titanet)
                    or any(payload.titanet for payload in direct_payloads)
                ),
                contributed_segments=contribution_counts["titanet"],
            ),
            emotion_categorical_3loi=ServiceInputUsage(
                provided=bool(
                    (upstream_file_level and upstream_file_level.emotion_categorical_3loi)
                    or any(payload.emotion_categorical_3loi for payload in direct_payloads)
                ),
                contributed_segments=contribution_counts["emotion_categorical_3loi"],
            ),
            emotion_multi_attributes_3loi=ServiceInputUsage(
                provided=bool(
                    (upstream_file_level and upstream_file_level.emotion_multi_attributes_3loi)
                    or any(payload.emotion_multi_attributes_3loi for payload in direct_payloads)
                ),
                contributed_segments=contribution_counts["emotion_multi_attributes_3loi"],
            ),
            atst_sed=ServiceInputUsage(
                provided=bool(
                    (upstream_file_level and upstream_file_level.atst_sed)
                    or any(payload.atst_sed for payload in direct_payloads)
                ),
                contributed_segments=contribution_counts["atst_sed"],
            ),
            qwen3_spelling_correction=ServiceInputUsage(
                provided=bool(
                    (upstream_file_level and upstream_file_level.qwen3_spelling_correction)
                    or any(payload.qwen3_spelling_correction for payload in direct_payloads)
                ),
                contributed_segments=contribution_counts["qwen3_spelling_correction"],
            ),
            qwen2_5_omni=ServiceInputUsage(
                provided=bool(
                    (upstream_file_level and upstream_file_level.qwen2_5_omni)
                    or any(payload.qwen2_5_omni for payload in direct_payloads)
                ),
                contributed_segments=contribution_counts["qwen2_5_omni"],
            ),
            mossformergan_se_16k=ServiceInputUsage(
                provided=contribution_counts["mossformergan_se_16k"] > 0
                or any(payload.mossformergan_se_16k for payload in direct_payloads),
                contributed_segments=contribution_counts["mossformergan_se_16k"],
            ),
        )

