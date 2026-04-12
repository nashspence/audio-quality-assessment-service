from __future__ import annotations

import io
import math
from typing import Any

import numpy as np
import soundfile as sf

EPS = 1e-10


def load_audio_from_bytes(file_bytes: bytes) -> tuple[np.ndarray, int]:
    try:
        audio, sample_rate = sf.read(
            io.BytesIO(file_bytes),
            always_2d=True,
            dtype="float32",
        )
    except Exception as exc:  # pragma: no cover - backend-specific error types vary
        raise ValueError("Unable to decode the uploaded audio file.") from exc
    if audio.size == 0:
        raise ValueError("Uploaded audio file is empty.")
    return audio, int(sample_rate)


def frame_signal(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if signal.ndim != 1:
        raise ValueError("frame_signal expects a mono waveform.")
    if signal.size < frame_length:
        signal = np.pad(signal, (0, frame_length - signal.size))
    remainder = (signal.size - frame_length) % hop_length
    if remainder:
        signal = np.pad(signal, (0, hop_length - remainder))
    return np.lib.stride_tricks.sliding_window_view(signal, frame_length)[::hop_length]


def compute_quality_metrics(
    signal: np.ndarray,
    sample_rate: int,
    channels_original: int,
    clip_threshold: float,
) -> dict[str, Any]:
    duration_seconds = float(signal.size) / float(sample_rate)
    frame_length = max(256, int(sample_rate * 0.025))
    hop_length = max(128, int(sample_rate * 0.010))
    frames = frame_signal(signal, frame_length, hop_length)

    rms = np.sqrt(np.mean(np.square(frames), axis=1) + EPS)
    rms_dbfs = 20.0 * np.log10(rms + EPS)
    total_rms = float(np.sqrt(np.mean(np.square(signal)) + EPS))
    peak = float(np.max(np.abs(signal)) + EPS)
    clipping_fraction = float(np.mean(np.abs(signal) >= clip_threshold))
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)

    window = np.hanning(frame_length).astype(np.float32)
    spectrum = np.fft.rfft(frames * window, axis=1)
    power = np.abs(spectrum) ** 2 + EPS
    freqs = np.fft.rfftfreq(frame_length, 1.0 / sample_rate)
    power_sum = np.sum(power, axis=1) + EPS

    spectral_centroid = np.sum(power * freqs, axis=1) / power_sum
    spectral_bandwidth = np.sqrt(
        np.sum(power * (freqs - spectral_centroid[:, None]) ** 2, axis=1) / power_sum
    )
    cumulative_power = np.cumsum(power, axis=1)
    rolloff_threshold = 0.85 * power_sum
    spectral_rolloff = freqs[
        np.argmax(cumulative_power >= rolloff_threshold[:, None], axis=1)
    ]
    spectral_flatness = np.exp(np.mean(np.log(power), axis=1)) / (
        np.mean(power, axis=1) + EPS
    )

    normalized_power = power / power_sum[:, None]
    spectral_flux = np.sqrt(np.sum(np.diff(normalized_power, axis=0) ** 2, axis=1))

    low_mask = freqs < 300.0
    speech_mask = (freqs >= 300.0) & (freqs < 3400.0)
    high_mask = freqs >= 3400.0
    low_band_ratio = np.sum(power[:, low_mask], axis=1) / power_sum
    speech_band_ratio = np.sum(power[:, speech_mask], axis=1) / power_sum
    high_band_ratio = np.sum(power[:, high_mask], axis=1) / power_sum

    noise_floor_db = float(np.percentile(rms_dbfs, 20))
    speech_threshold_db = max(noise_floor_db + 6.0, float(np.percentile(rms_dbfs, 55)))
    speech_frames = (
        (rms_dbfs >= speech_threshold_db)
        & (speech_band_ratio >= 0.2)
        & (spectral_flatness <= 0.75)
    )
    if int(np.count_nonzero(speech_frames)) == 0 and duration_seconds > 0.2:
        speech_frames = rms_dbfs >= float(np.percentile(rms_dbfs, 75))
    silence_frames = rms_dbfs <= noise_floor_db + 3.0

    speech_ratio = float(np.mean(speech_frames))
    silence_ratio = float(np.mean(silence_frames))
    speech_db = rms_dbfs[speech_frames]
    non_speech_db = rms_dbfs[silence_frames]
    if speech_db.size and non_speech_db.size:
        blind_snr_db = float(np.mean(speech_db) - np.mean(non_speech_db))
    else:
        blind_snr_db = float(np.percentile(rms_dbfs, 75) - np.percentile(rms_dbfs, 25))

    edge_frame_count = max(1, int(round(0.2 / (hop_length / sample_rate))))
    edge_speech_start = float(np.mean(speech_frames[:edge_frame_count]))
    edge_speech_end = float(np.mean(speech_frames[-edge_frame_count:]))

    energy_distribution = (rms**2) / (np.sum(rms**2) + EPS)
    entropy_denominator = (
        math.log(len(energy_distribution)) if len(energy_distribution) > 1 else 1.0
    )
    energy_entropy = float(
        -np.sum(energy_distribution * np.log(energy_distribution + EPS))
        / entropy_denominator
    )

    metrics = {
        "duration_seconds": round(duration_seconds, 6),
        "sample_rate_hz": int(sample_rate),
        "channels_original": int(channels_original),
        "rms_level_dbfs": round(20.0 * math.log10(total_rms + EPS), 6),
        "peak_level_dbfs": round(20.0 * math.log10(peak), 6),
        "dynamic_range_db": round(
            float(np.percentile(rms_dbfs, 95) - np.percentile(rms_dbfs, 5)), 6
        ),
        "crest_factor": round(peak / (total_rms + EPS), 6),
        "clipping_fraction": round(clipping_fraction, 6),
        "zero_crossing_rate": round(float(np.mean(zcr)), 6),
        "speech_ratio": round(speech_ratio, 6),
        "silence_ratio": round(silence_ratio, 6),
        "blind_snr_estimate_db": round(blind_snr_db, 6),
        "spectral_centroid_hz": round(float(np.mean(spectral_centroid)), 6),
        "spectral_bandwidth_hz": round(float(np.mean(spectral_bandwidth)), 6),
        "spectral_rolloff_hz": round(float(np.mean(spectral_rolloff)), 6),
        "spectral_flatness": round(float(np.mean(spectral_flatness)), 6),
        "spectral_flux": round(
            float(np.mean(spectral_flux)) if spectral_flux.size else 0.0, 6
        ),
        "low_band_energy_ratio": round(float(np.mean(low_band_ratio)), 6),
        "speech_band_energy_ratio": round(float(np.mean(speech_band_ratio)), 6),
        "high_band_energy_ratio": round(float(np.mean(high_band_ratio)), 6),
        "energy_entropy": round(energy_entropy, 6),
        "dc_offset": round(float(np.mean(signal)), 6),
        "speech_frame_count": int(np.count_nonzero(speech_frames)),
        "total_frame_count": int(frames.shape[0]),
        "edge_speech_start_ratio": round(edge_speech_start, 6),
        "edge_speech_end_ratio": round(edge_speech_end, 6),
    }
    return metrics
