from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from .config import Settings
from .schemas import QualityMetrics


EPS = 1e-10
WADA_DB_VALS = np.arange(-20, 101)
WADA_G_VALS = np.array(
    [
        0.40974774,
        0.40986926,
        0.40998566,
        0.40969089,
        0.40986186,
        0.40999006,
        0.41027138,
        0.41052627,
        0.41101024,
        0.41143264,
        0.41231718,
        0.41337272,
        0.41526426,
        0.4178192,
        0.42077252,
        0.42452799,
        0.42918886,
        0.43510373,
        0.44234195,
        0.45161485,
        0.46221153,
        0.47491647,
        0.48883809,
        0.50509236,
        0.52353709,
        0.54372088,
        0.56532427,
        0.58847532,
        0.61346212,
        0.63954496,
        0.66750818,
        0.69583724,
        0.72454762,
        0.75414799,
        0.78323148,
        0.81240985,
        0.84219775,
        0.87166406,
        0.90030504,
        0.92880418,
        0.95655449,
        0.9835349,
        1.01047155,
        1.0362095,
        1.06136425,
        1.08579312,
        1.1094819,
        1.13277995,
        1.15472826,
        1.17627308,
        1.19703503,
        1.21671694,
        1.23535898,
        1.25364313,
        1.27103891,
        1.28718029,
        1.30302865,
        1.31839527,
        1.33294817,
        1.34700935,
        1.3605727,
        1.37345513,
        1.38577122,
        1.39733504,
        1.40856397,
        1.41959619,
        1.42983624,
        1.43958467,
        1.44902176,
        1.45804831,
        1.46669568,
        1.47486938,
        1.48269965,
        1.49034339,
        1.49748214,
        1.50435106,
        1.51076426,
        1.51698915,
        1.5229097,
        1.528578,
        1.53389835,
        1.5391211,
        1.5439065,
        1.54858517,
        1.55310776,
        1.55744391,
        1.56164927,
        1.56566348,
        1.56938671,
        1.57307767,
        1.57654764,
        1.57980083,
        1.58304129,
        1.58602496,
        1.58880681,
        1.59162477,
        1.5941969,
        1.59693155,
        1.599446,
        1.60185011,
        1.60408668,
        1.60627134,
        1.60826199,
        1.61004547,
        1.61192472,
        1.61369656,
        1.61534074,
        1.61688905,
        1.61838916,
        1.61985374,
        1.62135878,
        1.62268119,
        1.62390423,
        1.62513143,
        1.62632463,
        1.6274027,
        1.62842767,
        1.62945532,
        1.6303307,
        1.63128026,
        1.63204102,
    ],
    dtype=np.float64,
)


@dataclass(slots=True)
class FrameSummary:
    rms: np.ndarray
    rms_db: np.ndarray
    speech_mask: np.ndarray
    silence_mask: np.ndarray


def clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def safe_db(value: float) -> float:
    return float(20.0 * np.log10(max(value, EPS)))


def _frame_summary(samples: np.ndarray, sample_rate: int, settings: Settings) -> FrameSummary:
    frame_length = max(128, int(round(sample_rate * settings.frame_length_ms / 1000.0)))
    hop_length = max(64, int(round(sample_rate * settings.frame_hop_ms / 1000.0)))
    padded = librosa.util.fix_length(samples, size=max(len(samples), frame_length))
    frames = librosa.util.frame(padded, frame_length=frame_length, hop_length=hop_length)
    rms = np.sqrt(np.mean(np.square(frames), axis=0) + EPS)
    rms_db = 20.0 * np.log10(rms + EPS)
    floor_db = float(np.percentile(rms_db, 20))
    peak_db = float(np.max(rms_db))
    speech_threshold = max(floor_db + settings.silence_floor_offset_db, peak_db - settings.speech_threshold_below_peak_db)
    silence_threshold = floor_db + 3.0
    speech_mask = rms_db >= speech_threshold
    silence_mask = rms_db <= silence_threshold
    return FrameSummary(rms=rms, rms_db=rms_db, speech_mask=speech_mask, silence_mask=silence_mask)


def _quantile_snr(samples: np.ndarray, sample_rate: int, settings: Settings) -> float:
    frame = _frame_summary(samples, sample_rate, settings)
    if frame.rms.size == 0:
        return settings.blind_snr_bad_db
    noise_rms = float(np.percentile(frame.rms, 20))
    signal_rms = float(np.percentile(frame.rms, 80))
    signal_power = max(signal_rms**2 - noise_rms**2, EPS)
    noise_power = max(noise_rms**2, EPS)
    return float(10.0 * np.log10(signal_power / noise_power))


def estimate_blind_snr_db(samples: np.ndarray, sample_rate: int, settings: Settings) -> float:
    wav = np.asarray(samples, dtype=np.float64)
    wav = wav - float(np.mean(wav))
    total_energy = float(np.sum(wav**2))
    if total_energy <= EPS:
        return settings.blind_snr_bad_db
    peak = float(np.max(np.abs(wav)))
    if peak <= EPS:
        return settings.blind_snr_bad_db

    abs_wav = np.abs(wav / peak)
    abs_wav[abs_wav < EPS] = EPS
    v1 = max(EPS, float(abs_wav.mean()))
    v2 = float(np.log(abs_wav).mean())
    v3 = float(np.log(v1) - v2)

    snr_estimate: float | None = None
    if np.any(WADA_G_VALS < v3):
        idx = int(np.where(WADA_G_VALS < v3)[0].max())
        if idx < len(WADA_DB_VALS) - 1:
            step = (v3 - WADA_G_VALS[idx]) / max(WADA_G_VALS[idx + 1] - WADA_G_VALS[idx], EPS)
            snr_estimate = float(WADA_DB_VALS[idx] + step * (WADA_DB_VALS[idx + 1] - WADA_DB_VALS[idx]))

    if snr_estimate is None or snr_estimate <= -19.0 or snr_estimate >= 45.0:
        snr_estimate = _quantile_snr(samples, sample_rate, settings)

    return clamp(float(snr_estimate), -20.0, 50.0)


def compute_quality_metrics(samples: np.ndarray, sample_rate: int, settings: Settings) -> QualityMetrics:
    if samples.size == 0:
        raise ValueError("Segment audio is empty.")

    duration_s = float(len(samples) / sample_rate)
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(np.square(samples)) + EPS))
    frame = _frame_summary(samples, sample_rate, settings)

    stft = np.abs(librosa.stft(samples, n_fft=1024, hop_length=256, win_length=512))
    centroid = float(np.mean(librosa.feature.spectral_centroid(S=stft, sr=sample_rate)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=stft, sr=sample_rate, roll_percent=0.85)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(S=stft, sr=sample_rate)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(samples, frame_length=1024, hop_length=256)))

    spectrum = np.abs(np.fft.rfft(samples)) ** 2
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)
    total_energy = float(np.sum(spectrum) + EPS)
    low_ratio = float(np.sum(spectrum[freqs <= settings.low_freq_cutoff_hz]) / total_energy)
    high_ratio = float(np.sum(spectrum[freqs >= settings.high_freq_cutoff_hz]) / total_energy)

    return QualityMetrics(
        duration_s=duration_s,
        speech_ratio=float(np.mean(frame.speech_mask)),
        silence_ratio=float(np.mean(frame.silence_mask)),
        rms_dbfs=safe_db(rms),
        peak_dbfs=safe_db(peak),
        dynamic_range_db=float(np.percentile(frame.rms_db, 95) - np.percentile(frame.rms_db, 5)),
        crest_factor_db=float(20.0 * np.log10((peak + EPS) / (rms + EPS))),
        clipping_fraction=float(np.mean(np.abs(samples) >= settings.clipping_threshold)),
        zero_crossing_rate=zcr,
        spectral_centroid_hz=centroid,
        spectral_rolloff_hz=rolloff,
        spectral_bandwidth_hz=bandwidth,
        low_freq_energy_ratio=low_ratio,
        high_freq_energy_ratio=high_ratio,
        blind_snr_db=estimate_blind_snr_db(samples, sample_rate, settings),
    )

