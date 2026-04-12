from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf


def speech_like(
    duration_seconds: float, sample_rate: int, base_hz: float, noise_scale: float
) -> np.ndarray:
    t = np.linspace(
        0.0, duration_seconds, int(duration_seconds * sample_rate), endpoint=False
    )
    envelope = 0.5 * (1.0 + np.sin(2.0 * math.pi * 1.8 * t))
    carrier = (
        0.52 * np.sin(2.0 * math.pi * base_hz * t)
        + 0.28 * np.sin(2.0 * math.pi * base_hz * 2.1 * t)
        + 0.16 * np.sin(2.0 * math.pi * base_hz * 3.9 * t)
    )
    noise = np.random.default_rng(7).normal(0.0, noise_scale, size=t.shape)
    return (carrier * envelope) + noise


def main() -> None:
    sample_rate = 16000
    first = speech_like(2.5, sample_rate, 180.0, 0.015)
    second = speech_like(2.5, sample_rate, 210.0, 0.02)
    traffic = np.random.default_rng(11).normal(0.0, 0.06, size=second.shape)
    second = (0.78 * second) + traffic
    waveform = np.concatenate([first, second]).astype(np.float32)
    waveform = np.clip(waveform, -0.98, 0.98)

    output_path = Path("examples/smoke-sample.wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, waveform, sample_rate)


if __name__ == "__main__":
    main()
