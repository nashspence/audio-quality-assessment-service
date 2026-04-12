from __future__ import annotations

import base64
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from .config import Settings
from .schemas import AudioInput, RollingWindowConfig, SegmentSpec


@dataclass(slots=True)
class LoadedAudio:
    samples: np.ndarray
    sample_rate: int
    label: str

    @property
    def duration_s(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return float(len(self.samples) / self.sample_rate)


def _mix_to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples.astype(np.float32)
    if samples.ndim == 2:
        return samples.mean(axis=0, dtype=np.float32)
    raise ValueError("Unsupported audio shape.")


def _load_audio_path(path: Path) -> LoadedAudio:
    try:
        samples, sample_rate = librosa.load(path.as_posix(), sr=None, mono=False)
    except Exception as exc:  # pragma: no cover - delegated to runtime codec support
        raise ValueError(f"Could not decode audio from {path}") from exc
    mono = _mix_to_mono(np.asarray(samples))
    return LoadedAudio(samples=mono, sample_rate=int(sample_rate), label=path.name)


def load_audio_input(audio_input: AudioInput, settings: Settings) -> LoadedAudio:
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    if audio_input.path:
        path = Path(audio_input.path).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Audio path does not exist: {path}")
        return _load_audio_path(path)

    suffix = ".bin"
    if audio_input.filename and "." in audio_input.filename:
        suffix = Path(audio_input.filename).suffix or suffix
    payload = base64.b64decode(audio_input.base64 or "")
    with tempfile.NamedTemporaryFile(
        dir=settings.temp_dir,
        suffix=suffix,
        delete=False,
    ) as handle:
        handle.write(payload)
        temp_path = Path(handle.name)
    try:
        return _load_audio_path(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def slice_audio(samples: np.ndarray, sample_rate: int, start: float, end: float) -> np.ndarray:
    start_idx = max(0, int(round(start * sample_rate)))
    end_idx = max(start_idx, int(round(end * sample_rate)))
    return np.asarray(samples[start_idx:end_idx], dtype=np.float32)


def validate_segments(segments: list[SegmentSpec], duration_s: float) -> None:
    for segment in segments:
        if segment.end > duration_s + 1e-6:
            raise ValueError(
                f"Segment '{segment.segment_id}' ends at {segment.end:.3f}s, beyond audio duration "
                f"{duration_s:.3f}s."
            )
    ids = [segment.segment_id for segment in segments]
    if len(ids) != len(set(ids)):
        raise ValueError("Segment IDs must be unique.")


def build_default_segments(
    duration_s: float,
    rolling_config: RollingWindowConfig | None,
    settings: Settings,
) -> list[SegmentSpec]:
    segments = [SegmentSpec(segment_id="full_file", start=0.0, end=duration_s)]
    config = rolling_config or RollingWindowConfig()
    if not config.enabled:
        return segments

    window_s = min(config.window_seconds or settings.rolling_window_seconds, duration_s)
    hop_s = config.hop_seconds or settings.rolling_hop_seconds
    if duration_s < settings.min_window_seconds or window_s >= duration_s - 1e-6:
        return segments

    starts: list[float] = []
    cursor = 0.0
    while cursor + window_s <= duration_s + 1e-6:
        starts.append(round(cursor, 6))
        cursor += hop_s
    last_start = max(duration_s - window_s, 0.0)
    if not starts or abs(starts[-1] - last_start) > 1e-6:
        starts.append(round(last_start, 6))

    for index, start in enumerate(starts):
        end = min(start + window_s, duration_s)
        segments.append(
            SegmentSpec(
                segment_id=f"window_{index:03d}",
                start=start,
                end=end,
            )
        )
    return segments

