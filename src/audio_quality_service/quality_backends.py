from __future__ import annotations

import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import librosa
import numpy as np
import soundfile as sf

from .config import Settings


def download_if_missing(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    with urlopen(url) as response, destination.open("wb") as handle:  # noqa: S310
        shutil.copyfileobj(response, handle)


@dataclass(slots=True)
class NISQAScores:
    mos: float | None
    noisiness: float | None
    coloration: float | None
    discontinuity: float | None
    loudness: float | None


@dataclass(slots=True)
class DNSMOSScores:
    ovrl: float | None
    sig: float | None
    bak: float | None
    p808_mos: float | None


class NISQABackend:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ready = False
        self.detail = "not_loaded"
        self.device_name = "cpu"
        self._torch = None
        self._pd = None
        self._nl = None
        self._args: dict[str, object] | None = None
        self._model = None

    def load(self) -> None:
        download_if_missing(self.settings.nisqa_model_url, self.settings.nisqa_model_path)
        source_dir = self.settings.nisqa_source_dir
        if not source_dir.exists():
            raise RuntimeError(f"NISQA source directory not found: {source_dir}")
        if source_dir.as_posix() not in sys.path:
            sys.path.insert(0, source_dir.as_posix())

        import pandas as pd
        import torch
        from nisqa import NISQA_lib as NL

        self._configure_torch_runtime(torch)
        self._torch = torch
        self._pd = pd
        self._nl = NL
        device = self._select_device(torch)
        self.device_name = self._describe_device(torch, device)
        try:
            checkpoint = torch.load(
                self.settings.nisqa_model_path.as_posix(),
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(self.settings.nisqa_model_path.as_posix(), map_location=device)

        args = dict(checkpoint["args"])
        model_name = str(args.get("model", checkpoint.get("model_name", "NISQA_DIM")))
        model_args = {
            key: args[key]
            for key in (
                "ms_seg_length",
                "ms_n_mels",
                "cnn_model",
                "cnn_c_out_1",
                "cnn_c_out_2",
                "cnn_c_out_3",
                "cnn_kernel_size",
                "cnn_dropout",
                "cnn_pool_1",
                "cnn_pool_2",
                "cnn_pool_3",
                "cnn_fc_out_h",
                "td",
                "td_sa_d_model",
                "td_sa_nhead",
                "td_sa_pos_enc",
                "td_sa_num_layers",
                "td_sa_h",
                "td_sa_dropout",
                "td_lstm_h",
                "td_lstm_num_layers",
                "td_lstm_dropout",
                "td_lstm_bidirectional",
                "td_2",
                "td_2_sa_d_model",
                "td_2_sa_nhead",
                "td_2_sa_pos_enc",
                "td_2_sa_num_layers",
                "td_2_sa_h",
                "td_2_sa_dropout",
                "td_2_lstm_h",
                "td_2_lstm_num_layers",
                "td_2_lstm_dropout",
                "td_2_lstm_bidirectional",
                "pool",
                "pool_att_h",
                "pool_att_dropout",
            )
        }
        model_class = getattr(NL, model_name)
        model = model_class(**model_args)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device)
        model.eval()

        self._args = args
        self._model = model
        self.ready = True
        self.detail = f"loaded:{model_name}:{self.device_name}:torch={torch.__version__}"
        self._warmup()

    def _configure_torch_runtime(self, torch) -> None:
        if not self.settings.prefer_gpu:
            return
        if not torch.cuda.is_available():
            return
        torch.cuda.set_device(0)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    def _select_device(self, torch) -> object:
        if not self.settings.prefer_gpu:
            return torch.device("cpu")
        if not torch.cuda.is_available():
            if self.settings.require_gpu_for_nisqa:
                raise RuntimeError("CUDA is required for NISQA, but torch.cuda.is_available() is false.")
            return torch.device("cpu")
        return torch.device("cuda:0")

    def _describe_device(self, torch, device) -> str:
        if getattr(device, "type", str(device)) != "cuda":
            return str(device)
        props = torch.cuda.get_device_properties(device)
        capability = f"sm_{props.major}{props.minor}"
        return f"{device}:{props.name}:{capability}"

    def _warmup(self) -> None:
        silence = np.zeros(int(self.settings.nisqa_warmup_seconds * 16000), dtype=np.float32)
        self.score_segments([("warmup", silence, 16000)])
        if self.device_name.startswith("cuda") and self._torch:
            self._torch.cuda.synchronize()

    def _pad_if_needed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        min_length = int(self.settings.nisqa_min_segment_seconds * sample_rate)
        if len(samples) >= min_length:
            return samples
        pad = np.zeros(min_length - len(samples), dtype=np.float32)
        return np.concatenate([samples, pad], axis=0)

    def score_segments(self, segments: list[tuple[str, np.ndarray, int]]) -> dict[str, NISQAScores]:
        if not self.ready or not self._model or not self._args or not self._nl or not self._pd or not self._torch:
            raise RuntimeError("NISQA backend is not ready.")

        filenames: list[str] = []
        id_by_filename: dict[str, str] = {}
        with tempfile.TemporaryDirectory(dir=self.settings.temp_dir.as_posix()) as temp_dir:
            for index, (segment_id, samples, sample_rate) in enumerate(segments):
                padded = self._pad_if_needed(samples, sample_rate)
                filename = f"{index:04d}.wav"
                path = Path(temp_dir) / filename
                sf.write(path.as_posix(), padded, sample_rate)
                filenames.append(filename)
                id_by_filename[filename] = segment_id

            df = self._pd.DataFrame({"deg": filenames})
            ds = self._nl.SpeechQualityDataset(
                df,
                df_con=None,
                data_dir=temp_dir,
                filename_column="deg",
                mos_column="predict_only",
                seg_length=self._args["ms_seg_length"],
                max_length=self._args.get("ms_max_segments"),
                to_memory=False,
                to_memory_workers=None,
                seg_hop_length=self._args.get("ms_seg_hop_length", 1),
                transform=None,
                ms_n_fft=self._args["ms_n_fft"],
                ms_hop_length=self._args["ms_hop_length"],
                ms_win_length=self._args["ms_win_length"],
                ms_n_mels=self._args["ms_n_mels"],
                ms_sr=self._args["ms_sr"],
                ms_fmax=self._args.get("ms_fmax"),
                ms_channel=self._args.get("ms_channel"),
                double_ended=False,
                dim=bool(self._args.get("model") == "NISQA_DIM"),
                filename_column_ref=None,
            )

            device = self._torch.device("cuda:0" if self.device_name.startswith("cuda") else self.device_name)
            with self._torch.inference_mode():
                if bool(self._args.get("model") == "NISQA_DIM"):
                    self._nl.predict_dim(
                        self._model,
                        ds,
                        self.settings.nisqa_batch_size,
                        device,
                        num_workers=0,
                    )
                    scores = {
                        id_by_filename[row.deg]: NISQAScores(
                            mos=float(row.mos_pred),
                            noisiness=float(row.noi_pred),
                            coloration=float(row.col_pred),
                            discontinuity=float(row.dis_pred),
                            loudness=float(row.loud_pred),
                        )
                        for row in ds.df.itertuples()
                    }
                else:
                    self._nl.predict_mos(
                        self._model,
                        ds,
                        self.settings.nisqa_batch_size,
                        device,
                        num_workers=0,
                    )
                    scores = {
                        id_by_filename[row.deg]: NISQAScores(
                            mos=float(row.mos_pred),
                            noisiness=None,
                            coloration=None,
                            discontinuity=None,
                            loudness=None,
                        )
                        for row in ds.df.itertuples()
                    }
            if self.device_name.startswith("cuda"):
                self._torch.cuda.synchronize()
            return scores


class DNSMOSBackend:
    INPUT_LENGTH_SECONDS = 9.01
    TARGET_SAMPLE_RATE = 16000

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ready = False
        self.detail = "disabled" if not settings.enable_dnsmos else "not_loaded"
        self.providers: list[str] = ["CPUExecutionProvider"]
        self._ort = None
        self._primary = None
        self._p808 = None

    def load(self) -> None:
        if not self.settings.enable_dnsmos:
            return
        download_if_missing(self.settings.dnsmos_primary_model_url, self.settings.dnsmos_primary_model_path)
        download_if_missing(self.settings.dnsmos_p808_model_url, self.settings.dnsmos_p808_model_path)

        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        if self.settings.prefer_gpu and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self._ort = ort
        self.providers = providers
        self._primary = ort.InferenceSession(
            self.settings.dnsmos_primary_model_path.as_posix(),
            providers=providers,
        )
        self._p808 = ort.InferenceSession(
            self.settings.dnsmos_p808_model_path.as_posix(),
            providers=providers,
        )
        self.ready = True
        self.detail = "loaded"
        self._warmup()

    def _warmup(self) -> None:
        silence = np.zeros(int(self.INPUT_LENGTH_SECONDS * self.TARGET_SAMPLE_RATE), dtype=np.float32)
        self.score_segment(silence, self.TARGET_SAMPLE_RATE)

    def _melspec(self, audio: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.TARGET_SAMPLE_RATE,
            n_fft=321,
            hop_length=160,
            n_mels=120,
        )
        mel = (librosa.power_to_db(mel, ref=np.max) + 40.0) / 40.0
        return mel.T.astype(np.float32)

    def score_segment(self, samples: np.ndarray, sample_rate: int) -> DNSMOSScores:
        if not self.settings.enable_dnsmos:
            return DNSMOSScores(ovrl=None, sig=None, bak=None, p808_mos=None)
        if not self.ready or not self._primary or not self._p808:
            raise RuntimeError("DNSMOS backend is not ready.")

        audio = np.asarray(samples, dtype=np.float32)
        if sample_rate != self.TARGET_SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.TARGET_SAMPLE_RATE)
        if audio.size == 0:
            return DNSMOSScores(ovrl=None, sig=None, bak=None, p808_mos=None)

        target_samples = int(self.INPUT_LENGTH_SECONDS * self.TARGET_SAMPLE_RATE)
        while len(audio) < target_samples:
            audio = np.concatenate([audio, audio], axis=0)

        num_hops = max(1, int(np.floor(len(audio) / self.TARGET_SAMPLE_RATE - self.INPUT_LENGTH_SECONDS) + 1))
        hop_samples = self.TARGET_SAMPLE_RATE
        sig_scores = []
        bak_scores = []
        ovr_scores = []
        p808_scores = []
        for index in range(num_hops):
            start = index * hop_samples
            end = start + target_samples
            window = audio[start:end]
            if len(window) < target_samples:
                continue
            primary_input = {"input_1": window[np.newaxis, :].astype(np.float32)}
            p808_input = {"input_1": self._melspec(window[:-160])[np.newaxis, :, :]}
            sig_raw, bak_raw, ovr_raw = self._primary.run(None, primary_input)[0][0]
            p808 = float(self._p808.run(None, p808_input)[0][0][0])
            sig = float(np.poly1d([-0.08397278, 1.22083953, 0.0052439])(sig_raw))
            bak = float(np.poly1d([-0.13166888, 1.60915514, -0.39604546])(bak_raw))
            ovr = float(np.poly1d([-0.06766283, 1.11546468, 0.04602535])(ovr_raw))
            sig_scores.append(sig)
            bak_scores.append(bak)
            ovr_scores.append(ovr)
            p808_scores.append(p808)

        return DNSMOSScores(
            ovrl=float(np.mean(ovr_scores)) if ovr_scores else None,
            sig=float(np.mean(sig_scores)) if sig_scores else None,
            bak=float(np.mean(bak_scores)) if bak_scores else None,
            p808_mos=float(np.mean(p808_scores)) if p808_scores else None,
        )
