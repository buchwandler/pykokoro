"""Runtime for the registry's ``split-onnx-v1`` Thai Wayu model."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..model_registry import ModelRegistryError
from .model_assets import ResolvedRuntimeAssets

HARMONICS = 9
UPSAMPLE_SCALE = 300
SAMPLE_RATE = 24000
VOICED_THRESHOLD = 10.0
SINE_AMP = 0.1
NOISE_STD = 0.003
N_FFT = 20
HOP = 5


class ThaiWayuRuntime:
    """Small, explicit inference adapter for the Thai split graph."""

    COMPONENTS = ("prosody", "curves", "decoder")

    def __init__(
        self,
        assets: ResolvedRuntimeAssets | Mapping[str, Path],
        *,
        session_factory: Callable[[Path], Any] | None = None,
    ) -> None:
        if isinstance(assets, ResolvedRuntimeAssets):
            self.assets: ResolvedRuntimeAssets | None = assets
            paths = self._paths_from_assets(assets)
            self.max_tokens = assets.model.max_tokens
        else:
            self.assets = None
            paths = dict(assets)
            self.max_tokens = 510

        missing = [component for component in self.COMPONENTS if component not in paths]
        if missing:
            raise ModelRegistryError(
                "Thai Wayu distribution is missing model components: " + ", ".join(missing)
            )
        self.asset_paths = paths
        manifest_path = paths.get("manifest") or paths.get("config")
        if manifest_path is None:
            raise ModelRegistryError("Thai Wayu distribution is missing its ONNX manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.vocab = self._manifest_vocab(manifest)
        self.sessions = {
            component: (session_factory or self._default_session)(paths[component])
            for component in self.COMPONENTS
        }
        source_path = paths.get("source_params")
        voices_path = paths.get("voices")
        if source_path is None or voices_path is None:
            raise ModelRegistryError(
                "Thai Wayu distribution is missing source parameters or voices"
            )
        with np.load(source_path, allow_pickle=False) as source:
            self.source_weight = np.asarray(source["weight"])
            self.source_bias = np.asarray(source["bias"])
            self.window = np.asarray(source["window"])
        with np.load(voices_path, allow_pickle=False) as voices:
            self.voices = {name: np.asarray(voices[name]) for name in voices.files}
        self._validate_assets()

    @staticmethod
    def _default_session(path: Path) -> Any:
        import onnxruntime as ort

        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    @classmethod
    def _paths_from_assets(cls, assets: ResolvedRuntimeAssets) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for artifact in assets.distribution.artifacts:
            path = assets.artifacts[artifact.id]
            if artifact.role == "model" and artifact.component in cls.COMPONENTS:
                paths[artifact.component] = path
            elif artifact.role == "config":
                paths["config"] = path
            elif artifact.role == "metadata" and artifact.local_name.startswith("source-params"):
                paths["source_params"] = path
            elif artifact.role == "voices":
                paths["voices"] = path
        if "config" in paths:
            paths["manifest"] = paths["config"]
        return paths

    @staticmethod
    def _manifest_vocab(manifest: Mapping[str, Any]) -> dict[str, int]:
        vocab = manifest.get("vocab")
        if not isinstance(vocab, Mapping):
            raise ModelRegistryError("Thai Wayu manifest has no vocabulary")
        return {str(symbol): int(token) for symbol, token in vocab.items()}

    def _validate_assets(self) -> None:
        for name, session in self.sessions.items():
            inputs: list[Any] = getattr(session, "get_inputs", lambda: [])()
            if not inputs:
                continue
            names = {str(getattr(item, "name", "")) for item in inputs}
            if name == "prosody" and not {"input_ids", "style_dur", "speed"} <= names:
                raise ModelRegistryError(
                    "Thai Wayu prosody session has an incompatible input contract"
                )
            if name == "curves" and not {"en", "style_dur"} <= names:
                raise ModelRegistryError(
                    "Thai Wayu curves session has an incompatible input contract"
                )
            if (
                name == "decoder"
                and not {"asr", "f0_curve", "n_curve", "style_acou", "har"} <= names
            ):
                raise ModelRegistryError(
                    "Thai Wayu decoder session has an incompatible input contract"
                )

    def phonemize(self, text: str) -> str:
        import kokorog2p

        result = kokorog2p.phonemize(text, language="th", return_ids=False)
        phonemes = result.phonemes
        return phonemes if isinstance(phonemes, str) else "".join(phonemes)

    def synthesize(
        self, text: str, voice: str, *, speed: float = 1.0, seed: int = 1234
    ) -> np.ndarray:
        ids = [self.vocab[phoneme] for phoneme in self.phonemize(text) if phoneme in self.vocab]
        if not ids:
            raise RuntimeError("Thai frontend produced no vocabulary symbols")
        if len(ids) > self.max_tokens:
            raise ValueError(
                f"Thai frontend produced {len(ids)} tokens, maximum is {self.max_tokens}"
            )
        if voice not in self.voices:
            raise KeyError(f"Unknown Thai voice: {voice}")

        style = np.asarray(
            self.voices[voice][min(len(ids) - 1, self.voices[voice].shape[0] - 1)], dtype=np.float32
        ).reshape(1, 256)
        style_dur, style_acou = style[:, 128:], style[:, :128]
        pred_dur, d, t_en = self.sessions["prosody"].run(
            None,
            {
                "input_ids": np.asarray([[0, *ids, 0]], dtype=np.int64),
                "style_dur": style_dur,
                "speed": np.asarray([speed], dtype=np.float32),
            },
        )
        index = np.repeat(np.arange(pred_dur.shape[0], dtype=np.int64), pred_dur)
        en = np.ascontiguousarray(d.transpose(0, 2, 1)[:, :, index])
        asr = np.ascontiguousarray(t_en[:, :, index])
        f0_curve, n_curve = self.sessions["curves"].run(None, {"en": en, "style_dur": style_dur})
        har = self._harmonic_source(f0_curve, seed)
        (audio,) = self.sessions["decoder"].run(
            None,
            {
                "asr": asr,
                "f0_curve": f0_curve,
                "n_curve": n_curve,
                "style_acou": style_acou,
                "har": har,
            },
        )
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def _harmonic_source(self, f0_curve: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        f0 = np.repeat(np.asarray(f0_curve, dtype=np.float64), UPSAMPLE_SCALE, axis=1)[..., None]
        rad = f0 * np.arange(1, HARMONICS + 1, dtype=np.float64) / SAMPLE_RATE
        rad -= np.floor(rad)
        rand_ini = rng.random((1, HARMONICS))
        rand_ini[:, 0] = 0.0
        rad[:, 0, :] += rand_ini
        phase = np.cumsum(self._resample(rad, 1 / UPSAMPLE_SCALE), axis=1) * 2 * np.pi
        phase = self._resample(phase * UPSAMPLE_SCALE, UPSAMPLE_SCALE)
        voiced = (f0 > VOICED_THRESHOLD).astype(np.float64)
        amplitude = voiced * NOISE_STD + (1 - voiced) * SINE_AMP / 3
        noise = rng.standard_normal((1, f0.shape[1], HARMONICS))
        waves = np.sin(phase) * SINE_AMP * voiced + amplitude * noise
        merged = np.tanh(waves @ self.source_weight.T + self.source_bias)
        return self._stft(merged[0, :, 0].astype(np.float32))

    @staticmethod
    def _resample(values: np.ndarray, scale: float) -> np.ndarray:
        length = values.shape[1]
        output_length = int(length * scale)
        source = (np.arange(output_length, dtype=np.float64) + 0.5) / scale - 0.5
        np.clip(source, 0, length - 1, out=source)
        lower = np.floor(source).astype(np.int64)
        upper = np.minimum(lower + 1, length - 1)
        weight = (source - lower)[None, :, None]
        return values[:, lower] * (1 - weight) + values[:, upper] * weight

    def _stft(self, audio: np.ndarray) -> np.ndarray:
        padded = np.pad(audio, N_FFT // 2, mode="reflect")
        frames = np.lib.stride_tricks.as_strided(
            padded,
            shape=((len(padded) - N_FFT) // HOP + 1, N_FFT),
            strides=(padded.strides[0] * HOP, padded.strides[0]),
        )
        spectrum = np.fft.rfft(frames * self.window, axis=1).T[None]
        return np.concatenate([np.abs(spectrum), np.angle(spectrum)], axis=1).astype(np.float32)
