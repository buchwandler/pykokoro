from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pykokoro.exceptions import ConfigurationError
from pykokoro.model_registry import RuntimeArtifact, RuntimeDistribution, RuntimeModel
from pykokoro.runtime.dispatcher import UnsupportedRuntimeLayout, create_runtime
from pykokoro.runtime.model_assets import ResolvedRuntimeAssets
from pykokoro.runtime.thai_wayu import ThaiWayuRuntime
from pykokoro.types import PhonemeSegment
from pykokoro.voice_manager import VoiceBlend


def _assets(tmp_path: Path, layout: str = "split-onnx-v1") -> ResolvedRuntimeAssets:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"vocab": {"a": 1}}', encoding="utf-8")
    source = tmp_path / "source-params.npz"
    np.savez(source, weight=np.zeros((1, 9)), bias=np.zeros(1), window=np.ones(20))
    voices = tmp_path / "voices.npz"
    np.savez(voices, default=np.zeros((1, 256), dtype=np.float32))
    paths = {"manifest": manifest, "source_params": source, "voices": voices}
    artifacts = [
        RuntimeArtifact(
            f"model-{component}",
            "model",
            "onnx",
            f"https://example/{component}",
            f"{component}.onnx",
            1,
            "0" * 64,
            quality="fp32",
            component=component,
        )
        for component in ("prosody", "curves", "decoder")
    ]
    artifacts.extend(
        [
            RuntimeArtifact(
                "manifest", "config", "json", "https://example/manifest", manifest.name, 1, "0" * 64
            ),
            RuntimeArtifact(
                "source", "metadata", "numpy-npz", "https://example/source", source.name, 1, "0" * 64
            ),
            RuntimeArtifact(
                "voices", "voices", "numpy-npz", "https://example/voices", voices.name, 1, "0" * 64
            ),
        ]
    )
    materialized = dict(paths)
    for artifact in artifacts:
        path = tmp_path / artifact.local_name
        if artifact.role == "model":
            path.write_bytes(b"0")
        else:
            path = next(value for value in paths.values() if value.name == artifact.local_name)
        materialized[artifact.id] = path
    model = RuntimeModel(
        "thai",
        {"runtime": {"layout": layout, "max_tokens": 510, "default_voice": "default", "voices": ["default"]}},
        (RuntimeDistribution("dist", "github-release", "https", True, tuple(artifacts)),),
    )
    return ResolvedRuntimeAssets(
        "thai", "dist", "github-release", layout, materialized, model, model.distributions[0]
    )


def test_thai_dispatch_uses_registry_component_ids(tmp_path: Path) -> None:
    assets = _assets(tmp_path)
    requested: list[str] = []

    def factory(path: Path):
        requested.append(path.name)
        return object()

    runtime = create_runtime(assets, session_factory=factory)

    assert isinstance(runtime, ThaiWayuRuntime)
    assert requested == ["prosody.onnx", "curves.onnx", "decoder.onnx"]


def test_unknown_runtime_layout_is_explicit(tmp_path: Path) -> None:
    assets = _assets(tmp_path, layout="future-layout")

    with pytest.raises(UnsupportedRuntimeLayout, match="future-layout"):
        create_runtime(assets)


class FakeThaiRuntime:
    def __init__(self) -> None:
        self.voices = {
            "f_young_clear": np.zeros((1, 256), dtype=np.float32),
            "f_young_warm": np.ones((1, 256), dtype=np.float32),
        }
        self.calls: list[tuple[str, str, float]] = []

    def synthesize(self, text: str, voice: str, *, speed: float) -> np.ndarray:
        if voice not in self.voices:
            raise KeyError(f"Unknown Thai voice: {voice}")
        self.calls.append((text, voice, speed))
        return np.asarray([len(self.calls)], dtype=np.float32)


def _fake_kokoro(monkeypatch):
    from pykokoro.onnx_backend import Kokoro

    kokoro = Kokoro.__new__(Kokoro)
    kokoro._runtime = FakeThaiRuntime()
    kokoro._voice_manager = None
    monkeypatch.setattr(kokoro, "_init_kokoro", lambda: None)
    return kokoro


def _segment(text: str, voice_name: str | None = None) -> PhonemeSegment:
    return PhonemeSegment(
        id="seg",
        segment_id="seg",
        phoneme_id=0,
        text=text,
        phonemes="a",
        tokens=[1],
        lang="th",
        voice_name=voice_name,
    )


def test_split_runtime_resolves_voice_without_voice_manager(monkeypatch) -> None:
    kokoro = _fake_kokoro(monkeypatch)

    style = kokoro.resolve_voice_style("f_young_clear")

    assert style.shape == (1, 1, 256)


def test_split_runtime_honors_configured_and_segment_voices(monkeypatch) -> None:
    kokoro = _fake_kokoro(monkeypatch)
    segments = [_segment("one"), _segment("two", "f_young_warm")]

    kokoro.generate_raw_audio_segments(
        segments,
        np.zeros((1, 1), dtype=np.float32),
        1.0,
        None,
        default_voice_name="f_young_clear",
    )

    assert kokoro._runtime.calls == [
        ("one", "f_young_clear", 1.0),
        ("two", "f_young_warm", 1.0),
    ]


def test_split_runtime_unknown_voice_fails_clearly(monkeypatch) -> None:
    kokoro = _fake_kokoro(monkeypatch)

    with pytest.raises(KeyError, match="Unknown Thai voice"):
        kokoro.generate_raw_audio_segments(
            [_segment("one")],
            np.zeros((1, 1), dtype=np.float32),
            1.0,
            None,
            default_voice_name="missing",
        )


@pytest.mark.parametrize("voice", [VoiceBlend([("f_young_clear", 1.0)]), np.zeros(256)])
def test_split_runtime_rejects_voice_styles(monkeypatch, voice) -> None:
    kokoro = _fake_kokoro(monkeypatch)

    with pytest.raises(ConfigurationError, match="not supported by this runtime layout"):
        kokoro.resolve_voice_style(voice)
