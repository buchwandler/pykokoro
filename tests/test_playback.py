"""Hardware-free tests for direct playback APIs."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pykokoro.playback import play_audio
from pykokoro.synthesis_types import RenderedSegment


def _rendered_result(audio: np.ndarray) -> RenderedSegment:
    return RenderedSegment(
        id="request-1",
        audio=audio,
        sample_rate=24_000,
        text="hello",
        language="en-us",
        voice="af_heart",
        phonemes="həlˈoʊ",
        token_ids=(1, 2, 3),
    )


def test_play_audio_delegates_waveform_and_options(monkeypatch: pytest.MonkeyPatch) -> None:
    audio = np.array([0.0, 0.1, -0.1], dtype=np.float32)
    calls: list[tuple[object, dict[str, object]]] = []
    backend = SimpleNamespace(play=lambda samples, **kwargs: calls.append((samples, kwargs)))
    monkeypatch.setitem(sys.modules, "sounddevice", backend)

    play_audio(audio, 24_000, device="test-device")

    assert calls == [(audio, {"samplerate": 24_000, "blocking": True, "device": "test-device"})]


def test_play_audio_omits_unrequested_device(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    backend = SimpleNamespace(play=lambda samples, **kwargs: calls.append(kwargs))
    monkeypatch.setitem(sys.modules, "sounddevice", backend)

    play_audio(np.ones(2, dtype=np.float32), 24_000)

    assert calls == [{"samplerate": 24_000, "blocking": True}]


def test_play_audio_validates_before_import() -> None:
    with pytest.raises(RuntimeError, match="empty|released"):
        play_audio(np.empty(0, dtype=np.float32), 24_000)

    with pytest.raises(ValueError, match="positive"):
        play_audio(np.ones(1, dtype=np.float32), 0)


def test_missing_sounddevice_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = __import__

    def blocked_import(name: str, *args: object, **kwargs: object):
        if name == "sounddevice":
            raise ModuleNotFoundError("blocked sounddevice", name="sounddevice")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    monkeypatch.setattr("builtins.__import__", blocked_import)

    with pytest.raises(ImportError, match=r"sounddevice.*pykokoro\[playback\]"):
        play_audio(np.ones(2, dtype=np.float32), 24_000)


def test_rendered_segment_play_delegates_to_shared_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    audio = np.ones(3, dtype=np.float32)
    result = _rendered_result(audio)
    calls: list[tuple[np.ndarray, int, str | None]] = []

    def fake_play(samples: np.ndarray, sample_rate: int, *, device: str | None = None) -> None:
        calls.append((samples, sample_rate, device))

    monkeypatch.setattr("pykokoro.playback.play_audio", fake_play)
    result.play(device="one")

    assert calls == [(audio, 24_000, "one")]


def test_import_boundary_without_sounddevice() -> None:
    code = """
import builtins
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'sounddevice':
        raise ModuleNotFoundError('blocked sounddevice', name='sounddevice')
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
import pykokoro
from pykokoro.synthesis_types import RenderedSegment
import numpy as np
RenderedSegment('request-1', np.ones(1, dtype=np.float32), 24000, 'x', 'en-us', None, '', ())
print('ok')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"
