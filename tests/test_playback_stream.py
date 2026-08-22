"""Hardware-free tests for the persistent direct playback queue."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from pykokoro.playback import SoundDevicePlayer


class FakeOutputStream:
    def __init__(self, *, fail: BaseException | None = None, block: bool = False, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.fail = fail
        self.block = block
        self.started = False
        self.stopped = False
        self.closed = False
        self.writes: list[np.ndarray] = []
        self.write_started = threading.Event()
        self.release_write = threading.Event()

    def start(self) -> None:
        self.started = True

    def write(self, audio: np.ndarray) -> None:
        self.write_started.set()
        if self.block:
            self.release_write.wait(timeout=2)
        if self.fail is not None:
            raise self.fail
        self.writes.append(audio.copy())

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


def _install_backend(monkeypatch: pytest.MonkeyPatch, stream: FakeOutputStream) -> None:
    monkeypatch.setitem(
        __import__("sys").modules,
        "sounddevice",
        SimpleNamespace(OutputStream=lambda **kwargs: _make_stream(stream, kwargs)),
    )


def _make_stream(stream: FakeOutputStream, kwargs: dict[str, object]) -> FakeOutputStream:
    stream.kwargs = kwargs
    return stream


def test_multiple_units_use_one_stream_in_order_and_copy_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = FakeOutputStream()
    _install_backend(monkeypatch, stream)
    player = SoundDevicePlayer(24_000, device="test-device")
    first = np.array([1.0, 2.0], dtype=np.float64)
    second = np.array([3.0, 4.0], dtype=np.float32)

    player.start()
    player.submit(first)
    first[:] = 0
    player.submit(second)
    player.drain()
    player.close()
    player.close()

    assert stream.kwargs == {"samplerate": 24_000, "device": "test-device"}
    assert stream.started and stream.stopped and stream.closed
    assert len(stream.writes) == 2
    np.testing.assert_array_equal(stream.writes[0], [1.0, 2.0])
    np.testing.assert_array_equal(stream.writes[1], [3.0, 4.0])
    assert stream.writes[0].dtype == np.float32


def test_queue_is_bounded_and_submit_applies_backpressure(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = FakeOutputStream(block=True)
    _install_backend(monkeypatch, stream)
    player = SoundDevicePlayer(24_000, queue_size=1).start()

    player.submit(np.ones(2, dtype=np.float32))
    assert stream.write_started.wait(timeout=1)
    player.submit(np.full(2, 2, dtype=np.float32))
    assert player._queue.qsize() == 1

    stream.release_write.set()
    player.drain()
    player.close()
    assert len(stream.writes) == 2


def test_stream_errors_are_raised_to_producer_and_close(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = FakeOutputStream(fail=RuntimeError("device failed"))
    _install_backend(monkeypatch, stream)
    player = SoundDevicePlayer(24_000).start()
    player.submit(np.ones(2, dtype=np.float32))

    deadline = time.monotonic() + 1
    while not player._error and time.monotonic() < deadline:
        time.sleep(0.01)

    with pytest.raises(RuntimeError, match="device failed"):
        player.drain()
    with pytest.raises(RuntimeError, match="device failed"):
        player.close()
    assert stream.closed


def test_context_manager_closes_on_generation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = FakeOutputStream()
    _install_backend(monkeypatch, stream)

    with pytest.raises(ValueError, match="generation"), SoundDevicePlayer(24_000) as player:
        player.submit(np.ones(2, dtype=np.float32))
        raise ValueError("generation failed")

    assert stream.stopped and stream.closed


def test_player_validates_configuration_and_audio() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        SoundDevicePlayer(0)
    with pytest.raises(ValueError, match="queue_size"):
        SoundDevicePlayer(24_000, queue_size=0)

    player = SoundDevicePlayer(24_000)
    with pytest.raises(RuntimeError, match="not been started"):
        player.submit(np.ones(1, dtype=np.float32))
