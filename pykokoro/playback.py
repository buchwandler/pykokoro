"""Optional direct playback helpers backed by :mod:`sounddevice`."""

from __future__ import annotations

import queue
import threading
from typing import Any

import numpy as np

__all__ = ["SoundDevicePlayer", "play_audio"]


def _import_sounddevice() -> Any:
    try:
        import sounddevice as sd
    except ImportError:
        raise ImportError(
            "Audio playback requires the optional 'sounddevice' dependency. "
            'Install it with: pip install "pykokoro[playback]"'
        ) from None
    return sd


def _validate_audio(audio: np.ndarray, *, allow_copy: bool = False) -> np.ndarray:
    samples = np.asarray(audio)
    if samples.size == 0:
        raise RuntimeError("Audio is empty or has already been released")
    if samples.ndim not in (1, 2):
        raise ValueError("audio must be mono or two-dimensional")
    if allow_copy:
        return np.ascontiguousarray(samples, dtype=np.float32).copy()
    return samples


def play_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    device: int | str | None = None,
) -> None:
    """Play a waveform synchronously without creating an audio file.

    Requires the optional ``sounddevice`` dependency. The input waveform is
    passed to the backend without amplitude, dtype, or sample-rate conversion.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    samples = _validate_audio(audio)
    kwargs: dict[str, Any] = {"samplerate": sample_rate, "blocking": True}
    if device is not None:
        kwargs["device"] = device
    _import_sounddevice().play(samples, **kwargs)


_SENTINEL = object()


class SoundDevicePlayer:
    """Queue generated waveforms on one persistent ``sounddevice`` stream.

    Submitted arrays are copied into contiguous ``float32`` buffers before they
    enter the bounded queue. This makes ownership safe when a source result is
    released after :meth:`submit` returns. ``submit`` blocks when the queue is
    full, providing backpressure to the synthesis producer.
    """

    def __init__(
        self,
        sample_rate: int,
        *,
        device: int | str | None = None,
        queue_size: int = 2,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        self.sample_rate = sample_rate
        self.device = device
        self.queue_size = queue_size
        self._queue: queue.Queue[np.ndarray | object] = queue.Queue(maxsize=queue_size)
        self._stream: Any = None
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._state_lock = threading.Lock()
        self._closed = False

    def start(self) -> SoundDevicePlayer:
        """Open the output stream and start its playback worker."""
        with self._state_lock:
            if self._thread is not None:
                return self
            if self._closed:
                raise RuntimeError("SoundDevicePlayer is closed")

            sd = _import_sounddevice()
            kwargs: dict[str, Any] = {"samplerate": self.sample_rate}
            if self.device is not None:
                kwargs["device"] = self.device
            self._stream = sd.OutputStream(**kwargs)
            try:
                self._stream.start()
            except BaseException:
                self._stream.close()
                self._stream = None
                raise
            self._thread = threading.Thread(
                target=self._run,
                name="pykokoro-sounddevice-player",
                daemon=True,
            )
            self._thread.start()
        return self

    def submit(self, audio: np.ndarray) -> None:
        """Copy and queue audio, blocking while the bounded queue is full."""
        with self._state_lock:
            if self._thread is None:
                raise RuntimeError("SoundDevicePlayer has not been started")
            if self._closed:
                raise RuntimeError("SoundDevicePlayer is closed")
            self._raise_error()

        owned = _validate_audio(audio, allow_copy=True)
        while True:
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("SoundDevicePlayer is closed")
                self._raise_error()
            try:
                self._queue.put(owned, timeout=0.05)
                return
            except queue.Full:
                continue

    def drain(self) -> None:
        """Wait until all submitted audio has been consumed by the stream."""
        self._queue.join()
        self._raise_error()

    def close(self) -> None:
        """Stop playback and release the output stream; safe to call repeatedly."""
        with self._state_lock:
            if self._thread is None:
                self._closed = True
                self._raise_error()
                return
            if self._closed and self._stream is None:
                self._raise_error()
                return
            if self._closed:
                thread = self._thread
            else:
                self._closed = True
                thread = self._thread
                has_error = self._error is not None
                if not has_error:
                    self._put_sentinel()

        thread.join()
        stream = self._stream
        close_error: BaseException | None = None
        if stream is not None:
            try:
                stream.stop()
            except BaseException as exc:
                close_error = exc
            try:
                stream.close()
            except BaseException as exc:
                close_error = close_error or exc
        with self._state_lock:
            self._stream = None
        if close_error is not None and self._error is None:
            self._error = close_error
        self._raise_error()

    def __enter__(self) -> SoundDevicePlayer:
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _put_sentinel(self) -> None:
        while True:
            try:
                self._queue.put(_SENTINEL, timeout=0.05)
                return
            except queue.Full:
                if self._error is not None:
                    return

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _SENTINEL:
                    return
                assert isinstance(item, np.ndarray)
                self._stream.write(item)
            except BaseException as exc:
                self._error = exc
                self._discard_pending()
                return
            finally:
                self._queue.task_done()

    def _discard_pending(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
            else:
                self._queue.task_done()

    def _raise_error(self) -> None:
        if self._error is not None:
            raise self._error
