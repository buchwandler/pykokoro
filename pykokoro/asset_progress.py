"""Progress events and console reporting for managed runtime assets."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal, Protocol, TextIO

AssetProgressPhase = Literal[
    "download-start",
    "download-progress",
    "verify-start",
    "download-complete",
]


@dataclass(frozen=True, slots=True)
class AssetProgressEvent:
    """A lifecycle or byte-progress update for one runtime artifact."""

    phase: AssetProgressPhase
    model_id: str
    distribution_id: str
    artifact_id: str
    role: str
    filename: str
    bytes_done: int
    bytes_total: int
    target: str


class AssetProgressCallback(Protocol):
    """Callable interface for receiving runtime asset progress events."""

    def __call__(self, event: AssetProgressEvent) -> None: ...


def format_bytes(value: int) -> str:
    """Format a byte count using binary units."""
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


_ROLE_NAMES = {
    "model": "model",
    "voice": "voice",
    "voices": "voices",
    "config": "config",
    "vocabulary": "vocabulary",
    "bundle": "bundle",
}


class ConsoleAssetProgress:
    """Render runtime asset download progress to a text stream."""

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        min_percent_step: int = 5,
    ) -> None:
        if min_percent_step <= 0 or min_percent_step > 100:
            raise ValueError("min_percent_step must be between 1 and 100")
        self.stream = stream if stream is not None else sys.stderr
        self.min_percent_step = min_percent_step
        self._last_percent: dict[str, int] = {}
        self._download_started = False
        self._active_tty_artifact: str | None = None

    def __call__(self, event: AssetProgressEvent) -> None:
        if event.phase == "download-start":
            if not self._download_started:
                print(
                    "Required runtime assets are not cached; downloading them now.",
                    file=self.stream,
                )
                self._download_started = True
            self._last_percent[event.artifact_id] = -self.min_percent_step
            self._active_tty_artifact = event.artifact_id if self._is_tty() else None
            print(
                f"Downloading {self._role(event.role)}: {event.filename} "
                f"({format_bytes(event.bytes_total)})",
                file=self.stream,
                flush=True,
            )
            return

        if event.phase == "download-progress":
            percent = _percent(event.bytes_done, event.bytes_total)
            previous = self._last_percent.get(event.artifact_id, -self.min_percent_step)
            if percent < 100 and percent - previous < self.min_percent_step:
                return
            self._last_percent[event.artifact_id] = percent
            if self._is_tty():
                print(
                    f"\r  {percent:3d}% {format_bytes(event.bytes_done)} / "
                    f"{format_bytes(event.bytes_total)}",
                    end="",
                    file=self.stream,
                    flush=True,
                )
            return

        if event.phase == "verify-start":
            if self._is_tty() and self._active_tty_artifact == event.artifact_id:
                print(file=self.stream)
                self._active_tty_artifact = None
            print(
                f"Verifying {self._role(event.role)}: {event.filename}",
                file=self.stream,
                flush=True,
            )
            return

        if event.phase == "download-complete":
            if self._is_tty() and self._active_tty_artifact == event.artifact_id:
                print(file=self.stream)
                self._active_tty_artifact = None
            print(f"Ready: {event.filename}", file=self.stream, flush=True)

    def _is_tty(self) -> bool:
        isatty = getattr(self.stream, "isatty", None)
        return bool(isatty()) if callable(isatty) else False

    @staticmethod
    def _role(role: str) -> str:
        return _ROLE_NAMES.get(role, role)


def _percent(done: int, total: int) -> int:
    if total <= 0:
        return 0
    return min(100, max(0, done * 100 // total))
