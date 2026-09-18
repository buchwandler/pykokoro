"""Progress events and console reporting for managed runtime assets."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal, Protocol, TextIO

AssetProgressPhase = Literal[
    "download-start",
    "download-progress",
    "verify-start",
    "verify-complete",
    "download-complete",
    "cache-hit",
    "artifact-installed",
    "install-complete",
    "install-failed",
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
    bytes_total: int | None
    target: str
    message: str | None = None


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
    "artifact": "artifact",
}


class ConsoleAssetProgress:
    """Render runtime asset progress to a text stream."""

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
        self._last_bytes: dict[str, int] = {}
        self._download_started = False
        self._active_tty_artifact: str | None = None

    def __call__(self, event: AssetProgressEvent) -> None:
        artifact_id = event.artifact_id or event.filename or event.target
        if event.phase == "download-start":
            if not self._download_started:
                print(
                    "Required runtime assets are not cached; downloading them now.",
                    file=self.stream,
                )
                self._download_started = True
            self._last_percent[artifact_id] = -self.min_percent_step
            self._last_bytes[artifact_id] = 0
            self._active_tty_artifact = artifact_id if self._is_tty() else None
            size = (
                format_bytes(event.bytes_total) if event.bytes_total is not None else "size unknown"
            )
            print(
                f"Downloading {self._role(event.role)}: {self._artifact_name(event)} ({size})",
                file=self.stream,
                flush=True,
            )
            return

        if event.phase == "download-progress":
            if event.bytes_total is None:
                if (
                    event.bytes_done
                    and event.bytes_done - self._last_bytes.get(artifact_id, 0) < 1024 * 1024
                ):
                    return
                self._last_bytes[artifact_id] = event.bytes_done
                message = f"  {format_bytes(event.bytes_done)} / (size unknown)"
            else:
                percent = _percent(event.bytes_done, event.bytes_total)
                previous = self._last_percent.get(artifact_id, -self.min_percent_step)
                if percent < 100 and percent - previous < self.min_percent_step:
                    return
                self._last_percent[artifact_id] = percent
                message = (
                    f"  {percent:3d}% {format_bytes(event.bytes_done)} / "
                    f"{format_bytes(event.bytes_total)}"
                )
            if self._is_tty():
                print(f"\r{message}", end="", file=self.stream, flush=True)
            else:
                print(message, file=self.stream, flush=True)
            return

        if event.phase == "verify-start":
            self._finish_tty_line(artifact_id)
            print(
                f"Verifying {self._role(event.role)}: {self._artifact_name(event)}",
                file=self.stream,
                flush=True,
            )
            return

        if event.phase == "verify-complete":
            print(
                f"Verified {self._role(event.role)}: {self._artifact_name(event)}",
                file=self.stream,
                flush=True,
            )
            return

        if event.phase == "cache-hit":
            print(
                f"Using cached {self._role(event.role)}: {self._artifact_name(event)}",
                file=self.stream,
                flush=True,
            )
            return

        if event.phase == "install-complete":
            self._finish_tty_line(artifact_id)
            heading = (
                "Runtime assets already installed:"
                if event.message == "already installed"
                else "Runtime assets ready:"
            )
            print(heading, file=self.stream, flush=True)
            if event.target:
                print(f"  {event.target}", file=self.stream, flush=True)
            return

        if event.phase == "install-failed":
            self._finish_tty_line(artifact_id)
            print(
                f"Runtime asset installation failed: {event.target or self._artifact_name(event)}",
                file=self.stream,
                flush=True,
            )

    def _finish_tty_line(self, artifact_id: str) -> None:
        if self._is_tty() and self._active_tty_artifact == artifact_id:
            print(file=self.stream)
            self._active_tty_artifact = None

    def _is_tty(self) -> bool:
        isatty = getattr(self.stream, "isatty", None)
        return bool(isatty()) if callable(isatty) else False

    @staticmethod
    def _role(role: str) -> str:
        return _ROLE_NAMES.get(role, role or "artifact")

    @staticmethod
    def _artifact_name(event: AssetProgressEvent) -> str:
        return event.filename or event.artifact_id or "artifact"


def _percent(done: int, total: int | None) -> int:
    if total is None or total <= 0:
        return 0
    return min(100, max(0, done * 100 // total))
