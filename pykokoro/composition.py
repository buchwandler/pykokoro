"""AudioCompose integration for PyKokoro batch output."""

from __future__ import annotations

from typing import Any

from audiocompose import AudioJob, Composer, CompositionResult


def compose_audio_job(job: AudioJob) -> CompositionResult:
    """Compose a prepared job through AudioCompose's single batch path."""
    return Composer(sample_rate=job.output.sample_rate).compose(job)


def write_audio_job(job: AudioJob, path: str) -> Any:
    """Compose a job and write its final waveform as a WAV file."""
    return Composer(sample_rate=job.output.sample_rate).to_wav(job, path)


__all__ = ["compose_audio_job", "write_audio_job"]
