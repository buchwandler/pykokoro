"""Runtime protocol used by PyKokoro's producer-side audio generator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np


class KokoroInferenceRuntime(Protocol):
    @property
    def sample_rate(self) -> int: ...

    def enroll_voice(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        enroller: str,
        options: Mapping[str, Any] | None = None,
        name: str | None = None,
    ) -> Any: ...

    def prepare_reference(
        self,
        reference_token_ids: Sequence[int],
        *,
        audio_24k: np.ndarray,
        audio_16k: np.ndarray,
    ) -> Any: ...

    def infer(
        self,
        token_ids: Sequence[int],
        *,
        style: np.ndarray | None = None,
        reference: Any | None = None,
        speed: float = 1.0,
        seed: int | None = None,
    ) -> Any: ...

    def close(self) -> None: ...
