"""Runtime protocol used by PyKokoro's producer-side audio generator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np


class KokoroInferenceRuntime(Protocol):
    @property
    def sample_rate(self) -> int: ...

    def infer(
        self,
        token_ids: Sequence[int],
        *,
        style: np.ndarray,
        speed: float,
        seed: int | None = None,
    ) -> Any: ...

    def close(self) -> None: ...
