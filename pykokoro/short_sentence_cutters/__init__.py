"""Short-sentence phrase audio cutters."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .energy_valley import cut_with_energy_valley

PhraseCutter = Callable[[np.ndarray, dict[str, object]], np.ndarray | None]

_CUTTERS: dict[str, PhraseCutter] = {
    "energy-valley": cut_with_energy_valley,
}


def cut_phrase_audio(audio: np.ndarray, metadata: dict[str, object]) -> np.ndarray | None:
    """Cut phrase-generated audio using the configured cutter."""
    cutter_name = str(metadata.get("cutter", "energy-valley"))
    cutter = _CUTTERS.get(cutter_name)
    if cutter is None:
        return None
    return cutter(audio, metadata)
