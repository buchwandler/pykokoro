from __future__ import annotations

import numpy as np

from pykokoro.short_sentence_cutters.energy_valley import find_energy_valley_cut_bounds
from pykokoro.short_sentence_cutters.timestamp_adaptive import (
    cut_with_timestamp_adaptive,
    find_timestamp_adaptive_cut_bounds,
)


def _metadata(*, left: bool = True, right: bool = True) -> dict[str, object]:
    return {
        "target_start_ts": 1600 / 24000,
        "target_end_ts": 2400 / 24000,
        "previous_token_end_ts": 600 / 24000,
        "next_token_start_ts": 3400 / 24000,
        "has_left_context": left,
        "has_right_context": right,
        "frame_duration_ms": 5,
        "energy_threshold": 0.05,
        "min_silence_seconds": 0.02,
    }


def test_timestamp_adaptive_preserves_strict_energy_bounds() -> None:
    audio = np.zeros(4000, dtype=np.float32)
    metadata = _metadata()

    expected = find_energy_valley_cut_bounds(audio, metadata)
    actual = find_timestamp_adaptive_cut_bounds(audio, metadata)

    assert expected is not None
    assert actual == expected
    assert metadata["cut_strategy"] == "energy-valley"


def test_timestamp_adaptive_uses_smooth_context_points_after_strict_failure(monkeypatch) -> None:
    audio = np.ones(4000, dtype=np.float32)
    metadata = _metadata()
    calls: list[tuple[int, int, int]] = []

    monkeypatch.setattr(
        "pykokoro.short_sentence_cutters.timestamp_adaptive.find_energy_valley_cut_bounds",
        lambda _audio, _metadata: None,
    )

    def smooth_point(_audio, *, start, end, anchor, window_length):
        calls.append((start, end, anchor))
        return start if anchor == 1600 else end - 1

    monkeypatch.setattr(
        "pykokoro.short_sentence_cutters.timestamp_adaptive.find_smooth_cut_point",
        smooth_point,
    )

    cut = cut_with_timestamp_adaptive(audio, metadata)

    assert cut is not None
    assert metadata["cut_strategy"] == "timestamp-smooth"
    assert metadata["cut_left"] == calls[0][0]
    assert metadata["cut_right"] == calls[1][1] - 1
    assert calls[0][1] <= 1600
    assert calls[1][0] >= 2400
    assert metadata["cut_left"] < 1600
    assert metadata["cut_right"] > 2400


def test_timestamp_adaptive_shrinks_guard_for_narrow_context(monkeypatch) -> None:
    audio = np.ones(3000, dtype=np.float32)
    metadata = _metadata()
    metadata.update(
        {
            "previous_token_end_ts": 1500 / 24000,
            "next_token_start_ts": 2500 / 24000,
        }
    )
    intervals: list[tuple[int, int]] = []

    monkeypatch.setattr(
        "pykokoro.short_sentence_cutters.timestamp_adaptive.find_energy_valley_cut_bounds",
        lambda _audio, _metadata: None,
    )

    def smooth_point(_audio, *, start, end, anchor, window_length):
        intervals.append((start, end))
        return start if anchor == 1600 else end - 1

    monkeypatch.setattr(
        "pykokoro.short_sentence_cutters.timestamp_adaptive.find_smooth_cut_point",
        smooth_point,
    )

    assert find_timestamp_adaptive_cut_bounds(audio, metadata) is not None
    assert intervals == [(1500, 1567), (2433, 2500)]


def test_timestamp_adaptive_preserves_one_sided_context(monkeypatch) -> None:
    audio = np.ones(3000, dtype=np.float32)
    metadata = _metadata(left=False, right=True)
    monkeypatch.setattr(
        "pykokoro.short_sentence_cutters.timestamp_adaptive.find_energy_valley_cut_bounds",
        lambda _audio, _metadata: None,
    )
    monkeypatch.setattr(
        "pykokoro.short_sentence_cutters.timestamp_adaptive.find_smooth_cut_point",
        lambda _audio, *, start, end, anchor, window_length: end - 1,
    )

    bounds = find_timestamp_adaptive_cut_bounds(audio, metadata)

    assert bounds is not None
    assert bounds[0] == 0
    assert bounds[1] > 2400


def test_timestamp_adaptive_rejects_invalid_timestamp_geometry() -> None:
    audio = np.ones(4000, dtype=np.float32)
    metadata = _metadata()
    metadata["target_end_ts"] = metadata["target_start_ts"]

    assert find_timestamp_adaptive_cut_bounds(audio, metadata) is None


def test_timestamp_adaptive_accepts_zero_width_context_gaps(monkeypatch) -> None:
    audio = np.ones(3000, dtype=np.float32)
    metadata = _metadata()
    metadata.update(
        {
            "previous_token_end_ts": 1000 / 24000,
            "target_start_ts": 1000 / 24000,
            "target_end_ts": 2000 / 24000,
            "next_token_start_ts": 2000 / 24000,
        }
    )
    intervals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "pykokoro.short_sentence_cutters.timestamp_adaptive.find_energy_valley_cut_bounds",
        lambda _audio, _metadata: None,
    )
    def smooth_point(_audio, *, start, end, anchor, window_length):
        intervals.append((start, end))
        return anchor
    monkeypatch.setattr(
        "pykokoro.short_sentence_cutters.timestamp_adaptive.find_smooth_cut_point",
        smooth_point,
    )
    assert find_timestamp_adaptive_cut_bounds(audio, metadata) == (1000, 2000)
    assert intervals == [(1000, 1001), (2000, 2001)]
    assert metadata["cut_strategy"] == "timestamp-anchor"


def test_timestamp_adaptive_accepts_terminal_target_at_audio_end(monkeypatch) -> None:
    audio = np.ones(3000, dtype=np.float32)
    metadata = _metadata(left=True, right=False)
    metadata["target_end_ts"] = 3000 / 24000
    monkeypatch.setattr(
        "pykokoro.short_sentence_cutters.timestamp_adaptive.find_energy_valley_cut_bounds",
        lambda _audio, _metadata: None,
    )
    monkeypatch.setattr(
        "pykokoro.short_sentence_cutters.timestamp_adaptive.find_smooth_cut_point",
        lambda _audio, *, anchor, **kwargs: anchor,
    )
    assert find_timestamp_adaptive_cut_bounds(audio, metadata) == (1600, 3000)


def test_timestamp_adaptive_accepts_target_at_audio_start(monkeypatch) -> None:
    audio = np.ones(3000, dtype=np.float32)
    metadata = _metadata(left=False, right=True)
    metadata["target_start_ts"] = 0.0
    metadata["target_end_ts"] = 800 / 24000
    monkeypatch.setattr(
        "pykokoro.short_sentence_cutters.timestamp_adaptive.find_smooth_cut_point",
        lambda _audio, *, anchor, **kwargs: anchor,
    )
    assert find_timestamp_adaptive_cut_bounds(audio, metadata) == (0, 800)


def test_timestamp_adaptive_accepts_one_sample_context_gaps(monkeypatch) -> None:
    audio = np.ones(3000, dtype=np.float32)
    metadata = _metadata()
    metadata.update(
        {
            "previous_token_end_ts": 999 / 24000,
            "target_start_ts": 1000 / 24000,
            "target_end_ts": 2000 / 24000,
            "next_token_start_ts": 2001 / 24000,
        }
    )
    monkeypatch.setattr(
        "pykokoro.short_sentence_cutters.timestamp_adaptive.find_smooth_cut_point",
        lambda _audio, *, anchor, **kwargs: anchor,
    )
    assert find_timestamp_adaptive_cut_bounds(audio, metadata) == (1000, 2000)
