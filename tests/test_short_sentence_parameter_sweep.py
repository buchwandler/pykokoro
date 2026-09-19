from __future__ import annotations

import pytest

pytest.importorskip("pykokoro.stages.doc_parsers.plain")


import subprocess
import sys
from pathlib import Path

import numpy as np

from benchmarks.short_sentence_parameter_sweep import (
    RenderedAudio,
    assemble_sweep_audio,
    build_short_sentence_config,
    format_parameter_value,
    parse_parameter_values,
    run_parameter_sweep,
    validate_parameter,
)


def test_parameter_values_keep_cli_order_and_use_stable_formatting() -> None:
    assert parse_parameter_values("energy-threshold", "0.05,0.02,0.10") == [0.05, 0.02, 0.10]
    assert format_parameter_value(0.02) == "0.02"
    assert format_parameter_value(5.0) == "5"


def test_parameter_validation_rejects_unused_and_incompatible_parameters() -> None:
    with pytest.raises(ValueError, match="not used"):
        parse_parameter_values("silence-threshold", "0.01")
    with pytest.raises(ValueError, match="incompatible"):
        validate_parameter("search-radius-ms", "energy-valley", [20.0])


def test_candidate_config_pins_phrase_and_varies_only_requested_field() -> None:
    config = build_short_sentence_config(
        "timestamp-adaptive",
        "Fixed {segment}",
        "energy-threshold",
        0.02,
    )
    mode = config.resolve_modes["phrase"]
    assert mode.neutral_phrase == "Fixed {segment}"
    assert mode.end_phrase == "Fixed {segment}"
    assert mode.energy_threshold == 0.02
    assert mode.cutter == "timestamp-adaptive"
    assert config.phrase_fallback_tries == 0


def test_assemble_sweep_audio_returns_ordered_candidate_offsets() -> None:
    audio, offsets = assemble_sweep_audio(
        np.ones(1, dtype=np.float32),
        [
            (np.ones(2, dtype=np.float32), np.ones(3, dtype=np.float32)),
            (np.ones(4, dtype=np.float32), np.ones(5, dtype=np.float32)),
        ],
        10,
    )

    assert offsets == [(17, 20), (33, 38)]
    assert len(audio) == 45
    assert audio.dtype == np.float32


def test_run_sweep_disables_announcements_and_keeps_candidates_deterministic(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, bool, str | None, float | None]] = []

    def render_text(text: str, config) -> RenderedAudio:
        if config.enabled:
            mode = config.resolve_modes["phrase"]
            calls.append((text, config.enabled, mode.neutral_phrase, mode.energy_threshold))
            return RenderedAudio(
                np.full(3, mode.energy_threshold, dtype=np.float32),
                10,
                {"kind": "phrase", "cut_strategy": mode.cutter},
            )
        calls.append((text, config.enabled, None, None))
        return RenderedAudio(np.ones(2, dtype=np.float32), 10, {})

    _, manifest = run_parameter_sweep(
        text="Why?",
        cutter="timestamp-adaptive",
        parameter="energy-threshold",
        values=[0.02, 0.05],
        phrase_template="Fixed {segment}",
        voice="af_sarah",
        language="en-us",
        model_source="github",
        model_variant="v1.0",
        render_text=render_text,
        save_individual=False,
        individual_dir=tmp_path,
    )

    assert [call[0] for call in calls] == [
        "Short sentence parameter sweep.",
        "Energy threshold is 0.02.",
        "Why?",
        "Energy threshold is 0.05.",
        "Why?",
    ]
    assert [call[1] for call in calls] == [False, False, True, False, True]
    assert [call[3] for call in calls if call[1]] == [0.02, 0.05]
    assert [candidate["value"] for candidate in manifest["candidates"]] == [0.02, 0.05]
    assert manifest["candidates"][0]["start_seconds"] < manifest["candidates"][1]["start_seconds"]
    assert manifest["phrase_fallback_tries"] == 0
    assert manifest["short_sentence_config"]["announcement_enabled"] is False


def test_parameter_sweep_dry_run_does_not_load_models() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/short_sentence_parameter_sweep.py",
            "--text",
            "Why?",
            "--parameter",
            "energy-threshold",
            "--values",
            "0.02,0.05,0.08",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Text: Why?" in result.stdout
    assert "Retries: 0" in result.stdout
    assert "3  energy-threshold=0.08" in result.stdout


def test_parameter_sweep_rejects_pre_cutter_timing_failure_by_default() -> None:
    calls: list[str] = []

    def render_text(text: str, config) -> RenderedAudio:
        _ = config
        calls.append(text)
        if text == "Why?":
            return RenderedAudio(
                np.ones(2, dtype=np.float32),
                10,
                {
                    "failure_stage": "timing-alignment",
                    "timing_failure_reason": "timing-model-position-mismatch",
                    "cutter_reached": False,
                },
            )
        return RenderedAudio(np.ones(2, dtype=np.float32), 10, {})

    with pytest.raises(ValueError, match="not affect the observed failure"):
        run_parameter_sweep(
            text="Why?",
            cutter="timestamp-adaptive",
            parameter="search-radius-ms",
            values=[15.0, 25.0],
            phrase_template="Fixed {segment}",
            voice="af_sarah",
            language="en-us",
            model_source="github",
            model_variant="v1.0",
            render_text=render_text,
        )
    assert calls == ["Short sentence parameter sweep.", "Search radius ms is 15.", "Why?"]
