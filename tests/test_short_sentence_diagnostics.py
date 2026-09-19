from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmarks.short_sentence_diagnostics import (
    SCHEMA,
    ShortSentenceDiagnosticCase,
    _parameter_is_relevant,
    build_diagnostic_cases,
    compute_geometry_rows,
    format_case_report,
    main,
    run_phrase_probe,
)


def test_dry_run_does_not_construct_runtime(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "benchmarks.short_sentence_diagnostics._runtime",
        lambda _args: (_ for _ in ()).throw(AssertionError("runtime loaded")),
    )
    assert main(["--dry-run", "--matrix", "alignment"]) == 0
    output = capsys.readouterr().out
    assert "text: 'who'" in output
    assert "CASE 001 axis=baseline" in output
    assert "axis=phrase-template" in output
    assert "axis=phrase-selection" in output
    assert "axis=speed" in output
    assert "silence_threshold" in output
    assert "[inactive]" in output
    assert "wav_output: disabled" in output
    assert "soundfile" not in output


def test_matrix_is_deterministic_and_covers_axes() -> None:
    first = build_diagnostic_cases(matrix="all")
    second = build_diagnostic_cases(matrix="all")
    assert first == second
    assert [case.axis for case in first[:4]] == [
        "baseline",
        "phrase-template",
        "phrase-template",
        "phrase-template",
    ]
    assert {case.axis for case in first} == {
        "baseline",
        "phrase-template",
        "phrase-selection",
        "speed",
        "cutter",
        "frame-duration-ms",
        "energy-threshold",
        "min-silence-seconds",
        "search-radius-ms",
        "context-guard-ms",
        "analysis-window-ms",
    }
    assert len(first) == 54


def test_catalog_mode_uses_localized_templates_for_terminal_form() -> None:
    cases = build_diagnostic_cases(
        catalog=True,
        language="es",
        terminal_form="fragment",
    )

    assert len(cases) == 2
    assert all(case.axis == "phrase-template" for case in cases)
    assert all(case.settings.phrase_language == "es" for case in cases)
    assert [case.settings.neutral_phrase for case in cases] == [
        "La nota decía: {segment}",
        "El mensaje corto decía: {segment}",
    ]
    assert all(
        case.settings.neutral_phrase != "The short message read: {segment}" for case in cases
    )


def test_geometry_rows_keep_unresolved_counts_visible() -> None:
    rows = compute_geometry_rows(
        [
            {"text": "The", "model_token_count": 3, "model_span_token_count": 4},
            {"text": "who", "model_token_count": None, "model_span_token_count": None},
        ]
    )
    assert rows[0]["cumulative_span_end"] == 4
    assert rows[1]["model_token_count"] is None
    assert rows[1]["cumulative_start"] is None


def test_stage_relevance_marks_cutter_parameter_irrelevant_before_alignment() -> None:
    assert not _parameter_is_relevant("cut-search", {"failure_stage": "timing-alignment"})
    assert _parameter_is_relevant("context-g2p", {"failure_stage": "timing-alignment"})


class _Tokenizer:
    def tokenize(self, phonemes: str) -> list[int]:
        return list(range(len(phonemes)))


class _AudioGenerator:
    def _run_onnx(self, _phonemes, _style, _speed, *, trace, tokens):
        trace.inference.append({"runtime_ms": 1.25, "cache_hit": False})
        return np.zeros(50000, dtype=np.float32), np.ones(len(tokens) + 2, dtype=np.float32)


class _Backend:
    tokenizer = _Tokenizer()
    _audio_generator = _AudioGenerator()

    def resolve_voice_style(self, _voice):
        return np.zeros((16, 256), dtype=np.float32)


def _segment() -> SimpleNamespace:
    return SimpleNamespace(
        text="who",
        phonemes="who",
        tokens=[1, 2, 3],
        lang="en-us",
        char_start=0,
        char_end=3,
        voice_name="af_sarah",
    )


def _context_result(_text: str, _language: str) -> SimpleNamespace:
    texts = ("The", "short", "message", "read", ":", "who")
    counts = (3, 4, 5, 3, 0, 7)
    starts = (0, 4, 10, 18, 22, 24)
    tokens = [
        SimpleNamespace(
            text=text,
            phonemes="x" * count,
            whitespace=" " if index < 5 else "",
            char_start=start,
            char_end=start + len(text),
            model_token_count=count,
            model_span_token_count=count,
        )
        for index, (text, count, start) in enumerate(zip(texts, counts, starts, strict=True))
    ]
    return SimpleNamespace(phonemes="x" * sum(counts), ids=list(range(sum(counts))), tokens=tokens)


def test_successful_probe_reports_geometry_timestamps_and_cut(monkeypatch) -> None:
    def cut(audio, metadata):
        metadata.update(
            {
                "cutter_invoked": True,
                "cutter_reached": True,
                "cut_left": 100,
                "cut_right": 200,
                "cut_strategy": "timestamp-smooth",
            }
        )
        return audio[100:200]

    monkeypatch.setattr("pykokoro.short_sentence_handler.cut_short_sentence_phrase_audio", cut)
    result = run_phrase_probe(
        build_diagnostic_cases(matrix="alignment")[0],
        segment=_segment(),
        backend=_Backend(),
        context_phonemizer=_context_result,
    )
    assert result.succeeded
    assert result.metadata["timing_alignment_complete"] is True
    assert result.metadata["join_complete"] is True
    assert result.invariants["C_phrase_span_sum"]
    assert result.invariants["E_timestamp_cursor"]
    assert result.invariants["I_cut_bounds"]
    assert result.metadata["parameter_relevant"] is True
    assert "TIMING TOKENS" in format_case_report(result)


def test_probe_distinguishes_unresolved_geometry(monkeypatch) -> None:
    def unresolved(_text: str, _language: str) -> SimpleNamespace:
        result = _context_result(_text, _language)
        for token in result.tokens:
            token.model_token_count = None
            token.model_span_token_count = None
        return result

    result = run_phrase_probe(
        build_diagnostic_cases(matrix="alignment")[0],
        segment=_segment(),
        backend=_Backend(),
        context_phonemizer=unresolved,
    )
    assert result.metadata["timing_model_position_count"] == 0
    assert result.metadata["timing_failure_detail"] == "unresolved-model-span"
    assert result.failure_stage == "timing-alignment"
    report = format_case_report(result)
    assert "model_count span_count" in report
    assert "None None" in report


def test_probe_duration_mismatch_is_an_alignment_failure(monkeypatch) -> None:
    class WrongDuration(_AudioGenerator):
        def _run_onnx(self, _phonemes, _style, _speed, *, trace, tokens):
            trace.inference.append({"runtime_ms": 1.0, "cache_hit": True})
            return np.zeros(1000, dtype=np.float32), np.ones(len(tokens) + 1, dtype=np.float32)

    backend = _Backend()
    backend._audio_generator = WrongDuration()
    result = run_phrase_probe(
        build_diagnostic_cases(matrix="alignment")[0],
        segment=_segment(),
        backend=backend,
        context_phonemizer=_context_result,
    )
    assert result.failure_stage == "timing-alignment"
    assert result.metadata["timing_failure_detail"] == "duration-position-count-mismatch"
    assert result.metadata["join_attempted"] is False


def test_case_json_is_versioned_and_contains_human_report_fields() -> None:
    result = ShortSentenceDiagnosticCase("001", "baseline", "default", {"silence_threshold": 1e-4})
    payload = result.to_dict()
    assert payload["schema"] == SCHEMA
    assert "settings" in payload
    assert "metadata" in payload
    assert Path("benchmarks/short_sentence_parameter_sweep.py").exists()
    assert Path("benchmarks/short_sentence_latency.py").exists()
