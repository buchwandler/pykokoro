from __future__ import annotations

import pytest

pytest.importorskip("pykokoro.stages.doc_parsers.plain")


from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmarks import voice_loudness as benchmark
from pykokoro.discovery import ModelCapabilities, VoiceCapabilities


def test_count_stimulus_uses_independent_digit_free_cardinals() -> None:
    for locale in ("en-US", "en-GB", "de", "zh", "ar", "kk"):
        stimulus = benchmark.resolve_count_stimulus(locale)
        assert stimulus.generator == "spokenform"
        assert len(stimulus.spoken_text.split(", ")) == 10
        assert not any(character.isdigit() for character in stimulus.spoken_text)
        assert stimulus.fallback_used is False


def test_locale_aliases_are_resolved_by_spokenform() -> None:
    assert benchmark.resolve_count_stimulus("en-US").normalized_language == "en_US"
    assert benchmark.resolve_count_stimulus("en-GB").normalized_language == "en_GB"
    assert benchmark.resolve_count_stimulus("pt-PT").normalized_language == "pt_PT"


def test_unsupported_cardinals_require_reviewed_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        benchmark, "count_words", lambda _locale: (_ for _ in ()).throw(ValueError("no"))
    )
    stimulus = benchmark.resolve_count_stimulus("xx", {"xx": {"text": "uno, dos, tres."}})
    assert stimulus.generator == "fallback"
    assert stimulus.fallback_used is True

    with pytest.raises(benchmark.StimulusResolutionError, match="no reviewed count fallback"):
        benchmark.resolve_count_stimulus("yy", {})


def _model(*, experimental: bool = False) -> ModelCapabilities:
    detail = VoiceCapabilities("voice", "female", "en", "en-US", "English")
    return ModelCapabilities(
        model_id="model",
        source="github",
        languages=("en-US",),
        voices=("voice",),
        default_voice="voice",
        qualities=("fp32",),
        g2p_backend=None,
        lexicons=None,
        frontend="kokoro",
        status="experimental" if experimental else "ready",
        experimental=experimental,
        runtime_available=True,
        redistribution_allowed=True,
        sample_rate=24000,
        voice_details=(detail,),
    )


def test_entries_include_all_discovered_locales_and_experimental_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        benchmark, "_selected_models", lambda _args: [(_model(experimental=True), "fp32")]
    )
    entries = benchmark._entries(SimpleNamespace())
    assert len(entries) == 1
    assert entries[0]["locale"] == "en-US"
    assert entries[0]["language"] == "en"
    assert entries[0]["experimental"] is True


def test_measurement_repeats_and_propagates_experimental_frontend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class Pipeline:
        def run(self, text: str, **kwargs: object) -> SimpleNamespace:
            calls.append({"text": text, **kwargs})
            return SimpleNamespace(audio=np.ones(100, dtype=np.float32), sample_rate=24000)

    monkeypatch.setattr(
        benchmark,
        "measure_loudness",
        lambda _audio, sample_rate: SimpleNamespace(
            integrated_lufs=-24.0, sample_peak_dbfs=-1.0, true_peak_dbtp=-1.5
        ),
    )
    policy = benchmark.BenchmarkPolicy(1, "test", 1, 10, 3, -24.0, -1.0, 8.0, 0.75)
    entry = {
        "model_source": "github",
        "model_id": "model",
        "quality": "fp32",
        "voice": "voice",
        "locale": "en-US",
        "gender": "female",
        "experimental": True,
    }
    measurements, failures = benchmark._measure_entry(entry, Pipeline(), policy)
    assert not failures
    assert [item["repeat"] for item in measurements] == [0, 1, 2]
    assert [call["allow_experimental_frontend"] for call in calls] == [True, True, True]
    assert [call["generation"].random_seed for call in calls] == [0, 1, 2]


def _measurement(
    lufs: float, peak: float = -2.0, voice: str = "voice", repeat: int = 0
) -> dict[str, object]:
    return {
        "model_source": "github",
        "model_id": "model",
        "quality": "fp32",
        "voice": voice,
        "locale": "en-US",
        "gender": "female",
        "integrated_lufs": lufs,
        "sample_peak_dbfs": peak,
        "true_peak_dbtp": peak,
        "duration_seconds": 1.0,
        "repeat": repeat,
    }


def test_aggregation_records_median_mad_peaks_and_repeats() -> None:
    rows = benchmark._aggregate(
        [
            _measurement(-27.0, repeat=0),
            _measurement(-26.0, repeat=1),
            _measurement(-25.0, repeat=2),
        ]
    )
    assert rows[0]["median_lufs"] == -26.0
    assert rows[0]["mad_lu"] == 1.0
    assert rows[0]["max_true_peak_dbtp"] == -2.0
    assert rows[0]["repeat_count"] == 3
    assert rows[0]["status"] == "high_variability"


def test_fixed_reference_is_population_independent() -> None:
    first = benchmark._aggregate([_measurement(-27.0, repeat=i) for i in range(3)])[0]
    second_rows = benchmark._aggregate(
        [_measurement(-27.0, repeat=i) for i in range(3)]
        + [_measurement(-18.0, voice="other", repeat=i) for i in range(3)]
    )
    second = next(row for row in second_rows if row["voice"] == "voice")
    assert benchmark._calibration_candidate(first)["requested_gain_db"] == 3.0
    assert (
        benchmark._calibration_candidate(first)["requested_gain_db"]
        == benchmark._calibration_candidate(second)["requested_gain_db"]
    )


def test_headroom_limiting_attenuates_known_peak_violation() -> None:
    row = benchmark._calibration_candidate(
        {"status": "eligible", "median_lufs": -27.0, "max_true_peak_dbtp": -2.0}
    )
    assert row["requested_gain_db"] == 3.0
    assert row["gain_db"] == 1.0
    assert row["headroom_limited"] is True
    assert row["target_reached"] is False

    row = benchmark._calibration_candidate(
        {"status": "eligible", "median_lufs": -27.0, "max_true_peak_dbtp": 1.0}
    )
    assert row["gain_db"] == -2.0


def test_gain_review_separates_large_attenuation_from_boost() -> None:
    attenuation = benchmark._calibration_candidate(
        {"status": "eligible", "median_lufs": -37.0, "max_true_peak_dbtp": 0.0},
        max_boost_db=8.0,
        max_attenuation_db=12.0,
    )
    assert attenuation["requested_gain_db"] == 13.0
    assert attenuation["status"] in {"review_large_boost", "headroom_limited"}
    attenuation = benchmark._calibration_candidate(
        {"status": "eligible", "median_lufs": -12.5, "max_true_peak_dbtp": 0.0},
        max_boost_db=8.0,
        max_attenuation_db=12.0,
    )
    assert attenuation["requested_gain_db"] == -11.5
    assert attenuation["status"] == "eligible"
    large_attenuation = benchmark._calibration_candidate(
        {"status": "eligible", "median_lufs": -11.0, "max_true_peak_dbtp": 0.0},
        max_boost_db=8.0,
        max_attenuation_db=10.0,
    )
    assert large_attenuation["status"] == "review_large_attenuation"
    assert large_attenuation["gain_db"] == -13.0


def test_candidate_writer_protects_production_catalog(tmp_path: Path) -> None:
    policy = benchmark.BenchmarkPolicy(1, "test", 1, 10, 3, -24.0, -1.0, 8.0, 0.75)
    with pytest.raises(ValueError, match="refusing"):
        benchmark._write_calibration_candidate(
            benchmark.PRODUCTION_CALIBRATION_PATH, policy, [], {}
        )


def test_policy_and_fallback_files_are_versioned() -> None:
    assert benchmark._load_policy().name == "pykokoro-count-1-to-10-v2"
    assert "hi" in benchmark._load_fallbacks()


def test_coverage_counts_unique_voice_failures_and_utterances() -> None:
    entry = {
        "model_source": "github",
        "model_id": "model",
        "quality": "fp32",
        "voice": "voice",
    }
    failures = [dict(entry, repeat=index) for index in range(3)]
    policy = benchmark.BenchmarkPolicy(1, "test", 1, 10, 3, -24.0, -1.0, 8.0, 0.75)
    coverage = benchmark._coverage([entry], [], failures, [], policy)
    assert coverage["voices_attempted"] == 1
    assert coverage["voices_succeeded"] == 0
    assert coverage["voices_failed"] == 1
    assert coverage["utterances_failed"] == 3


def test_summary_uses_identity_counts_not_failure_events(tmp_path: Path) -> None:
    policy = benchmark.BenchmarkPolicy(1, "test", 1, 10, 3, -24.0, -1.0, 8.0, 0.75)
    aggregate = benchmark._aggregate([_measurement(-24.0, repeat=index) for index in range(3)])[0]
    output = tmp_path / "benchmark"
    benchmark._write_outputs(
        output,
        [],
        [{} for _ in range(3)],
        [aggregate],
        policy,
        coverage={
            "voices_attempted": 2,
            "voices_succeeded": 1,
            "voices_eligible": 1,
            "voices_review_required": 0,
            "voices_failed": 1,
            "utterances_succeeded": 3,
            "utterances_failed": 3,
        },
    )
    summary = (output / "summary.md").read_text(encoding="utf-8")
    assert "Complete measured voices: 1" in summary
    assert "Automatically eligible voices: 1" in summary
    assert "Unmeasured/failed voices: 1" in summary
    assert "Failed utterance attempts: 3" in summary
    assert "Failed/incomplete voices:" not in summary
