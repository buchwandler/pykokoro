#!/usr/bin/env python3
"""Benchmark PyKokoro voice loudness with a versioned count-to-ten stimulus."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from audiosig import measure_loudness  # noqa: E402
from spokenform import normalize_language, normalize_numbers  # noqa: E402

from pykokoro import LoudnessConfig, PipelineConfig  # noqa: E402
from pykokoro.discovery import ModelCapabilities, discover_models  # noqa: E402
from pykokoro.generation_config import GenerationConfig  # noqa: E402
from pykokoro.voice_level import VoiceCalibrationKey  # noqa: E402

POLICY_PATH = Path(__file__).with_name("data") / "voice_loudness_policy.json"
FALLBACKS_PATH = Path(__file__).with_name("data") / "voice_loudness_count_fallbacks.json"
DEFAULT_OUTPUT = Path("artifacts/voice_loudness")
COUNT_SOURCE = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10."
COUNT_VALUES = tuple(range(1, 11))
PRODUCTION_CALIBRATION_PATH = ROOT / "pykokoro" / "data" / "voice_level_calibration.json"


class StimulusResolutionError(ValueError):
    """Raised when a locale has neither safe Spokenform output nor a reviewed fallback."""


@dataclass(frozen=True, slots=True)
class LoudnessStimulus:
    locale: str
    normalized_language: str
    source: str
    spoken_text: str
    generator: str
    fallback_used: bool


@dataclass(frozen=True, slots=True)
class BenchmarkPolicy:
    schema: int
    name: str
    first: int
    last: int
    repeats: int
    reference_lufs: float
    calibration_peak_ceiling_dbtp: float
    max_abs_requested_gain_db: float
    max_mad_lu: float
    max_boost_db: float | None = None
    max_attenuation_db: float | None = None

    @property
    def boost_limit_db(self) -> float:
        return (
            self.max_boost_db if self.max_boost_db is not None else self.max_abs_requested_gain_db
        )

    @property
    def attenuation_limit_db(self) -> float:
        return (
            self.max_attenuation_db
            if self.max_attenuation_db is not None
            else self.max_abs_requested_gain_db
        )


def _load_policy(path: Path = POLICY_PATH) -> BenchmarkPolicy:
    data = json.loads(path.read_text(encoding="utf-8"))
    stimulus = data.get("stimulus", {})
    schema = data.get("schema")
    if schema not in {1, 2} or not isinstance(data.get("name"), str):
        raise ValueError("voice loudness policy must use schema 1 or 2 and have a name")
    if schema == 1:
        max_abs = float(data["max_abs_requested_gain_db"])
        max_boost = None
        max_attenuation = None
    else:
        max_boost = float(data["max_boost_db"])
        max_attenuation = float(data["max_attenuation_db"])
        max_abs = max(max_boost, max_attenuation)
    policy = BenchmarkPolicy(
        schema=schema,
        name=data["name"],
        first=int(stimulus.get("first", 0)),
        last=int(stimulus.get("last", 0)),
        repeats=int(data.get("repeats", 0)),
        reference_lufs=float(data["reference_lufs"]),
        calibration_peak_ceiling_dbtp=float(data["calibration_peak_ceiling_dbtp"]),
        max_abs_requested_gain_db=max_abs,
        max_mad_lu=float(data["max_mad_lu"]),
        max_boost_db=max_boost,
        max_attenuation_db=max_attenuation,
    )
    if (policy.first, policy.last) != (1, 10) or policy.repeats < 1:
        raise ValueError("voice loudness policy must define the count stimulus from 1 through 10")
    return policy


def _policy_dict(policy: BenchmarkPolicy) -> dict[str, Any]:
    return {
        "schema": 2,
        "name": policy.name,
        "stimulus": {"kind": "count", "first": policy.first, "last": policy.last},
        "repeats": policy.repeats,
        "reference_lufs": policy.reference_lufs,
        "calibration_peak_ceiling_dbtp": policy.calibration_peak_ceiling_dbtp,
        "max_boost_db": policy.boost_limit_db,
        "max_attenuation_db": policy.attenuation_limit_db,
        "max_mad_lu": policy.max_mad_lu,
    }


def _load_fallbacks(path: Path = FALLBACKS_PATH) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != 1 or data.get("stimulus") != "count-1-to-10":
        raise ValueError("count fallback data has an invalid schema or stimulus")
    locales = data.get("locales")
    if not isinstance(locales, dict):
        raise ValueError("count fallback locales must be an object")
    return locales


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _generated_with() -> dict[str, str]:
    return {
        "pykokoro": _package_version("pykokoro"),
        "audiosig": _package_version("audiosig"),
        "spokenform": _package_version("spokenform"),
    }


def count_words(locale: str) -> tuple[str, ...]:
    """Render each cardinal independently through Spokenform."""
    language = normalize_language(locale)
    words = tuple(normalize_numbers(str(value), language=language) for value in COUNT_VALUES)
    if len(words) != len(COUNT_VALUES) or any(
        not word.strip() or any(character.isdigit() for character in word) for word in words
    ):
        raise StimulusResolutionError(
            f"Spokenform did not produce ten digit-free cardinal words for {locale!r}"
        )
    return words


def _fallback_for_locale(
    locale: str, normalized_language: str, fallbacks: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    base_language = normalized_language.split("_", 1)[0]
    for key in (locale, normalized_language, base_language):
        fallback = fallbacks.get(key)
        if fallback is not None:
            return fallback
    return None


def resolve_count_stimulus(
    locale: str, fallbacks: dict[str, dict[str, Any]] | None = None
) -> LoudnessStimulus:
    """Resolve one cached-by-caller count stimulus without silently using digits."""
    fallbacks = _load_fallbacks() if fallbacks is None else fallbacks
    try:
        normalized_language = normalize_language(locale)
        words = count_words(locale)
    except Exception as spokenform_error:
        try:
            normalized_language = normalize_language(locale)
        except Exception:
            normalized_language = locale
        fallback = _fallback_for_locale(locale, normalized_language, fallbacks)
        if fallback is None:
            raise StimulusResolutionError(
                f"no reviewed count fallback for unsupported locale {locale!r}"
            ) from spokenform_error
        spoken_text = fallback.get("text")
        if not isinstance(spoken_text, str) or not spoken_text.strip():
            raise StimulusResolutionError(f"fallback for locale {locale!r} has no text") from None
        if any(character.isdigit() for character in spoken_text):
            raise StimulusResolutionError(
                f"fallback for locale {locale!r} contains digits"
            ) from None
        return LoudnessStimulus(
            locale=locale,
            normalized_language=normalized_language,
            source=COUNT_SOURCE,
            spoken_text=spoken_text,
            generator="fallback",
            fallback_used=True,
        )
    return LoudnessStimulus(
        locale=locale,
        normalized_language=normalized_language,
        source=COUNT_SOURCE,
        spoken_text=", ".join(words) + ".",
        generator="spokenform",
        fallback_used=False,
    )


def _quality(model: ModelCapabilities, requested: str | None) -> str:
    if requested is not None:
        if requested not in model.qualities:
            raise ValueError(f"quality {requested!r} is unavailable for model {model.model_id!r}")
        return requested
    if "fp32" in model.qualities:
        return "fp32"
    if not model.qualities:
        raise ValueError(f"model {model.model_id!r} has no runnable quality")
    return model.qualities[0]


def _rms_dbfs(audio: np.ndarray) -> float:
    rms = (
        float(np.sqrt(np.mean(np.square(np.asarray(audio, dtype=np.float64)))))
        if audio.size
        else 0.0
    )
    return -math.inf if rms == 0.0 else 20.0 * math.log10(rms)


def _selected_models(args: argparse.Namespace) -> list[tuple[ModelCapabilities, str]]:
    discovery = discover_models(
        offline=getattr(args, "offline", False), preference=getattr(args, "preference", "auto")
    )
    selected: list[tuple[ModelCapabilities, str]] = []
    for model in discovery.models:
        if getattr(args, "model_source", None) and model.source != args.model_source:
            continue
        if getattr(args, "model", None) and model.model_id != args.model:
            continue
        if model.status not in {"ready", "experimental"} or not model.runtime_available:
            continue
        selected.append((model, _quality(model, getattr(args, "quality", None))))
    return selected


def _entries(
    args: argparse.Namespace, _policy: BenchmarkPolicy | None = None
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for model, quality in _selected_models(args):
        details = {detail.name: detail for detail in model.voice_details}
        for voice in model.voices:
            detail = details.get(voice)
            locale = (
                detail.locale
                if detail is not None
                else (model.languages[0] if model.languages else "")
            )
            language = detail.language if detail is not None else locale
            if not locale or not language:
                continue
            if getattr(args, "voice", None) and voice != args.voice:
                continue
            if getattr(args, "locale", None) and locale != args.locale:
                continue
            result.append(
                {
                    "model_source": model.source,
                    "model_id": model.model_id,
                    "quality": quality,
                    "voice": voice,
                    "locale": locale,
                    "language": language,
                    "gender": detail.gender if detail is not None else "unknown",
                    "experimental": model.experimental,
                    "status": model.status,
                    "sample_rate": model.sample_rate or 24000,
                }
            )
    max_voices = getattr(args, "max_voices", None)
    return result[:max_voices] if max_voices else result


def _stimulus_record(stimulus: LoudnessStimulus) -> dict[str, Any]:
    return asdict(stimulus)


def _measure_entry(
    entry: dict[str, Any],
    pipeline: Any,
    policy: BenchmarkPolicy | str | None = None,
    stimulus: LoudnessStimulus | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(policy, str):
        policy = BenchmarkPolicy(1, policy, 1, 10, 1, -24.0, -1.0, 8.0, 0.75)
    policy = policy or _load_policy()
    stimulus = stimulus or resolve_count_stimulus(entry["locale"])
    measurements: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for repeat in range(policy.repeats):
        try:
            result = pipeline.run(
                stimulus.spoken_text,
                model_source=entry["model_source"],
                model_variant=entry["model_id"],
                model_quality=entry["quality"],
                voice=entry["voice"],
                lang=entry["locale"],
                generation=GenerationConfig(lang=entry["locale"], speed=1.0, random_seed=repeat),
                allow_experimental_frontend=entry["experimental"],
                loudness=LoudnessConfig(),
            )
            audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
            metrics = measure_loudness(audio, sample_rate=result.sample_rate)
            measurements.append(
                {
                    "model_source": entry["model_source"],
                    "model_id": entry["model_id"],
                    "quality": entry["quality"],
                    "voice": entry["voice"],
                    "locale": entry["locale"],
                    "gender": entry["gender"],
                    "corpus": policy.name,
                    "stimulus_id": "count-1-to-10",
                    "repeat": repeat,
                    "random_seed": repeat,
                    "duration_seconds": len(audio) / result.sample_rate,
                    "sample_rate": result.sample_rate,
                    "integrated_lufs": metrics.integrated_lufs,
                    "sample_peak_dbfs": metrics.sample_peak_dbfs,
                    "true_peak_dbtp": metrics.true_peak_dbtp,
                    "rms_dbfs": _rms_dbfs(audio),
                    "pykokoro_version": _generated_with()["pykokoro"],
                    "audiosig_version": _generated_with()["audiosig"],
                    "spokenform_version": _generated_with()["spokenform"],
                }
            )
        except Exception as exc:
            failures.append(
                {
                    **{
                        key: entry[key]
                        for key in ("model_source", "model_id", "quality", "voice", "locale")
                    },
                    "stimulus_id": "count-1-to-10",
                    "repeat": repeat,
                    "random_seed": repeat,
                    "error": str(exc),
                }
            )
    return measurements, failures


def _aggregate(
    measurements: list[dict[str, Any]],
    repeats: int = 3,
    max_mad_lu: float = 0.75,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for item in measurements:
        key = (item["model_source"], item["model_id"], item["quality"], item["voice"])
        grouped.setdefault(key, []).append(item)
    rows: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        loudness = [float(item["integrated_lufs"]) for item in items]
        finite = [value for value in loudness if math.isfinite(value)]
        if finite:
            median_lufs = statistics.median(finite)
            mad_lu = statistics.median(abs(value - median_lufs) for value in finite)
            status = "eligible"
            if len(finite) != len(loudness):
                status = "non_finite_loudness"
            elif len(finite) != repeats:
                status = "insufficient_repeats"
            elif mad_lu > max_mad_lu:
                status = "high_variability"
            min_lufs = min(finite)
            max_lufs = max(finite)
        else:
            median_lufs = math.nan
            mad_lu = math.nan
            min_lufs = math.nan
            max_lufs = math.nan
            status = "non_finite_loudness"
        true_peaks = [float(item["true_peak_dbtp"]) for item in items]
        sample_peaks = [float(item["sample_peak_dbfs"]) for item in items]
        durations = [float(item["duration_seconds"]) for item in items]
        rows.append(
            {
                "model_source": key[0],
                "model_id": key[1],
                "quality": key[2],
                "voice": key[3],
                "locale": items[0]["locale"],
                "gender": items[0]["gender"],
                "samples": len(items),
                "repeat_count": len(items),
                "median_lufs": median_lufs,
                "mad_lu": mad_lu,
                "min_lufs": min_lufs,
                "max_lufs": max_lufs,
                "median_duration_seconds": statistics.median(durations),
                "max_sample_peak_dbfs": max(sample_peaks),
                "max_true_peak_dbtp": max(true_peaks),
                "status": status,
            }
        )
    return rows


def _calibration_candidate(
    aggregate: dict[str, Any],
    reference_lufs: float = -24.0,
    peak_ceiling_dbtp: float = -1.0,
    max_abs_requested_gain_db: float | None = None,
    max_boost_db: float = 8.0,
    max_attenuation_db: float = 12.0,
) -> dict[str, Any]:
    if max_abs_requested_gain_db is not None:
        max_boost_db = max_abs_requested_gain_db
        max_attenuation_db = max_abs_requested_gain_db
    measured_lufs = float(aggregate["median_lufs"])
    max_true_peak = float(aggregate["max_true_peak_dbtp"])
    requested_gain = reference_lufs - measured_lufs
    max_safe_gain = peak_ceiling_dbtp - max_true_peak
    gain = min(requested_gain, max_safe_gain) if requested_gain > 0 else requested_gain
    headroom_limited = gain < requested_gain
    status = aggregate["status"]
    if status == "eligible":
        if headroom_limited:
            status = "headroom_limited"
        elif requested_gain > max_boost_db:
            status = "review_large_boost"
        elif requested_gain < -max_attenuation_db:
            status = "review_large_attenuation"
    return {
        **aggregate,
        "measured_lufs": measured_lufs,
        "reference_lufs": reference_lufs,
        "requested_gain_db": requested_gain,
        "gain_db": gain,
        "max_safe_gain_db": max_safe_gain,
        "projected_true_peak_dbtp": max_true_peak + gain,
        "headroom_limited": headroom_limited,
        "target_reached": not headroom_limited,
        "status": status,
    }


def _candidate_rows(
    aggregates: list[dict[str, Any]], policy: BenchmarkPolicy
) -> list[dict[str, Any]]:
    return [
        _calibration_candidate(
            row,
            reference_lufs=policy.reference_lufs,
            peak_ceiling_dbtp=policy.calibration_peak_ceiling_dbtp,
            max_boost_db=policy.boost_limit_db,
            max_attenuation_db=policy.attenuation_limit_db,
        )
        for row in aggregates
    ]


def _coverage(
    entries: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
    policy: BenchmarkPolicy,
) -> dict[str, int | str]:
    attempted_keys = {
        (entry["model_source"], entry["model_id"], entry["quality"], entry["voice"])
        for entry in entries
    }
    complete_keys = {
        (row["model_source"], row["model_id"], row["quality"], row["voice"])
        for row in aggregates
        if row["repeat_count"] == policy.repeats
    }
    complete_candidates = [
        row for row in _candidate_rows(aggregates, policy) if row["repeat_count"] == policy.repeats
    ]
    eligible_keys = {
        (row["model_source"], row["model_id"], row["quality"], row["voice"])
        for row in complete_candidates
        if row["status"] == "eligible"
    }
    return {
        "models_discovered": len({(entry["model_source"], entry["model_id"]) for entry in entries}),
        "voices_discovered": len(attempted_keys),
        "voices_attempted": len(attempted_keys),
        "voices_succeeded": len(complete_keys),
        "voices_eligible": len(eligible_keys),
        "voices_review_required": len(complete_keys - eligible_keys),
        "voices_failed": len(attempted_keys - complete_keys),
        "utterances_succeeded": len(measurements),
        "utterances_failed": len(failures),
        "registry_source": "discovery",
    }


def _write_calibration_candidate(
    path: Path,
    policy: BenchmarkPolicy,
    candidates: list[dict[str, Any]],
    coverage: dict[str, int],
) -> None:
    if path.resolve() == PRODUCTION_CALIBRATION_PATH.resolve():
        raise ValueError("refusing to overwrite packaged production calibration data")
    path.parent.mkdir(parents=True, exist_ok=True)
    voices = {
        str(
            VoiceCalibrationKey(row["model_source"], row["model_id"], row["quality"], row["voice"])
        ): {
            "measured_lufs": row["measured_lufs"],
            "reference_lufs": row["reference_lufs"],
            "mad_lu": row["mad_lu"],
            "gain_db": row["gain_db"],
            "samples": row["samples"],
            "method": "bs1770",
            "corpus_version": policy.name,
            "status": row["status"],
            "max_true_peak_dbtp": row["max_true_peak_dbtp"],
            "projected_true_peak_dbtp": row["projected_true_peak_dbtp"],
            "headroom_limited": row["headroom_limited"],
            "target_reached": row["target_reached"],
        }
        for row in candidates
        if row["repeat_count"] == policy.repeats
    }
    path.write_text(
        json.dumps(
            {
                "schema": 2,
                "method": "bs1770",
                "corpus": policy.name,
                "policy": _policy_dict(policy),
                "generated_with": _generated_with(),
                "coverage": coverage,
                "voices": voices,
                "candidates": _json_safe(candidates),
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _write_outputs(
    output: Path,
    measurements: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    policy: BenchmarkPolicy | str,
    calibration_path: Path | None = None,
    *,
    stimuli: dict[str, LoudnessStimulus] | None = None,
    coverage: dict[str, int] | None = None,
) -> None:
    if isinstance(policy, str):
        policy = BenchmarkPolicy(1, policy, 1, 10, 1, -24.0, -1.0, 8.0, 0.75)
    output.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_rows(aggregates, policy)
    payload = {
        "schema": 2,
        "corpus": policy.name,
        "policy": _policy_dict(policy),
        "generated_with": _generated_with(),
        "coverage": coverage or {},
        "stimuli": {locale: _stimulus_record(item) for locale, item in (stimuli or {}).items()},
        "measurements": measurements,
        "failures": failures,
        "aggregates": candidates,
    }
    (output / "measurements.json").write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    fields = (
        list(measurements[0])
        if measurements
        else [
            "model_source",
            "model_id",
            "quality",
            "voice",
            "locale",
            "stimulus_id",
            "repeat",
            "integrated_lufs",
            "sample_peak_dbfs",
            "true_peak_dbtp",
        ]
    )
    with (output / "measurements.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(measurements)
    lines = [
        "# Voice loudness benchmark",
        "",
        "## Coverage",
        "",
        f"- Registry source: {coverage.get('registry_source', 'unknown') if coverage else 'unknown'}",
        f"- Runnable models discovered: {coverage.get('models_discovered', 0) if coverage else 0}",
        f"- Runnable voices discovered: {coverage.get('voices_discovered', 0) if coverage else 0}",
        f"- Voices attempted: {coverage.get('voices_attempted', 0) if coverage else 0}",
        f"- Complete measured voices: {coverage.get('voices_succeeded', 0) if coverage else 0}",
        f"- Automatically eligible voices: {coverage.get('voices_eligible', 0) if coverage else 0}",
        f"- Review-required complete voices: {coverage.get('voices_review_required', 0) if coverage else 0}",
        f"- Unmeasured/failed voices: {coverage.get('voices_failed', 0) if coverage else 0}",
        f"- Failed utterance attempts: {coverage.get('utterances_failed', 0) if coverage else 0}",
        f"- Successful utterance attempts: {coverage.get('utterances_succeeded', 0) if coverage else 0}",
        f"- Distinct locales: {len(stimuli or {})}",
        f"- Fallback-stimulus locales: {sum(item.fallback_used for item in (stimuli or {}).values())}",
        "",
        "## Measurement policy",
        "",
        f"- Corpus ID: `{policy.name}`",
        f"- Count range: {policy.first} through {policy.last}",
        f"- Repeats: {policy.repeats}",
        f"- Reference LUFS: {policy.reference_lufs:.2f}",
        f"- Calibration peak ceiling: {policy.calibration_peak_ceiling_dbtp:.2f} dBTP",
        f"- Audiosig version: {_generated_with()['audiosig']}",
        f"- Spokenform version: {_generated_with()['spokenform']}",
        f"- PyKokoro version: {_generated_with()['pykokoro']}",
        "",
        "## Candidate calibration",
        "",
    ]
    complete = [row for row in candidates if row["repeat_count"] == policy.repeats]
    if complete:
        eligible = [row for row in complete if row["status"] == "eligible"]
        review_required = [row for row in complete if row["status"] != "eligible"]
        lines.extend(
            [
                f"- Maximum requested boost: {max(row['requested_gain_db'] for row in complete):+.2f} dB",
                f"- Maximum requested attenuation: {min(row['requested_gain_db'] for row in complete):+.2f} dB",
                f"- Automatically eligible voices: {len(eligible)}",
                f"- Review-required complete voices: {len(review_required)}",
                f"- Headroom-limited voices: {sum(row['headroom_limited'] for row in complete)}",
                f"- High-variability voices: {sum(row['status'] == 'high_variability' for row in candidates)}",
            ]
        )
    else:
        lines.append("- No complete candidate measurements were recorded.")
    lines.extend(
        [
            "",
            "## Locale coverage table",
            "",
            "| Locale | Voices | Stimulus source | Complete | Failures |",
            "| --- | ---: | --- | ---: | ---: |",
        ]
    )
    by_locale: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        by_locale.setdefault(row["locale"], []).append(row)
    for locale, stimulus in sorted((stimuli or {}).items()):
        rows = by_locale.get(locale, [])
        lines.append(
            f"| {locale} | {len(rows)} | {stimulus.generator} | "
            f"{sum(row['repeat_count'] == policy.repeats for row in rows)} | "
            f"{sum(item.get('locale') == locale for item in failures)} |"
        )
    lines.extend(
        [
            "",
            "## Voice table",
            "",
            "| Key | Locale | Repeats | Median LUFS | MAD | Max true peak | Requested gain | Candidate gain | Projected true peak | Status |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in candidates:
        values = (
            f"{row['model_source']}:{row['model_id']}:{row['quality']}:{row['voice']}",
            row["locale"],
            row["repeat_count"],
            row["median_lufs"],
            row["mad_lu"],
            row["max_true_peak_dbtp"],
            row["requested_gain_db"],
            row["gain_db"],
            row["projected_true_peak_dbtp"],
            row["status"],
        )
        lines.append(
            f"| `{values[0]}` | {values[1]} | {values[2]} | {values[3]:.2f} | {values[4]:.2f} | "
            f"{values[5]:.2f} | {values[6]:+.2f} | {values[7]:+.2f} | {values[8]:.2f} | {values[9]} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if calibration_path is not None:
        _write_calibration_candidate(calibration_path, policy, candidates, coverage or {})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-source", choices=("github", "huggingface"))
    parser.add_argument("--model")
    parser.add_argument("--voice")
    parser.add_argument("--locale")
    parser.add_argument("--quality")
    parser.add_argument("--max-voices", type=int)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--preference", choices=("auto", "github", "huggingface", "upstream"), default="auto"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--reference-lufs", type=float)
    parser.add_argument("--calibration-peak-ceiling-dbtp", type=float)
    parser.add_argument("--list-stimuli", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--write-calibration-candidate", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policy = _load_policy()
    if args.repeats is not None:
        policy = BenchmarkPolicy(
            policy.schema,
            policy.name,
            policy.first,
            policy.last,
            args.repeats,
            policy.reference_lufs,
            policy.calibration_peak_ceiling_dbtp,
            policy.max_abs_requested_gain_db,
            policy.max_mad_lu,
        )
    if args.reference_lufs is not None:
        policy = BenchmarkPolicy(
            policy.schema,
            policy.name,
            policy.first,
            policy.last,
            policy.repeats,
            args.reference_lufs,
            policy.calibration_peak_ceiling_dbtp,
            policy.max_abs_requested_gain_db,
            policy.max_mad_lu,
        )
    if args.calibration_peak_ceiling_dbtp is not None:
        policy = BenchmarkPolicy(
            policy.schema,
            policy.name,
            policy.first,
            policy.last,
            policy.repeats,
            policy.reference_lufs,
            args.calibration_peak_ceiling_dbtp,
            policy.max_abs_requested_gain_db,
            policy.max_mad_lu,
        )
    entries = _entries(args, policy)
    if not entries:
        raise SystemExit("No runnable voices matched the requested filters")
    fallbacks = _load_fallbacks()
    stimuli: dict[str, LoudnessStimulus] = {}
    failures: list[dict[str, Any]] = []
    measurable_entries: list[dict[str, Any]] = []
    for entry in entries:
        if entry["locale"] not in stimuli:
            try:
                stimuli[entry["locale"]] = resolve_count_stimulus(entry["locale"], fallbacks)
            except StimulusResolutionError as exc:
                failures.append(
                    {
                        **{
                            key: entry[key]
                            for key in ("model_source", "model_id", "quality", "voice", "locale")
                        },
                        "error": str(exc),
                        "status": "unsupported_stimulus",
                    }
                )
                continue
        measurable_entries.append(entry)
    if args.list_stimuli:
        for locale, stimulus in sorted(stimuli.items()):
            print(f"{locale}  {stimulus.generator}  {stimulus.spoken_text}")
        return 0 if not failures else 1
    if not measurable_entries:
        _write_outputs(args.output, [], failures, [], policy, None, stimuli=stimuli)
        return 1
    from pykokoro import KokoroPipeline

    first = measurable_entries[0]
    config = PipelineConfig(
        model_source=first["model_source"],
        model_variant=first["model_id"],
        model_quality=first["quality"],
        voice=first["voice"],
        allow_experimental_frontend=first["experimental"],
        generation=GenerationConfig(lang=first["locale"], speed=1.0, random_seed=0),
    )
    measurements: list[dict[str, Any]] = []
    with KokoroPipeline(config) as pipeline:
        for index, entry in enumerate(measurable_entries, 1):
            print(
                f"[{index:03d}/{len(measurable_entries)}] {entry['model_id']} / {entry['voice']} / {entry['locale']}"
            )
            current, current_failures = _measure_entry(
                entry, pipeline, policy, stimuli[entry["locale"]]
            )
            measurements.extend(current)
            failures.extend(current_failures)
    aggregates = _aggregate(measurements, policy.repeats, policy.max_mad_lu)
    coverage = _coverage(entries, aggregates, failures, measurements, policy)
    incomplete = (
        bool(failures)
        or coverage["voices_failed"] > 0
        or any(row["repeat_count"] != policy.repeats for row in aggregates)
    )
    candidate_path = args.write_calibration_candidate if not incomplete else None
    _write_outputs(
        args.output,
        measurements,
        failures,
        aggregates,
        policy,
        candidate_path,
        stimuli=stimuli,
        coverage=coverage,
    )
    if incomplete and not args.allow_failures:
        print(
            f"Benchmark incomplete: {len(failures)} failures; no calibration candidate emitted",
            file=sys.stderr,
        )
        return 1
    print(f"Successful utterances: {len(measurements)}; failures: {len(failures)}")
    print(f"Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
