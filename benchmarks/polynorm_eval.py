from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from importlib import metadata
from pathlib import Path
from typing import Any

from pykokoro import __version__ as pykokoro_version
from pykokoro.constants import SUPPORTED_LANGUAGES
from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline import KokoroPipeline
from pykokoro.pipeline_config import PipelineConfig, resolve_model_defaults
from pykokoro.spacy_models import make_spacy_model_request
from pykokoro.stages.audio_generation.noop import NoopAudioGenerationAdapter
from pykokoro.stages.audio_postprocessing.noop import NoopAudioPostprocessingAdapter
from pykokoro.stages.doc_parsers.plain import PlainTextDocumentParser
from pykokoro.stages.doc_parsers.ssmd import SsmdDocumentParser
from pykokoro.stages.phoneme_processing.noop import NoopPhonemeProcessorAdapter
from pykokoro.tokenizer import TokenizerConfig

from .polynorm_data import (
    POLYNORM_COMMIT,
    POLYNORM_LICENSE,
    POLYNORM_REPOSITORY,
    POLYNORM_TO_PYKOKORO_LANGUAGE,
    PolyNormCase,
)

SEMANTIC_SOURCE_SYMBOLS = frozenset("$€£%@/°+=#&")
PRESENTATION_PUNCTUATION = frozenset(".,!?;:\"'“”‘’()[]{}-–—…")
SUMMARY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PhonemeObservation:
    phonemes: str
    tokens: tuple[int, ...]
    segment_count: int
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    case_id: str
    locale: str
    language: str
    category: str
    pipeline: str
    is_transformation: bool
    original_text: str
    normalized_text: str
    raw_phoneme_exact: bool
    semantic_phoneme_exact: bool
    token_exact: bool
    phoneme_edit_distance: int
    token_edit_distance: int
    original_segment_count: int
    expected_segment_count: int
    warnings: tuple[str, ...]
    likely_owner: str
    original_error: str | None = None
    expected_error: str | None = None
    original_observation: PhonemeObservation | None = None
    expected_observation: PhonemeObservation | None = None
    direct_kokorog2p_original: PhonemeObservation | None = None
    direct_kokorog2p_expected: PhonemeObservation | None = None
    direct_spokenform_text: str | None = None
    quarantined: bool = False
    quarantine_reason: str | None = None

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    @property
    def failed(self) -> bool:
        if self.quarantined:
            return False
        if self.original_error or self.expected_error:
            return True
        return not (self.semantic_phoneme_exact and self.token_exact)

    @property
    def failure_id(self) -> str:
        return f"{self.pipeline}:{self.case_id}"


class PyKokoroPhonemeHarness:
    def __init__(
        self,
        language: str,
        backend: str,
        *,
        ssmd: bool = False,
        tokenizer_config: TokenizerConfig | None = None,
    ) -> None:
        base_tokenizer = tokenizer_config or TokenizerConfig()
        tokenizer = replace(base_tokenizer, backend=backend, use_spacy=False)
        self.language = language
        self.backend = backend
        self.pipeline_name = "ssmd" if ssmd else "plain"
        self.tokenizer_config = tokenizer
        self.config = PipelineConfig(
            generation=GenerationConfig(lang=language),
            tokenizer_config=tokenizer,
            cache_dir=None,
            return_trace=True,
        )
        self.resolved_config = resolve_model_defaults(self.config)

        parser = SsmdDocumentParser() if ssmd else PlainTextDocumentParser()
        self.pipeline = KokoroPipeline(
            self.config,
            doc_parser=parser,
            phoneme_processing=NoopPhonemeProcessorAdapter(),
            audio_generation=NoopAudioGenerationAdapter(seconds_per_segment=0.0),
            audio_postprocessing=NoopAudioPostprocessingAdapter(),
        )

    def phonemize(self, text: str) -> PhonemeObservation:
        result = self.pipeline.run(text)
        phonemes = " ".join(
            segment.phonemes for segment in result.phoneme_segments if segment.phonemes
        )
        phonemes = _collapse_whitespace(phonemes)
        tokens = tuple(
            token for segment in result.phoneme_segments for token in segment.tokens
        )
        warnings = tuple(result.trace.warnings) if result.trace is not None else ()
        return PhonemeObservation(
            phonemes=phonemes,
            tokens=tokens,
            segment_count=len(result.phoneme_segments),
            warnings=warnings,
        )

    def close(self) -> None:
        self.pipeline.close()

    def __enter__(self) -> PyKokoroPhonemeHarness:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def semantic_phoneme_key(value: str) -> str:
    value = unicodedata.normalize("NFC", value)
    out: list[str] = []
    for char in value:
        if char in SEMANTIC_SOURCE_SYMBOLS:
            out.append(char)
            continue
        if char in PRESENTATION_PUNCTUATION:
            continue
        out.append(char)
    return "".join(out).replace(" ", "")


def edit_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for row, left_item in enumerate(left, start=1):
        current = [row]
        for col, right_item in enumerate(right, start=1):
            replace_cost = 0 if left_item == right_item else 1
            current.append(
                min(
                    previous[col] + 1,
                    current[col - 1] + 1,
                    previous[col - 1] + replace_cost,
                )
            )
        previous = current
    return previous[-1]


def evaluate_case(
    case: PolyNormCase,
    harness: PyKokoroPhonemeHarness | Any,
    *,
    pipeline: str,
    direct_kokorog2p: Callable[[str], PhonemeObservation] | None = None,
    direct_spokenform: Callable[[str], str | None] | None = None,
    quarantine: dict[str, str] | None = None,
) -> CaseEvaluation:
    original_observation: PhonemeObservation | None = None
    expected_observation: PhonemeObservation | None = None
    original_error: str | None = None
    expected_error: str | None = None

    try:
        original_observation = harness.phonemize(case.original_text)
    except Exception as exc:  # pragma: no cover - exercised via tests
        original_error = f"{type(exc).__name__}: {exc}"

    try:
        expected_observation = harness.phonemize(case.normalized_text)
    except Exception as exc:  # pragma: no cover - exercised via tests
        expected_error = f"{type(exc).__name__}: {exc}"

    warnings = tuple(
        warning
        for observation in (original_observation, expected_observation)
        if observation is not None
        for warning in observation.warnings
    )
    raw_phoneme_exact = False
    semantic_phoneme_exact = False
    token_exact = False
    phoneme_edit = 0
    token_edit = 0
    original_segments = original_observation.segment_count if original_observation else 0
    expected_segments = expected_observation.segment_count if expected_observation else 0

    if original_observation is not None and expected_observation is not None:
        raw_phoneme_exact = original_observation.phonemes == expected_observation.phonemes
        semantic_phoneme_exact = (
            semantic_phoneme_key(original_observation.phonemes)
            == semantic_phoneme_key(expected_observation.phonemes)
        )
        token_exact = original_observation.tokens == expected_observation.tokens
        phoneme_edit = edit_distance(
            tuple(original_observation.phonemes),
            tuple(expected_observation.phonemes),
        )
        token_edit = edit_distance(original_observation.tokens, expected_observation.tokens)
    elif original_observation is not None or expected_observation is not None:
        observed = original_observation or expected_observation
        assert observed is not None
        phoneme_edit = len(observed.phonemes)
        token_edit = len(observed.tokens)

    direct_original: PhonemeObservation | None = None
    direct_expected: PhonemeObservation | None = None
    direct_spokenform_text: str | None = None
    provisional = CaseEvaluation(
        case_id=case.case_id,
        locale=case.polynorm_locale,
        language=POLYNORM_TO_PYKOKORO_LANGUAGE[case.polynorm_locale],
        category=case.category,
        pipeline=pipeline,
        is_transformation=case.is_transformation,
        original_text=case.original_text,
        normalized_text=case.normalized_text,
        raw_phoneme_exact=raw_phoneme_exact,
        semantic_phoneme_exact=semantic_phoneme_exact,
        token_exact=token_exact,
        phoneme_edit_distance=phoneme_edit,
        token_edit_distance=token_edit,
        original_segment_count=original_segments,
        expected_segment_count=expected_segments,
        warnings=warnings,
        likely_owner="unknown",
        original_error=original_error,
        expected_error=expected_error,
        original_observation=original_observation,
        expected_observation=expected_observation,
        quarantined=case.case_id in (quarantine or {}),
        quarantine_reason=(quarantine or {}).get(case.case_id),
    )

    if provisional.failed and direct_kokorog2p is not None:
        direct_original = direct_kokorog2p(case.original_text)
        direct_expected = direct_kokorog2p(case.normalized_text)
    if provisional.failed and direct_spokenform is not None:
        direct_spokenform_text = direct_spokenform(case.original_text)

    return replace(
        provisional,
        direct_kokorog2p_original=direct_original,
        direct_kokorog2p_expected=direct_expected,
        direct_spokenform_text=direct_spokenform_text,
        likely_owner=_classify_likely_owner(
            provisional,
            direct_original=direct_original,
            direct_expected=direct_expected,
        ),
    )


def aggregate_results(
    results: Iterable[CaseEvaluation],
    *,
    environment: dict[str, Any],
    baseline_failure_ids: Iterable[str] = (),
) -> dict[str, Any]:
    all_results = list(results)
    evaluated = [result for result in all_results if not result.quarantined]
    failures = [result for result in evaluated if result.failed]
    baseline_ids = set(baseline_failure_ids)
    failure_ids = sorted(result.failure_id for result in failures)
    new_failures = sorted(set(failure_ids) - baseline_ids)
    resolved_failures = sorted(baseline_ids - set(failure_ids))
    remaining_failures = sorted(set(failure_ids) & baseline_ids)

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "dataset": {
            "repository": POLYNORM_REPOSITORY,
            "commit": POLYNORM_COMMIT,
            "license": POLYNORM_LICENSE,
        },
        "environment": environment,
        "config_hash": environment["config_hash"],
        "counts": {
            "total_rows": len(all_results),
            "evaluated_rows": len(evaluated),
            "quarantined_rows": sum(1 for result in all_results if result.quarantined),
            "transformation_rows": sum(1 for result in evaluated if result.is_transformation),
            "identity_rows": sum(1 for result in evaluated if not result.is_transformation),
            "failure_rows": len(failures),
            "warning_rows": sum(1 for result in evaluated if result.warning_count),
            "warning_count": sum(result.warning_count for result in evaluated),
            "error_rows": sum(
                1 for result in evaluated if result.original_error or result.expected_error
            ),
        },
        "metrics": _bucket_metrics(evaluated),
        "per_locale": _group_metrics(evaluated, lambda result: result.locale),
        "per_category": _group_metrics(evaluated, lambda result: result.category),
        "per_pipeline": _group_metrics(evaluated, lambda result: result.pipeline),
        "per_kind": {
            "transformation": _bucket_metrics(
                [result for result in evaluated if result.is_transformation]
            ),
            "identity": _bucket_metrics(
                [result for result in evaluated if not result.is_transformation]
            ),
        },
        "likely_owner": _likely_owner_counts(failures),
        "failure_ids": failure_ids,
        "baseline_comparison": {
            "new_failures": new_failures,
            "resolved_failures": resolved_failures,
            "remaining_failures": remaining_failures,
        },
    }
    return summary


def collect_environment_fingerprint(
    *,
    backend: str,
    tokenizer_config: TokenizerConfig,
    pipelines: Sequence[str],
) -> dict[str, Any]:
    model_variants = {
        locale: resolve_model_defaults(
            PipelineConfig(
                generation=GenerationConfig(lang=language),
                tokenizer_config=replace(tokenizer_config, backend=backend),
            )
        ).model_variant
        for locale, language in POLYNORM_TO_PYKOKORO_LANGUAGE.items()
    }
    fingerprint = {
        "benchmark_schema_version": SUMMARY_SCHEMA_VERSION,
        "pykokoro_version": pykokoro_version,
        "pykokoro_source_commit": _git_commit(),
        "kokorog2p_version": _package_version("kokorog2p"),
        "spokenform_version": _package_version("spokenform"),
        "abbr2words_version": _package_version("abbr2words"),
        "num2words_version": _package_version("num2words"),
        "phrasplit_version": _package_version("phrasplit"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "polynorm_repository": POLYNORM_REPOSITORY,
        "polynorm_commit": POLYNORM_COMMIT,
        "locale_language_map": dict(POLYNORM_TO_PYKOKORO_LANGUAGE),
        "backend": backend,
        "pipelines": list(pipelines),
        "use_spacy": tokenizer_config.use_spacy,
        "load_gold": tokenizer_config.load_gold,
        "load_silver": tokenizer_config.load_silver,
        "use_espeak_fallback": tokenizer_config.use_espeak_fallback,
        "use_goruut_fallback": tokenizer_config.use_goruut_fallback,
        "model_variants": model_variants,
    }
    fingerprint["config_hash"] = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return fingerprint


def load_quarantine(path: str | Path) -> dict[str, str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {str(case_id): "quarantined" for case_id in data}
    if isinstance(data, dict):
        if isinstance(data.get("cases"), dict):
            return {str(key): str(value) for key, value in data["cases"].items()}
        return {str(key): str(value) for key, value in data.items()}
    raise ValueError("Unsupported quarantine format.")


def load_baseline(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Baseline file must contain a JSON object.")
    allowed = data.get("allowed_failure_ids", [])
    if not isinstance(allowed, list) or not all(isinstance(item, str) for item in allowed):
        raise ValueError("Baseline allowed_failure_ids must be a list of strings.")
    return {
        "dataset_commit": data.get("dataset_commit"),
        "config_hash": data.get("config_hash"),
        "allowed_failure_ids": list(allowed),
    }


def direct_kokorog2p_observer(
    *,
    language: str,
    tokenizer_config: TokenizerConfig,
    model_variant: str | None = None,
) -> Callable[[str], PhonemeObservation]:
    import kokorog2p

    version = "1.1" if model_variant == "v1.1-zh" else "1.0"
    request = make_spacy_model_request(
        model=tokenizer_config.spacy_model,
        size=tokenizer_config.spacy_model_size,
    )
    kokorog2p_language = SUPPORTED_LANGUAGES.get(language, language)

    def observe(text: str) -> PhonemeObservation:
        g2p_instance = kokorog2p.get_g2p(
            language=kokorog2p_language,
            version=version,
            phoneme_quotes="curly",
            use_goruut_fallback=tokenizer_config.use_goruut_fallback,
            use_espeak_fallback=tokenizer_config.use_espeak_fallback,
            use_spacy=tokenizer_config.use_spacy,
            spacy_model=request.model,
            spacy_model_size=request.size,
            backend=tokenizer_config.backend,
            load_gold=tokenizer_config.load_gold,
            load_silver=tokenizer_config.load_silver,
        )
        result = kokorog2p.phonemize(
            text,
            language=kokorog2p_language,
            return_phonemes=True,
            return_ids=True,
            g2p=g2p_instance,
        )
        phonemes = _collapse_whitespace(
            str(getattr(result, "phonemes", None) or getattr(result, "phoneme", ""))
        )
        tokens = tuple(getattr(result, "ids", None) or getattr(result, "token_ids", []))
        warnings = tuple(str(warning) for warning in getattr(result, "warnings", ()))
        return PhonemeObservation(
            phonemes=phonemes,
            tokens=tokens,
            segment_count=1,
            warnings=warnings,
        )

    return observe


def direct_spokenform_observer(locale: str) -> Callable[[str], str | None]:
    def observe(text: str) -> str | None:
        try:
            from spokenform import PreparationConfig, prepare_for_kokorog2p
        except ImportError:
            return None
        prepared = prepare_for_kokorog2p(
            text,
            language=locale,
            config=PreparationConfig.for_kokorog2p(locale),
        )
        return getattr(prepared, "spoken_text", None)

    return observe


def write_summary(path: str | Path, summary: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def write_failure_reports(
    results_dir: str | Path,
    results: Iterable[CaseEvaluation],
) -> tuple[Path, Path]:
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    failures = [result for result in results if result.failed]
    jsonl_path = results_path / "failures.jsonl"
    markdown_path = results_path / "failures.md"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for result in failures:
            handle.write(json.dumps(_failure_record(result), ensure_ascii=False) + "\n")

    lines = ["# PolyNorm benchmark failures", ""]
    if not failures:
        lines.append("No failures.")
    for result in failures:
        lines.extend(
            [
                f"## {result.failure_id}",
                "",
                f"- locale: `{result.locale}`",
                f"- category: `{result.category}`",
                f"- likely owner: `{result.likely_owner}`",
                f"- original text: `{result.original_text}`",
                f"- expected text: `{result.normalized_text}`",
                f"- original phonemes: `{result.original_observation.phonemes if result.original_observation else ''}`",
                f"- expected phonemes: `{result.expected_observation.phonemes if result.expected_observation else ''}`",
                "",
            ]
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jsonl_path, markdown_path


def _classify_likely_owner(
    result: CaseEvaluation,
    *,
    direct_original: PhonemeObservation | None,
    direct_expected: PhonemeObservation | None,
) -> str:
    if result.quarantined:
        return "quarantined"
    if result.original_error or result.expected_error:
        return "pykokoro_pipeline"
    if not result.failed:
        if result.original_segment_count != result.expected_segment_count:
            return "segmentation_only"
        return "none"
    if direct_original is not None and direct_expected is not None:
        direct_semantic = semantic_phoneme_key(direct_original.phonemes) == semantic_phoneme_key(
            direct_expected.phonemes
        )
        direct_tokens = direct_original.tokens == direct_expected.tokens
        if direct_semantic and direct_tokens:
            return "pykokoro_pipeline"
        return "kokorog2p_or_spokenform"
    return "needs_manual_review"


def _group_metrics(
    results: Iterable[CaseEvaluation],
    key_fn: Callable[[CaseEvaluation], str],
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[CaseEvaluation]] = {}
    for result in results:
        buckets.setdefault(key_fn(result), []).append(result)
    return {name: _bucket_metrics(bucket) for name, bucket in sorted(buckets.items())}


def _bucket_metrics(results: Sequence[CaseEvaluation]) -> dict[str, Any]:
    count = len(results)
    expected_tokens = sum(
        len(result.expected_observation.tokens) if result.expected_observation else 0
        for result in results
    )
    return {
        "cases": count,
        "failures": sum(1 for result in results if result.failed),
        "raw_phoneme_exact": sum(1 for result in results if result.raw_phoneme_exact),
        "raw_phoneme_exact_rate": _rate(
            sum(1 for result in results if result.raw_phoneme_exact),
            count,
        ),
        "semantic_phoneme_exact": sum(
            1 for result in results if result.semantic_phoneme_exact
        ),
        "semantic_phoneme_exact_rate": _rate(
            sum(1 for result in results if result.semantic_phoneme_exact),
            count,
        ),
        "token_exact": sum(1 for result in results if result.token_exact),
        "token_exact_rate": _rate(
            sum(1 for result in results if result.token_exact),
            count,
        ),
        "phoneme_edit_distance": sum(result.phoneme_edit_distance for result in results),
        "token_edit_distance": sum(result.token_edit_distance for result in results),
        "token_error_rate": _rate(
            sum(result.token_edit_distance for result in results),
            expected_tokens,
        ),
    }


def _likely_owner_counts(results: Iterable[CaseEvaluation]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.likely_owner] = counts.get(result.likely_owner, 0) + 1
    return dict(sorted(counts.items()))


def _failure_record(result: CaseEvaluation) -> dict[str, Any]:
    return {
        "failure_id": result.failure_id,
        "case_id": result.case_id,
        "locale": result.locale,
        "category": result.category,
        "pipeline": result.pipeline,
        "likely_owner": result.likely_owner,
        "original_text": result.original_text,
        "normalized_text": result.normalized_text,
        "original_phonemes": result.original_observation.phonemes if result.original_observation else "",
        "expected_phonemes": result.expected_observation.phonemes if result.expected_observation else "",
        "original_tokens": list(result.original_observation.tokens) if result.original_observation else [],
        "expected_tokens": list(result.expected_observation.tokens) if result.expected_observation else [],
        "warnings": list(result.warnings),
        "direct_spokenform_text": result.direct_spokenform_text,
        "phoneme_edit_distance": result.phoneme_edit_distance,
        "token_edit_distance": result.token_edit_distance,
    }


def _collapse_whitespace(value: str) -> str:
    return " ".join(value.split())


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        output = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return output.stdout.strip() or None


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator
