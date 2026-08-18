from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from pykokoro.tokenizer import TokenizerConfig

from .polynorm_data import POLYNORM_TO_PYKOKORO_LANGUAGE, PolyNormDataError, load_cases
from .polynorm_eval import (
    PyKokoroPhonemeHarness,
    aggregate_results,
    collect_environment_fingerprint,
    direct_kokorog2p_observer,
    direct_spokenform_observer,
    evaluate_case,
    load_baseline,
    load_quarantine,
    write_failure_reports,
    write_summary,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the PolyNorm end-to-end phoneme benchmark against PyKokoro."
    )
    parser.add_argument("--accept-license", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--locale", choices=sorted(POLYNORM_TO_PYKOKORO_LANGUAGE))
    parser.add_argument("--category")
    parser.add_argument("--case", dest="case_id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--backend", default="kokorog2p")
    parser.add_argument("--pipeline", choices=("plain", "ssmd", "both"), default="plain")
    parser.add_argument(
        "--baseline",
        default="benchmarks/baselines/polynorm_phoneme.json",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--show-failures", choices=("none", "all"), default="none")
    parser.add_argument("--results-dir")
    parser.add_argument("--cache-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.offline and args.refresh:
        print("error: --offline and --refresh cannot be used together", file=sys.stderr)
        return 2

    locales = [args.locale] if args.locale else None
    case_ids = [args.case_id] if args.case_id else None
    try:
        cases = load_cases(
            locales=locales,
            category=args.category,
            case_ids=case_ids,
            limit=args.limit,
            cache_dir=args.cache_dir,
            accept_license=args.accept_license,
            offline=args.offline,
            refresh=args.refresh,
        )
    except PolyNormDataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.download_only:
        print(f"Prepared {len(cases)} cached PolyNorm rows.")
        return 0
    if not cases:
        print("error: no PolyNorm cases matched the requested filters", file=sys.stderr)
        return 2

    pipelines = ("plain", "ssmd") if args.pipeline == "both" else (args.pipeline,)
    tokenizer_config = TokenizerConfig(
        backend=args.backend,
        use_spacy=False,
        load_gold=True,
        load_silver=True,
        use_espeak_fallback=True,
    )
    environment = collect_environment_fingerprint(
        backend=args.backend,
        tokenizer_config=tokenizer_config,
        pipelines=pipelines,
    )
    baseline_ids: list[str] = []
    baseline_path = Path(args.baseline)
    if baseline_path.exists():
        baseline = load_baseline(baseline_path)
        if baseline["dataset_commit"] not in {None, environment["polynorm_commit"]}:
            print("error: baseline dataset commit does not match this benchmark run", file=sys.stderr)
            return 2
        if baseline["config_hash"] not in {None, environment["config_hash"]}:
            print("error: baseline config hash does not match this benchmark run", file=sys.stderr)
            return 2
        baseline_ids = baseline["allowed_failure_ids"]

    quarantine_path = Path("benchmarks/polynorm_quarantine.json")
    quarantine = load_quarantine(quarantine_path) if quarantine_path.exists() else {}
    results = []
    harnesses: dict[tuple[str, str], PyKokoroPhonemeHarness] = {}
    try:
        for case in cases:
            language = POLYNORM_TO_PYKOKORO_LANGUAGE[case.polynorm_locale]
            for pipeline in pipelines:
                key = (language, pipeline)
                harness = harnesses.get(key)
                if harness is None:
                    harness = PyKokoroPhonemeHarness(
                        language,
                        args.backend,
                        ssmd=pipeline == "ssmd",
                        tokenizer_config=tokenizer_config,
                    )
                    harnesses[key] = harness
                results.append(
                    evaluate_case(
                        case,
                        harness,
                        pipeline=pipeline,
                        direct_kokorog2p=direct_kokorog2p_observer(
                            language=language,
                            tokenizer_config=harness.tokenizer_config,
                            model_variant=harness.resolved_config.model_variant,
                        ),
                        direct_spokenform=direct_spokenform_observer(case.polynorm_locale),
                        quarantine=quarantine,
                    )
                )
    finally:
        for harness in harnesses.values():
            harness.close()

    summary = aggregate_results(results, environment=environment, baseline_failure_ids=baseline_ids)
    results_dir = Path(args.results_dir) if args.results_dir else _default_results_dir()
    summary_path = write_summary(results_dir / "summary.json", summary)
    failure_jsonl, failure_md = write_failure_reports(results_dir, results)
    if args.show_failures == "all":
        print(failure_md.read_text(encoding="utf-8").rstrip())

    print(
        f"Evaluated {summary['counts']['evaluated_rows']} rows across {len(pipelines)} pipeline(s)."
    )
    print(f"Semantic exact: {summary['metrics']['semantic_phoneme_exact']}")
    print(f"Token exact: {summary['metrics']['token_exact']}")
    print(f"Failures: {summary['counts']['failure_rows']}")
    print(f"Summary: {summary_path}")
    print(f"Failures JSONL: {failure_jsonl}")

    if args.strict:
        return 1 if summary["counts"]["failure_rows"] else 0
    return 1 if summary["baseline_comparison"]["new_failures"] else 0


def _default_results_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(".benchmarks") / "polynorm" / stamp


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
