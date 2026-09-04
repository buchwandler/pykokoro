from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from .acoustic import run_acoustic
from .baseline import compare_baselines, load_baseline, make_baseline, save_baseline
from .data import available_languages, available_locales, load_cases
from .frontend import FrontendVariant, NoOnnxFrontend
from .metrics import summarize
from .phonemes import evaluate_case
from .reports import environment_fingerprint, write_reports
from .schema import CATEGORIES, HardCaseError
from .segmentation import evaluate_plan


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PyKokoro's first-party hard-case benchmark.")
    parser.add_argument("--list-languages", action="store_true")
    parser.add_argument("--list-locales", action="store_true")
    parser.add_argument("--list-categories", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--language")
    parser.add_argument("--locale")
    parser.add_argument("--category")
    parser.add_argument("--case", dest="case_id")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--level",
        choices=("normalization", "phoneme", "plan", "frontend", "acoustic", "all"),
        default="frontend",
    )
    parser.add_argument("--ssmd", action="store_true")
    parser.add_argument("--backend", default="kokorog2p")
    parser.add_argument("--frontend-variant", default="default")
    parser.add_argument("--lexicon", help="Comma-separated named frontend lexicons")
    parser.add_argument("--model")
    parser.add_argument("--voice")
    parser.add_argument("--render-audio", action="store_true")
    parser.add_argument("--results-dir")
    parser.add_argument("--baseline")
    parser.add_argument("--write-baseline")
    parser.add_argument("--show-details", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.list_languages:
        print("\n".join(available_languages()))
        return 0
    if args.list_locales:
        print("\n".join(available_locales()))
        return 0
    if args.list_categories:
        print("\n".join(CATEGORIES))
        return 0
    try:
        if args.list_cases:
            for case in load_cases(
                language=args.language, locale=args.locale, category=args.category
            ):
                print(case.id)
            return 0
        language, locale = _resolve_language_locale(args.language, args.locale)
        cases = load_cases(
            language=language, locale=locale, category=args.category, case_id=args.case_id
        )
        if args.limit is not None:
            if args.limit < 0:
                raise HardCaseError("limit must be non-negative")
            cases = cases[: args.limit]
        if not cases:
            raise HardCaseError("no hard-cases matched the requested filters")
        options = {"load_gold": True, "load_silver": True, "use_espeak_fallback": True}
        if args.lexicon:
            options["lexicons"] = tuple(
                item.strip() for item in args.lexicon.split(",") if item.strip()
            )
        variant = FrontendVariant(
            id=args.frontend_variant, language=_runtime_language(locale), options=options
        )
        levels = (
            ("normalization", "phoneme", "plan", "acoustic")
            if args.level == "all"
            else (args.level,)
        )
        results = []
        plans = []
        acoustics = []
        with NoOnnxFrontend(
            locale, backend=args.backend, ssmd=args.ssmd, variant=variant
        ) as frontend:
            for case in cases:
                if "normalization" in levels or "phoneme" in levels or "frontend" in levels:
                    results.append(evaluate_case(case, frontend, level=args.level))
                if "plan" in levels:
                    plans.append(evaluate_plan(case, frontend))
                if "acoustic" in levels:
                    output = None
                    if args.render_audio:
                        output = _audio_path(args.results_dir, case.id)
                    acoustics.append((case, run_acoustic(frontend, case.text, render_audio=output)))
        environment = environment_fingerprint(
            {
                "language": language,
                "locale": locale,
                "level": args.level,
                "backend": args.backend,
                "variant": args.frontend_variant,
                "lexicon": args.lexicon,
            }
        )
        paths = write_reports(
            args.results_dir or _default_results_dir(), results, environment=environment
        )
        _print_summary(results, plans, acoustics, paths, args.show_details)
        if args.write_baseline:
            baseline = make_baseline(
                summarize(results, environment=environment),
                case_ids=[case.id for case in cases],
                environment=environment,
            )
            save_baseline(args.write_baseline, baseline)
        if args.baseline:
            comparison = compare_baselines(
                load_baseline(args.baseline),
                make_baseline(
                    summarize(results, environment=environment),
                    case_ids=[case.id for case in cases],
                    environment=environment,
                ),
            )
            print(f"New baseline failures: {len(comparison['new_failures'])}")
            if comparison["new_failures"]:
                return 1
        return 1 if args.strict and any(item.failed for item in results) else 0
    except (HardCaseError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _resolve_language_locale(language: str | None, locale: str | None) -> tuple[str, str]:
    if locale is not None:
        if locale not in available_locales():
            raise HardCaseError(f"unsupported locale: {locale!r}")
        resolved = "de" if locale.startswith("de") else "en"
        if language is not None and language != resolved:
            raise HardCaseError(f"locale {locale!r} does not belong to language {language!r}")
        return resolved, locale
    resolved = language or "en"
    if resolved not in available_languages():
        raise HardCaseError(f"unsupported language: {resolved!r}")
    return resolved, "de-DE" if resolved == "de" else "en-US"


def _runtime_language(locale: str) -> str:
    return {"en-US": "en-us", "en-GB": "en-gb", "de-DE": "de"}[locale]


def _default_results_dir() -> Path:
    return (
        Path(".benchmarks") / "hard_cases" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )


def _audio_path(results_dir: str | None, case_id: str) -> Path:
    return Path(results_dir or _default_results_dir()) / "audio" / f"{case_id}.wav"


def _print_summary(
    results: list[object],
    plans: list[object],
    acoustics: list[object],
    paths: dict[str, Path],
    details: bool,
) -> None:
    print(f"Evaluated {len(results) or len(plans) or len(acoustics)} case(s).")
    if results:
        summary = summarize(results)
        print(f"Frontend failures: {summary['counts']['cases_failed']}")
        if details:
            for item in results:
                print(
                    f"{item.case_id}: {'PASS' if not item.failed else 'FAIL'} owner={item.likely_owner}"
                )
    if plans:
        print(f"Plan failures: {sum(not item.passed for item in plans)}")
    if acoustics:
        print(f"Acoustic health failures: {sum(not item.passed for _, item in acoustics)}")
    print(f"Summary: {paths['summary']}")


if __name__ == "__main__":
    raise SystemExit(main())
