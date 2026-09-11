"""Measure PyKokoro startup phases in fresh subprocesses.

Use explicit ``--model-path`` and ``--voices-path`` for a no-download run.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pykokoro import GenerationConfig, PipelineConfig
from pykokoro.pipeline import KokoroPipeline
from pykokoro.short_sentence_handler import PhraseResolveMode, ShortSentenceConfig
from pykokoro.tokenizer import TokenizerConfig


@dataclass(frozen=True)
class Scenario:
    name: str
    text: str
    quality: str
    spacy: str
    short_mode: str


SCENARIOS = (
    Scenario("warm-fp32-short", "hello", "fp32", "auto", "wrap"),
    Scenario("warm-fp32-normal", "This is a normal startup sentence.", "fp32", "auto", "wrap"),
    Scenario("warm-q8-short", "hello", "q8", "auto", "wrap"),
    Scenario("spacy-auto-short", "hello", "fp32", "auto", "wrap"),
    Scenario("spacy-off-short", "hello", "fp32", "off", "wrap"),
    Scenario("phrase-short", "hello", "fp32", "off", "phrase"),
    Scenario("wrap-short", "hello", "fp32", "off", "wrap"),
)

_ELAPSED_RE = re.compile(r"elapsed_ms=(?P<value>[0-9]+(?:\.[0-9]+)?)")


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="kokoro-v1.0")
    parser.add_argument("--model-source", choices=("github", "huggingface"), default="huggingface")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--voices-path", type=Path)
    parser.add_argument("--scenario", choices=tuple(item.name for item in SCENARIOS))
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help="print scenarios without starting subprocesses")


def _scenario(name: str | None) -> Scenario:
    if name is None:
        return SCENARIOS[0]
    return next(item for item in SCENARIOS if item.name == name)



def _phase_total_matching(records: list[logging.LogRecord], prefixes: tuple[str, ...]) -> float:
    total = 0.0
    for record in records:
        message = record.getMessage()
        if any(prefix in message for prefix in prefixes):
            match = _ELAPSED_RE.search(message)
            if match:
                total += float(match.group("value"))
    return total


def _frontend_metrics(trace: Any) -> dict[str, float]:
    totals: dict[str, float] = {}
    for event in getattr(trace, "events", ()):
        stage = str(getattr(event, "stage", "unknown"))
        totals[stage] = totals.get(stage, 0.0) + float(getattr(event, "ms", 0.0))
    return totals


def _child(args: argparse.Namespace) -> int:
    scenario = _scenario(args.scenario)
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("pykokoro")
    handler = Capture()
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    started = time.perf_counter()
    try:
        tokenizer_config = None if scenario.spacy == "auto" else TokenizerConfig(use_spacy=scenario.spacy == "required")
        short_config = ShortSentenceConfig()
        if scenario.short_mode == "phrase":
            short_config = ShortSentenceConfig(
                resolve_mode="phrase",
                resolve_modes={"phrase": PhraseResolveMode(neutral_phrase="Say {segment}.")},
            )
        config = PipelineConfig(
            voice="af_sarah",
            model_source=args.model_source,
            model_variant=args.model,
            model_quality=scenario.quality,
            model_path=args.model_path,
            voices_path=args.voices_path,
            generation=GenerationConfig(lang="en-us"),
            tokenizer_config=tokenizer_config,
            short_sentence_config=short_config,
            return_trace=True,
        )
        with KokoroPipeline(config) as pipeline:
            warmup_started = time.perf_counter()
            pipeline.warmup()
            warmup_ms = (time.perf_counter() - warmup_started) * 1000.0
            render_started = time.perf_counter()
            result = pipeline.run(scenario.text, return_trace=True)
            render_ms = (time.perf_counter() - render_started) * 1000.0
        trace = result.trace
        inference_ms = sum(
            float(item.get("runtime_ms", 0.0))
            for item in getattr(trace, "inference", ())
            if isinstance(item, dict)
        )
        payload = {
            "scenario": scenario.name,
            "quality": scenario.quality,
            "spacy": scenario.spacy,
            "short_mode": scenario.short_mode,
            "process_ms": (time.perf_counter() - started) * 1000.0,
            "warmup_ms": warmup_ms,
            "render_ms": render_ms,
            "final_audio_ms": render_ms,
            "frontend_ms": _frontend_metrics(trace),
            "registry_ms": _phase_total_matching(records, ("registry.load.",)),
            "artifact_ms": _phase_total_matching(records, ("artifact.materialize.", "artifact.cache.verify.")),
            "onnx_ms": _phase_total_matching(records, ("onnx.session.",)),
            "inference_ms": inference_ms,
            "audio_samples": int(result.audio.size),
        }
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    print(json.dumps(payload, sort_keys=True))
    return 0


def _parent(args: argparse.Namespace) -> int:
    if args.dry_run:
        for scenario in SCENARIOS:
            print(json.dumps(scenario.__dict__, sort_keys=True))
        return 0
    scenarios = (_scenario(args.scenario),) if args.scenario else SCENARIOS
    for scenario in scenarios:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--scenario",
            scenario.name,
            "--model",
            args.model,
            "--model-source",
            args.model_source,
        ]
        for option, value in (("--model-path", args.model_path), ("--voices-path", args.voices_path)):
            if value is not None:
                command.extend((option, str(value)))
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        if completed.returncode:
            print(completed.stderr, file=sys.stderr, end="")
            return completed.returncode
        print(completed.stdout, end="")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)
    args = parser.parse_args()
    return _child(args) if args.child else _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
