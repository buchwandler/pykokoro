#!/usr/bin/env python3
"""Browse the canonical PyKokoro model registry and synthesize one model."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import soundfile as sf

try:
    from ._output import artifact_dir
except ImportError:
    from _output import artifact_dir

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig
from pykokoro.model_profiles import (
    get_registry_model_profile,
    normalize_language_code,
    registry_support_status,
)
from pykokoro.model_registry import ModelRegistryError, load_registry

OUTPUT_DIR = Path("model_language_outputs")

SAMPLE_TEXTS = {
    "en": "Hello. This is PyKokoro speaking with the selected model.",
    "es": "Hola. Esta es una demostración de PyKokoro.",
    "fr": "Bonjour. Ceci est une démonstration de PyKokoro.",
    "hi": "नमस्ते। यह PyKokoro का एक छोटा सा उदाहरण है।",
    "it": "Ciao. Questa è una dimostrazione di PyKokoro.",
    "ja": "こんにちは。これは PyKokoro の音声サンプルです。",
    "pt": "Olá. Esta é uma demonstração do PyKokoro.",
    "zh": "你好。这是 PyKokoro 的语音示例。",
    "de": "Hallo. Dies ist eine deutsche PyKokoro-Demonstration.",
    "vi": "Xin chào. Đây là một ví dụ giọng nói của PyKokoro.",
    "ar": "مَرْحَبًا. هٰذَا مِثَالٌ صَوْتِيٌّ لِـ PyKokoro.",
    "sv": "Hej. Det här är ett röstexempel från PyKokoro.",
    "th": "สวัสดี นี่คือตัวอย่างเสียงจาก PyKokoro",
    "kk": "Сәлем. Бұл PyKokoro дауыс үлгісі.",
    "ru": "Привет. Это пример голоса PyKokoro.",
}


def _status_for_model(model: Any, registry: Any) -> str:
    status = registry_support_status(model)
    if status == "ready":
        profile = get_registry_model_profile(model.model_id, registry=registry)
        if profile.frontend_experimental:
            return "experimental"
    return status


def _distribution_providers(model: Any) -> str:
    providers = sorted({distribution.provider for distribution in model.distributions})
    return ", ".join(providers) or "none"


def _qualities(model: Any) -> tuple[str, ...]:
    if not model.distributions:
        return ()
    distribution = model.distribution() if hasattr(model, "distribution") else None
    distributions = (distribution,) if distribution is not None else model.distributions
    qualities = {
        quality
        for selected_distribution in distributions
        for quality in selected_distribution.qualities
        if quality is not None
    }
    return tuple(sorted(qualities))


def list_models(registry: Any) -> None:
    """Print model metadata without downloading model weights."""
    print("PyKokoro model and language registry")
    print("(Inventory only; model weights and voice packs are not downloaded.)")
    for model_id, model in registry.models.items():
        status = _status_for_model(model, registry)
        print(f"\n{model_id} [{status}]")
        print(f"  Languages: {', '.join(model.language_codes) or 'none'}")
        print(f"  Provider: {_distribution_providers(model)}")
        print(f"  Default voice: {model.default_voice}")
        print(f"  Voices ({len(model.voices)}): {', '.join(model.voices)}")
        print(f"  Qualities: {', '.join(_qualities(model)) or 'none'}")
        print(f"  Frontend: {model.frontend}")
        print(f"  Runtime layout: {model.layout}")
        print(f"  Runtime available: {model.runtime_available}")
        print(f"  Redistribution allowed: {model.redistribution_allowed}")


def _choose_language(model: Any, language: str | None) -> str:
    declared = tuple(model.language_codes)
    if not declared:
        raise ValueError(f"Model {model.model_id!r} declares no language codes")
    if language is None:
        return declared[0]
    normalized = normalize_language_code(language)
    for candidate in declared:
        if normalize_language_code(candidate) == normalized:
            return candidate
    available = ", ".join(declared)
    raise ValueError(
        f"Language {language!r} is not declared by model {model.model_id!r}. "
        f"Available languages: {available}"
    )


def _choose_voice(model: Any, voice: str | None) -> str:
    selected = model.default_voice if voice is None else voice
    if selected not in model.voices:
        raise ValueError(
            f"Voice {selected!r} is not available for model {model.model_id!r}. "
            f"Available voices: {', '.join(model.voices)}"
        )
    return selected


def _choose_quality(model: Any, quality: str | None) -> str:
    available = _qualities(model)
    if not available:
        raise ValueError(f"Model {model.model_id!r} has no registry-declared qualities")
    if quality is None:
        return "fp32" if "fp32" in available else available[0]
    if quality not in available:
        raise ValueError(
            f"Quality {quality!r} is not available for model {model.model_id!r}. "
            f"Available qualities: {', '.join(available)}"
        )
    return quality


def _sample_text(language: str) -> str:
    normalized = normalize_language_code(language)
    return SAMPLE_TEXTS.get(
        normalized, SAMPLE_TEXTS.get(normalized.split("-", 1)[0], SAMPLE_TEXTS["en"])
    )


def _safe_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "model"


def synthesize(
    registry: Any,
    *,
    model_id: str,
    language: str | None,
    voice: str | None,
    quality: str | None,
    include_experimental: bool,
    output_dir: Path,
) -> Path:
    """Synthesize one selected, runnable registry model."""
    model = registry.model(model_id)
    status = _status_for_model(model, registry)
    if status == "registry-unavailable":
        raise ValueError(f"Model {model_id!r} is registry-unavailable and cannot be synthesized")
    if status == "restricted":
        raise ValueError(f"Model {model_id!r} is restricted and cannot be synthesized")
    if status == "experimental" and not include_experimental:
        raise ValueError(
            f"Model {model_id!r} uses an experimental frontend; "
            "pass --include-experimental to enable it"
        )
    if status != "ready" and status != "experimental":
        raise ValueError(f"Model {model_id!r} is not runnable: {status}")

    selected_language = _choose_language(model, language)
    selected_voice = _choose_voice(model, voice)
    selected_quality = _choose_quality(model, quality)
    profile = get_registry_model_profile(model_id, registry=registry)
    config = PipelineConfig(
        model_source=profile.source,
        model_variant=profile.variant,
        model_quality=selected_quality,
        voice=selected_voice,
        allow_experimental_frontend=profile.frontend_experimental and include_experimental,
        generation=GenerationConfig(lang=selected_language, speed=1.0),
        return_trace=True,
    )

    print(
        f"Synthesizing {model_id} ({selected_language}, voice={selected_voice}, "
        f"quality={selected_quality})"
    )
    with KokoroPipeline(config) as pipeline:
        result = pipeline.run(_sample_text(selected_language))

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        f"{_safe_filename_part(model_id)}_{_safe_filename_part(selected_language)}_"
        f"{_safe_filename_part(selected_voice)}.wav"
    )
    sf.write(output, result.audio, result.sample_rate)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", metavar="MODEL_ID")
    parser.add_argument("--language", metavar="LANGUAGE")
    parser.add_argument("--voice", metavar="VOICE")
    parser.add_argument("--quality", metavar="QUALITY")
    parser.add_argument("--include-experimental", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=artifact_dir() / OUTPUT_DIR)
    args = parser.parse_args()

    try:
        registry = load_registry()
        if args.model is None:
            list_models(registry)
            return 0
        output = synthesize(
            registry,
            model_id=args.model,
            language=args.language,
            voice=args.voice,
            quality=args.quality,
            include_experimental=args.include_experimental,
            output_dir=args.output_dir,
        )
    except (ModelRegistryError, ValueError) as exc:
        parser.error(str(exc))
    else:
        print(f"Created {output}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
