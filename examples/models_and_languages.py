#!/usr/bin/env python3
"""Discover public model/voice capabilities, then optionally synthesize one result.

Discovery reads registry metadata only and never installs model assets. Unless
``--offline`` is selected, registry metadata may be refreshed from its configured
source. Selecting ``--model`` performs synthesis and may install model assets.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import soundfile as sf

try:
    from ._output import artifact_dir
except ImportError:
    from _output import artifact_dir

from pykokoro import (
    GenerationConfig,
    KokoroSynthesizer,
    ModelCapabilities,
    ModelDiscoveryResult,
    SynthesisConfig,
    discover_models,
)

OUTPUT_DIR = Path("model_language_outputs")
SAMPLE_TEXTS = {
    "en": "Hello. This is PyKokoro speaking with the selected model.",
    "de": "Hallo. Dies ist eine deutsche PyKokoro-Demonstration.",
    "es": "Hola. Esta es una demostración de PyKokoro.",
    "fr": "Bonjour. Ceci est une démonstration de PyKokoro.",
    "hi": "नमस्ते। यह PyKokoro का एक छोटा सा उदाहरण है।",
    "it": "Ciao. Questa è una dimostrazione di PyKokoro.",
    "ja": "こんにちは。これは PyKokoro の音声サンプルです。",
    "pt": "Olá. Esta é uma demonstração do PyKokoro.",
    "zh": "你好。这是 PyKokoro 的语音示例。",
    "vi": "Xin chào. Đây là một ví dụ giọng nói của PyKokoro.",
    "sv": "Hej. Det här är ett röstexempel från PyKokoro.",
    "th": "สวัสดี นี่คือตัวอย่างเสียงจาก PyKokoro",
    "ru": "Привет. Это пример голоса PyKokoro.",
    "cs": "Dobrý den. Toto je ukázka hlasu PyKokoro.",
}


def list_models(inventory: ModelDiscoveryResult) -> None:
    """Print the current public model and voice inventory without installing assets."""
    print("PyKokoro model and language capabilities")
    print(f"Registry: {inventory.registry_source} (offline={inventory.offline})")
    for model in inventory.models:
        print(f"\n{model.model_id} [{model.status}] source={model.source}")
        print(f"  Languages: {', '.join(model.languages) or 'none'}")
        print(f"  Default voice: {model.default_voice}")
        print(f"  Voices ({len(model.voices)}): {', '.join(model.voices) or 'none'}")
        print(f"  Qualities: {', '.join(model.qualities) or 'none'}")
        print(f"  Provider: {model.provider or 'unspecified'}")
        print(f"  Frontend: {model.frontend}")
        print(f"  Runtime available: {model.runtime_available}")
        print(f"  Experimental: {model.experimental}")
        print(f"  Redistribution allowed: {model.redistribution_allowed}")
        if model.sample_rate is not None:
            print(f"  Sample rate: {model.sample_rate}")
        if model.max_tokens is not None:
            print(f"  Maximum tokens: {model.max_tokens}")
        for voice in model.voice_details:
            print(
                f"  Voice detail: {voice.name} ({voice.language_label}, "
                f"{voice.gender}, locale={voice.locale})"
            )


def _choose_language(model: ModelCapabilities, language: str | None) -> str:
    if not model.languages:
        raise ValueError(f"Model {model.model_id!r} declares no languages")
    if language is None:
        return model.languages[0]
    normalized = language.casefold().replace("_", "-")
    exact = next(
        (
            candidate
            for candidate in model.languages
            if candidate.casefold().replace("_", "-") == normalized
        ),
        None,
    )
    if exact is not None:
        return exact
    base = normalized.split("-", 1)[0]
    compatible = next(
        (
            candidate
            for candidate in model.languages
            if candidate.casefold().split("-", 1)[0] == base
        ),
        None,
    )
    if compatible is None:
        raise ValueError(
            f"Language {language!r} is not declared by model {model.model_id!r}. "
            f"Available: {', '.join(model.languages)}"
        )
    return compatible


def _choose_voice(model: ModelCapabilities, voice: str | None) -> str:
    selected = model.default_voice if voice is None else voice
    if selected not in model.voices:
        raise ValueError(
            f"Voice {selected!r} is not available for {model.model_id!r}. "
            f"Available: {', '.join(model.voices)}"
        )
    return selected


def _choose_quality(model: ModelCapabilities, quality: str | None) -> str:
    if not model.qualities:
        raise ValueError(f"Model {model.model_id!r} declares no available qualities")
    selected = quality or ("fp32" if "fp32" in model.qualities else model.qualities[0])
    if selected not in model.qualities:
        raise ValueError(
            f"Quality {selected!r} is not available for {model.model_id!r}. "
            f"Available: {', '.join(model.qualities)}"
        )
    return selected


def _safe_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "model"


def synthesize(
    inventory: ModelDiscoveryResult,
    *,
    model_id: str,
    language: str | None,
    voice: str | None,
    quality: str | None,
    include_experimental: bool,
    output_dir: Path,
) -> Path:
    """Synthesize one model selected from public discovery metadata."""
    model = next((item for item in inventory.models if item.model_id == model_id), None)
    if model is None:
        raise ValueError(f"Unknown model {model_id!r}")
    if not model.runtime_available or not model.redistribution_allowed:
        raise ValueError(f"Model {model_id!r} is not currently runnable")
    if model.status not in {"ready", "experimental"}:
        raise ValueError(f"Model {model_id!r} is not runnable: {model.status}")
    if model.experimental and not include_experimental:
        raise ValueError("This model uses an experimental frontend; pass --include-experimental")

    selected_language = _choose_language(model, language)
    selected_voice = _choose_voice(model, voice)
    selected_quality = _choose_quality(model, quality)
    config = SynthesisConfig(
        model_source=model.source,
        model_variant=model.model_id,
        model_quality=selected_quality,
        voice=selected_voice,
        allow_experimental_frontend=model.experimental and include_experimental,
        generation=GenerationConfig(lang=selected_language),
        return_trace=True,
    )
    text = SAMPLE_TEXTS.get(selected_language.casefold().split("-", 1)[0], SAMPLE_TEXTS["en"])
    with KokoroSynthesizer(config) as synthesizer:
        result = synthesizer.synthesize_text(text, language=selected_language, voice=selected_voice)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        f"{_safe_filename_part(model_id)}_{_safe_filename_part(selected_language)}_"
        f"{_safe_filename_part(selected_voice)}.wav"
    )
    sf.write(output, result.audio, result.sample_rate)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", metavar="MODEL_ID")
    parser.add_argument("--language", metavar="LANGUAGE")
    parser.add_argument("--voice", metavar="VOICE")
    parser.add_argument("--quality", metavar="QUALITY")
    parser.add_argument("--offline", action="store_true", help="use cached discovery metadata only")
    parser.add_argument("--refresh", action="store_true", help="refresh registry metadata")
    parser.add_argument(
        "--preference", choices=("auto", "github", "huggingface", "upstream"), default="auto"
    )
    parser.add_argument("--include-experimental", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=artifact_dir() / OUTPUT_DIR)
    args = parser.parse_args(argv)

    try:
        inventory = discover_models(
            offline=args.offline, refresh=args.refresh, preference=args.preference
        )
        if args.model is None:
            list_models(inventory)
            return 0
        output = synthesize(
            inventory,
            model_id=args.model,
            language=args.language,
            voice=args.voice,
            quality=args.quality,
            include_experimental=args.include_experimental,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    else:
        print(f"Created {output}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
