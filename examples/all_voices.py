"""Discover every runnable PyKokoro voice and render one identification WAV."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import soundfile as sf
from audiosig import measure_loudness

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path

from pykokoro import LoudnessConfig
from pykokoro.discovery import (
    ModelCapabilities,
    ModelDiscoveryResult,
    VoiceCapabilities,
    VoiceGender,
    discover_models,
)
from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.short_sentence_handler import ShortSentenceConfig
from pykokoro.voice_level import VoiceCalibrationKey, default_voice_calibration

RUNNABLE_STATUSES = {"ready", "experimental"}
MODEL_PRIORITY = {"v1.0": 0, "v1.1-zh": 1}
DEFAULT_PAUSE = 0.35
DEFAULT_OUTPUT = artifact_path("all_voices.wav")


class ShowcaseError(RuntimeError):
    """Raised when the all-voices artifact cannot be produced completely."""


@dataclass(frozen=True, slots=True)
class VoiceShowcaseEntry:
    number: int
    model_id: str
    model_source: str
    quality: str
    voice: str
    language: str
    locale: str
    language_label: str
    gender: VoiceGender
    experimental: bool
    status: str
    sample_rate: int


@dataclass(frozen=True, slots=True)
class SkippedModel:
    model_id: str
    status: str
    voice_count: int


@dataclass(frozen=True, slots=True)
class ShowcaseCatalog:
    entries: tuple[VoiceShowcaseEntry, ...]
    skipped: tuple[SkippedModel, ...]
    registry_source: str


@dataclass(frozen=True, slots=True)
class SpeechLocale:
    locale: str
    render: Callable[[int, str, VoiceGender, str], str]


def choose_quality(model: ModelCapabilities) -> str:
    """Select the stable quality used for one model group."""
    if "fp32" in model.qualities:
        return "fp32"
    if not model.qualities:
        raise ShowcaseError(f"{model.model_id}: no registry-declared model quality")
    return model.qualities[0]


def model_sort_key(model: ModelCapabilities) -> tuple[int, str, str]:
    return (MODEL_PRIORITY.get(model.model_id, 100), model.model_id, model.source)


def _fallback_voice_detail(model: ModelCapabilities, voice: str) -> VoiceCapabilities:
    if not model.languages:
        raise ShowcaseError(f"{model.model_id}/{voice}: no model language metadata")
    language = model.languages[0]
    return VoiceCapabilities(
        name=voice,
        gender="unknown",
        language=language,
        locale=language,
        language_label=language,
    )


def _validate_voice_detail(model: ModelCapabilities, voice: str, detail: VoiceCapabilities) -> None:
    if detail.name != voice:
        raise ShowcaseError(f"{model.model_id}/{voice}: metadata is keyed to {detail.name!r}")
    if not voice or any(character.isspace() for character in voice):
        raise ShowcaseError(f"{model.model_id}/{voice}: invalid registry voice identifier")
    if not detail.language or not detail.language_label:
        raise ShowcaseError(f"{model.model_id}/{voice}: incomplete voice metadata")
    if detail.gender == "unknown":
        return
    if not detail.locale:
        raise ShowcaseError(f"{model.model_id}/{voice}: incomplete voice metadata")
    if detail.locale not in ANNOUNCEMENT_BUILDERS:
        raise ShowcaseError(
            f"{model.model_id}/{voice}: no announcement template for {detail.locale!r}"
        )


def build_catalog(
    discovery: ModelDiscoveryResult,
    *,
    include_experimental: bool = True,
) -> ShowcaseCatalog:
    """Build a stable numbered catalog without loading synthesis assets."""
    pending: list[VoiceShowcaseEntry] = []
    skipped: list[SkippedModel] = []

    for model in sorted(discovery.models, key=model_sort_key):
        if model.status not in RUNNABLE_STATUSES or (
            model.status == "experimental" and not include_experimental
        ):
            skipped.append(SkippedModel(model.model_id, model.status, len(model.voices)))
            continue
        if not model.source:
            raise ShowcaseError(f"{model.model_id}: runnable model has no source")
        if model.sample_rate is None or model.sample_rate <= 0:
            raise ShowcaseError(f"{model.model_id}: runnable model has no sample rate")

        quality = choose_quality(model)
        details = {detail.name: detail for detail in model.voice_details}
        for voice in model.voices:
            detail = details.get(voice) or _fallback_voice_detail(model, voice)
            _validate_voice_detail(model, voice, detail)
            pending.append(
                VoiceShowcaseEntry(
                    number=0,
                    model_id=model.model_id,
                    model_source=model.source,
                    quality=quality,
                    voice=voice,
                    language=detail.language,
                    locale=detail.locale,
                    language_label=detail.language_label,
                    gender=detail.gender,
                    experimental=model.experimental,
                    status=model.status,
                    sample_rate=model.sample_rate,
                )
            )

    entries = tuple(replace(entry, number=index) for index, entry in enumerate(pending, start=1))
    return ShowcaseCatalog(entries, tuple(skipped), discovery.registry_source)


def spoken_voice_name(name: str) -> str:
    """Make a registry identifier intelligible without changing the selected key."""
    prefix, separator, suffix = name.partition("_")
    if not separator:
        return name
    prefix_speech = " ".join(prefix.upper())
    suffix_speech = " ".join(suffix) if suffix.isdigit() else suffix.replace("_", " ")
    return f"{prefix_speech} {suffix_speech}"


def _english(number: int, voice: str, gender: VoiceGender, label: str) -> str:
    return f"{number}. This is {voice}, a {gender} {label} voice."


def _german(number: int, voice: str, gender: VoiceGender, _label: str) -> str:
    gender_word = {"female": "weibliche", "male": "männliche", "neutral": "neutrale"}[gender]
    return f"{number}. Das ist {voice}, eine {gender_word} deutsche Stimme."


def _french(number: int, voice: str, gender: VoiceGender, _label: str) -> str:
    gender_word = {"female": "féminine", "male": "masculine", "neutral": "neutre"}[gender]
    return f"{number}. Voici {voice}, une voix française {gender_word}."


def _spanish(number: int, voice: str, gender: VoiceGender, _label: str) -> str:
    gender_word = {"female": "femenina", "male": "masculina", "neutral": "neutra"}[gender]
    return f"{number}. Esta es {voice}, una voz española {gender_word}."


def _italian(number: int, voice: str, gender: VoiceGender, _label: str) -> str:
    gender_word = {"female": "femminile", "male": "maschile", "neutral": "neutra"}[gender]
    return f"{number}. Questa è {voice}, una voce italiana {gender_word}."


def _portuguese(number: int, voice: str, gender: VoiceGender, _label: str) -> str:
    gender_word = {"female": "feminina", "male": "masculina", "neutral": "neutra"}[gender]
    return f"{number}. Esta é {voice}, uma voz portuguesa {gender_word}."


def _chinese(number: int, voice: str, gender: VoiceGender, _label: str) -> str:
    gender_word = {"female": "女性", "male": "男性", "neutral": "中性"}[gender]
    return f"{number}。这是{voice}，一位{gender_word}中文声音。"


def _vietnamese(number: int, voice: str, gender: VoiceGender, _label: str) -> str:
    gender_word = {"female": "nữ", "male": "nam", "neutral": "trung tính"}[gender]
    return f"{number}. Đây là {voice}, giọng nói tiếng Việt {gender_word}."


def _arabic(number: int, voice: str, gender: VoiceGender, _label: str) -> str:
    gender_word = {"female": "أنثى", "male": "ذكر", "neutral": "محايد"}[gender]
    return f"{number}. هذا هو {voice}، صوت عربي {gender_word}."


def _japanese(number: int, voice: str, gender: VoiceGender, _label: str) -> str:
    gender_word = {"female": "女性", "male": "男性", "neutral": "中性的"}[gender]
    return f"{number}。これは{voice}、{gender_word}の日本語音声です。"


def _swedish(number: int, voice: str, gender: VoiceGender, _label: str) -> str:
    gender_word = {"female": "kvinnlig", "male": "manlig", "neutral": "neutral"}[gender]
    return f"{number}. Det här är {voice}, en {gender_word} svensk röst."


def _hindi(number: int, voice: str, gender: VoiceGender, _label: str) -> str:
    gender_word = {"female": "महिला", "male": "पुरुष", "neutral": "तटस्थ"}[gender]
    return f"{number}. यह {voice} की {gender_word} हिंदी आवाज़ है।"


def _thai(number: int, voice: str, gender: VoiceGender, _label: str) -> str:
    gender_word = {"female": "ผู้หญิง", "male": "ผู้ชาย", "neutral": "เป็นกลาง"}[gender]
    return f"{number} นี่คือ {voice} เสียงภาษาไทย{gender_word}"


def _kazakh(number: int, voice: str, gender: VoiceGender, _label: str) -> str:
    gender_word = {"female": "әйел", "male": "ер", "neutral": "бейтарап"}[gender]
    return f"{number}. Бұл {voice}, {gender_word} қазақша дауыс."


def _russian(number: int, voice: str, gender: VoiceGender, _label: str) -> str:
    gender_word = {"female": "женский", "male": "мужской", "neutral": "нейтральный"}[gender]
    return f"{number}. Это {voice}, {gender_word} русский голос."


ANNOUNCEMENT_BUILDERS: dict[str, Callable[[int, str, VoiceGender, str], str]] = {
    "en-US": _english,
    "en-GB": _english,
    "de": _german,
    "es": _spanish,
    "fr": _french,
    "hi": _hindi,
    "it": _italian,
    "ja": _japanese,
    "kk": _kazakh,
    "pt": _portuguese,
    "pt-PT": _portuguese,
    "ru": _russian,
    "sv": _swedish,
    "th": _thai,
    "vi": _vietnamese,
    "ar": _arabic,
    "zh": _chinese,
}


def announcement_for(entry: VoiceShowcaseEntry) -> str:
    voice = spoken_voice_name(entry.voice)
    if entry.gender == "unknown":
        return f"{entry.number}. This is {voice}. Language: {entry.language_label}."
    builder = ANNOUNCEMENT_BUILDERS[entry.locale]
    return builder(entry.number, voice, entry.gender, entry.language_label)


def format_table(entries: Sequence[VoiceShowcaseEntry]) -> str:
    lines = [
        " No.  Voice             Gender  Locale  Language           Model               Status",
        "----  ----------------  ------  ------  -----------------  ------------------  ------------",
    ]
    lines.extend(
        f"{entry.number:4d}  {entry.voice:<16}  {entry.gender:<6}  {entry.locale:<6}  "
        f"{entry.language_label:<17}  {entry.model_id:<18}  {entry.status}"
        for entry in entries
    )
    return "\n".join(lines)


def print_skipped_models(skipped: Iterable[SkippedModel]) -> None:
    skipped = tuple(skipped)
    if not skipped:
        return
    print("\nSkipped models:")
    for item in skipped:
        noun = "voice" if item.voice_count == 1 else "voices"
        print(f"  {item.model_id:<18} {item.status:<24} {item.voice_count} {noun}")


def print_summary(catalog: ShowcaseCatalog, output: Path, pause: float) -> None:
    print("PyKokoro all-voices showcase")
    print(f"Registry: {catalog.registry_source}")
    print(f"Runnable models: {len({entry.model_id for entry in catalog.entries})}")
    print(f"Voices scheduled: {len(catalog.entries)}")
    print(f"Skipped models: {len(catalog.skipped)}")
    print(f"Output: {output}")
    print(f"Pause between voices: {pause:.2f} s\n")


def silence_for(sample_rate: int, seconds: float) -> np.ndarray:
    if not 0.0 <= seconds <= 5.0:
        raise ShowcaseError("pause must be between 0.0 and 5.0 seconds")
    return np.zeros(round(sample_rate * seconds), dtype=np.float32)


def _common_sample_rate(entries: Sequence[VoiceShowcaseEntry]) -> int:
    rates = {entry.sample_rate for entry in entries}
    if len(rates) != 1:
        raise ShowcaseError(
            f"All-voices output requires one common sample rate; found: {sorted(rates)}"
        )
    return rates.pop()


def _validate_audio(
    result: object, entry: VoiceShowcaseEntry, expected_sample_rate: int
) -> np.ndarray:
    sample_rate = int(result.sample_rate)
    if sample_rate != expected_sample_rate:
        raise ShowcaseError(
            f"{entry.model_id}/{entry.voice}: result sample rate {sample_rate} != {expected_sample_rate}"
        )
    audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ShowcaseError(f"{entry.model_id}/{entry.voice}: rendered audio is empty")
    if not np.isfinite(audio).all():
        raise ShowcaseError(f"{entry.model_id}/{entry.voice}: rendered audio is not finite")
    return audio


def _progress(entry: VoiceShowcaseEntry, total: int) -> None:
    print(
        f"[{entry.number:03d}/{total}] {entry.model_id} / {entry.voice} / {entry.locale} / {entry.gender}"
    )


def synthesize_catalog(
    catalog: ShowcaseCatalog,
    *,
    output: Path,
    pause: float,
    voice_leveling: str = "off",
    print_levels: bool = False,
) -> float:
    """Render the catalog with one reusable pipeline into an atomic WAV."""
    if not catalog.entries:
        raise ShowcaseError("No runnable voices discovered")
    sample_rate = _common_sample_rate(catalog.entries)
    silence = silence_for(sample_rate, pause)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".part.wav")
    temporary.unlink(missing_ok=True)
    first = catalog.entries[0]

    from pykokoro import KokoroPipeline

    config = PipelineConfig(
        model_source=first.model_source,
        model_variant=first.model_id,
        model_quality=first.quality,
        voice=first.voice,
        allow_experimental_frontend=first.experimental,
        generation=GenerationConfig(lang=first.locale, speed=1.0),
        short_sentence_config=ShortSentenceConfig(resolve_mode="wrap"),
        loudness=LoudnessConfig(voice_leveling=voice_leveling),
    )
    frames = 0
    try:
        with (
            KokoroPipeline(config) as pipeline,
            sf.SoundFile(
                temporary,
                mode="w",
                samplerate=sample_rate,
                channels=1,
                format="WAV",
                subtype="PCM_16",
            ) as writer,
        ):
            for index, entry in enumerate(catalog.entries):
                _progress(entry, len(catalog.entries))
                try:
                    result = pipeline.run(
                        announcement_for(entry),
                        model_source=entry.model_source,
                        model_variant=entry.model_id,
                        model_quality=entry.quality,
                        voice=entry.voice,
                        lang=entry.locale,
                        allow_experimental_frontend=entry.experimental,
                    )
                    audio = _validate_audio(result, entry, sample_rate)
                    if print_levels:
                        metrics = measure_loudness(audio, sample_rate=sample_rate)
                        key = VoiceCalibrationKey(
                            entry.model_source, entry.model_id, entry.quality, entry.voice
                        )
                        calibration = default_voice_calibration().voices.get(key)
                        gain_db = 0.0 if calibration is None else calibration.gain_db
                        reference = None if calibration is None else calibration.reference_lufs
                        print(
                            f"          measured={metrics.integrated_lufs:.2f} LUFS "
                            f"calibration={gain_db:+.2f} dB reference={reference!s}"
                        )
                except Exception as exc:
                    raise ShowcaseError(
                        f"Failed at voice {entry.number}/{len(catalog.entries)}:\n"
                        f"  model: {entry.model_id}\n  voice: {entry.voice}\n"
                        f"  locale: {entry.locale}\n  error: {exc}"
                    ) from exc
                writer.write(audio)
                frames += audio.size
                if index + 1 < len(catalog.entries):
                    writer.write(silence)
                    frames += silence.size
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return frames / sample_rate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--refresh-registry", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--preference", choices=("auto", "github", "huggingface", "upstream"), default="auto"
    )
    parser.add_argument("--voice-leveling", choices=("off", "calibrated"), default="off")
    parser.add_argument("--print-levels", action="store_true")
    parser.add_argument("--compare-leveling", action="store_true")
    parser.add_argument("--include-experimental", dest="include_experimental", action="store_true")
    parser.add_argument("--no-experimental", dest="include_experimental", action="store_false")
    parser.set_defaults(include_experimental=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.pause <= 5.0:
        print("all_voices: --pause must be between 0.0 and 5.0 seconds", file=sys.stderr)
        return 2
    try:
        discovery = discover_models(
            offline=args.offline,
            refresh=args.refresh_registry,
            preference=args.preference,
        )
        catalog = build_catalog(discovery, include_experimental=args.include_experimental)
        output = args.output.resolve()
        print_summary(catalog, output, args.pause)
        print(format_table(catalog.entries))
        print_skipped_models(catalog.skipped)
        if args.list_only:
            return 0
        reported_output = output
        if args.compare_leveling:
            raw_output = output.with_name(f"{output.stem}_raw{output.suffix}")
            calibrated_output = output.with_name(f"{output.stem}_calibrated{output.suffix}")
            synthesize_catalog(
                catalog,
                output=raw_output,
                pause=args.pause,
                voice_leveling="off",
                print_levels=args.print_levels,
            )
            duration = synthesize_catalog(
                catalog,
                output=calibrated_output,
                pause=args.pause,
                voice_leveling="calibrated",
                print_levels=args.print_levels,
            )
            reported_output = calibrated_output
            print(f"Comparison outputs: {raw_output}, {calibrated_output}")
        else:
            duration = synthesize_catalog(
                catalog,
                output=output,
                pause=args.pause,
                voice_leveling=args.voice_leveling,
                print_levels=args.print_levels,
            )
        print(f"\nRendered voices: {len(catalog.entries)}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Output: {reported_output}")
        return 0
    except Exception as exc:
        print(f"all_voices: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
