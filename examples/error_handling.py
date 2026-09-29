#!/usr/bin/env python3
"""Catch expected request failures with narrow public exception types.

The normal request is valid and should synthesize successfully; the catches show
how an application can report invalid language, voice, pronunciation, or length.
Synthesis may install model assets.
"""

from __future__ import annotations

from pykokoro import (
    GenerationConfig,
    InvalidLanguageError,
    InvalidPronunciationError,
    InvalidVoiceError,
    KokoroSynthesizer,
    PronunciationOverride,
    SynthesisConfig,
    SynthesisInputTooLongError,
    SynthesisSegment,
)

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path


def make_config() -> SynthesisConfig:
    return SynthesisConfig(
        voice="af_sarah",
        generation=GenerationConfig(lang="en-us"),
    )


def make_request() -> SynthesisSegment:
    return SynthesisSegment(
        id="typed-error-handling",
        text="Hello Welt.",
        language="en-us",
        voice="af_sarah",
        pronunciation_overrides=(PronunciationOverride(start=6, end=10, language="de"),),
    )


def main() -> None:
    with KokoroSynthesizer(make_config()) as synthesizer:
        try:
            result = synthesizer.synthesize(make_request())
        except InvalidLanguageError as exc:
            print(f"Choose a supported request language: {exc}")
        except InvalidVoiceError as exc:
            print(f"Choose a voice available for the selected model: {exc}")
        except InvalidPronunciationError as exc:
            print(f"Check pronunciation span offsets and values: {exc}")
        except SynthesisInputTooLongError as exc:
            print(f"Shorten the request or enable sentence splitting: {exc}")
        else:
            output = artifact_path("error_handling.wav")
            result.save_wav(output)
            print(f"Wrote {output}")


if __name__ == "__main__":
    main()
