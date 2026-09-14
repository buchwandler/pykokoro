#!/usr/bin/env python3
"""Compare TTS-controlled and automatic parenthetical pauses in English.

Prerequisites:
    pip install -e ".[cpu]"

Usage:
    python examples/english_parenthetical_pause.py
"""

from __future__ import annotations

from typing import Any

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path


SAMPLE_FINAL = "They changed out their clothes (stained with blood)."
SAMPLE_MEDIAL = "They changed out their clothes (stained with blood) before entering the room."
LANG = "en-us"
VOICE = "af_heart"
PAUSE_PARENTHETICAL = 0.15


def _diagnostic(result: Any) -> dict[str, int]:
    if result.trace is None:
        return {}
    event = next(
        (item for item in result.trace.events if item.name == "parenthetical_boundaries"),
        None,
    )
    return {} if event is None else dict(event.details)


def main() -> None:
    """Generate final and medial parenthetical pause comparisons."""
    pipeline = KokoroPipeline(
        PipelineConfig(
            voice=VOICE,
            generation=GenerationConfig(lang=LANG),
            return_trace=True,
            retain_segment_audio=False,
        )
    )
    tts_config = GenerationConfig(lang=LANG, pause_mode="tts", pause_variance=0.0)
    auto_config = GenerationConfig(
        lang=LANG,
        pause_mode="auto",
        pause_parenthetical=PAUSE_PARENTHETICAL,
        pause_clause=0.0,
        pause_sentence=0.0,
        pause_paragraph=0.0,
        pause_variance=0.0,
    )

    outputs = (
        ("final", SAMPLE_FINAL),
        ("medial", SAMPLE_MEDIAL),
    )
    for label, text in outputs:
        tts_result = pipeline.run(text, generation=tts_config)
        auto_result = pipeline.run(text, generation=auto_config)
        tts_path = artifact_path(f"english_parenthetical_{label}_tts.wav")
        auto_path = artifact_path(f"english_parenthetical_{label}_auto.wav")
        tts_result.save_wav(str(tts_path))
        auto_result.save_wav(str(auto_path))
        print(f"{label.title()} text: {text}")
        print(f"  TTS-controlled output: {tts_path}")
        print(f"  Automatic output:       {auto_path}")
        print(f"  Automatic trace: {_diagnostic(auto_result)}")

    pipeline.close()
    print(f"Automatic parenthetical pause: {PAUSE_PARENTHETICAL:.2f}s")
    print("Listen for a short detachment before the aside and before resumed medial host text.")


if __name__ == "__main__":
    main()
