"""Enroll, persist, reload, and synthesize with an English reference voice.

Run with ``python examples/reference_voice.py reference.wav 'exact transcript' 'target text'``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path
from pykokoro import KokoroSynthesizer, ReferenceVoice


def run(
    reference_audio: Path,
    reference_text: str,
    text: str,
    state_path: Path,
    output_path: Path,
) -> None:
    """Enroll one reference and render a target request after reloading its saved state."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with KokoroSynthesizer() as synthesizer:
        enrolled = synthesizer.enroll_voice(
            reference_audio,
            reference_text,
            engine="akinvox",
            language="en-us",
            name="speaker",
        )
        enrolled.save(state_path)
        reusable = ReferenceVoice.load(state_path)
        rendered = synthesizer.synthesize_text(text, language="en-us", voice=reusable)
        rendered.save_wav(output_path)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_audio", type=Path)
    parser.add_argument("reference_text", help="exact transcript of the reference recording")
    parser.add_argument("text", help="English text to synthesize")
    parser.add_argument("--state", type=Path, default=Path("speaker-reference.npz"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output_path = args.output or artifact_path("reference_voice.wav")
    run(args.reference_audio, args.reference_text, args.text, args.state, output_path)
    print(f"Saved reusable reference state to {args.state}")
    print(f"Saved synthesized audio to {output_path}")


if __name__ == "__main__":
    main()
