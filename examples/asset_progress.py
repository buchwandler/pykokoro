#!/usr/bin/env python3
"""Report managed asset downloads using the console reporter or a callback.

The first synthesis may download model assets. Progress is emitted only for
runtime-managed model assets, not lexicon provisioning.
"""

from __future__ import annotations

import argparse

from pykokoro import (
    AssetProgressEvent,
    ConsoleAssetProgress,
    GenerationConfig,
    KokoroSynthesizer,
    SynthesisConfig,
)

try:
    from ._output import artifact_path
except ImportError:
    from _output import artifact_path


def progress(event: AssetProgressEvent) -> None:
    """A minimal application-owned progress callback."""
    print(event.phase, event.filename, event.bytes_done, event.bytes_total)


def make_config(*, custom_callback: bool = False) -> SynthesisConfig:
    reporter = progress if custom_callback else ConsoleAssetProgress()
    return SynthesisConfig(
        voice="af_sarah",
        generation=GenerationConfig(lang="en-us"),
        asset_progress=reporter,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--custom-callback", action="store_true")
    args = parser.parse_args(argv)
    with KokoroSynthesizer(make_config(custom_callback=args.custom_callback)) as synthesizer:
        result = synthesizer.synthesize_text("Tracking model asset progress.", language="en-us")
    output = artifact_path("asset_progress.wav")
    result.save_wav(output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
