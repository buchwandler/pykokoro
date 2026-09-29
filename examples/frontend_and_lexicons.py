#!/usr/bin/env python3
"""Inspect lexicon metadata and build an explicit offline frontend policy.

Discovery does not provision lexicon or model assets. The returned configuration
uses only installed lexicon data; actual synthesis is left to the caller.
"""

from __future__ import annotations

from pykokoro import (
    GenerationConfig,
    SynthesisConfig,
    TokenizerConfig,
    discover_lexicons,
)


def make_tokenizer_config() -> TokenizerConfig:
    return TokenizerConfig(
        backend="kokorog2p",
        fallback="espeak",
        lexicons=("gold",),
        lexicon_data_policy="installed-only",
        use_spacy=False,
    )


def make_synthesis_config() -> SynthesisConfig:
    return SynthesisConfig(
        generation=GenerationConfig(lang="de"),
        tokenizer_config=make_tokenizer_config(),
    )


def main() -> None:
    inventory = discover_lexicons(language="de", offline=True)
    print("Metadata discovery only; no lexicon data or model assets are installed.")
    for item in inventory.lexicons:
        print(item.selector, item.asset_id, item.installed, item.model_support)
    config = make_synthesis_config()
    print(f"Frontend configuration: {config.tokenizer_config}")
    print("Use this configuration with synthesis after the selected lexicon is installed.")


if __name__ == "__main__":
    main()
