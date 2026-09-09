#!/usr/bin/env python3
"""Inspect German Lexphon lexicon and provider fallback ownership."""

from __future__ import annotations

from pykokoro.tokenizer import Tokenizer, TokenizerConfig

CASES = (
    ("gold/none", ("gold",), "none"),
    ("gold/espeak", ("gold",), "espeak"),
    ("gold/goruut", ("gold",), "goruut"),
    ("olaph/none", ("olaph",), "none"),
    ("provider-only/espeak", (), "espeak"),
)


def main() -> None:
    for label, lexicons, fallback in CASES:
        config = TokenizerConfig(
            backend="kokorog2p",
            lexicons=lexicons,
            fallback=fallback,
            use_spacy=False,
        )
        tokenizer = Tokenizer(config=config)
        g2p = tokenizer._get_g2p("de")
        result = g2p.phonemize(
            "File",
            language="de",
            return_phonemes=True,
            return_ids=True,
            alignment="span",
        )

        print(f"[{label}]")
        print(f"selected lexicons: {lexicons}")
        print(f"fallback: {fallback}")
        print(f"final phonemes: {result.phonemes}")
        for token in result.tokens:
            metadata = getattr(token, "meta", {})
            print(
                "token: "
                f"{token.text!r}, "
                f"source={metadata.get('pronunciation_source')!r}, "
                f"lexicon={metadata.get('pronunciation_lexicon_id')!r}, "
                f"provider={metadata.get('pronunciation_provider')!r}, "
                f"language={metadata.get('pronunciation_requested_language')!r}"
            )
        print()


if __name__ == "__main__":
    main()
