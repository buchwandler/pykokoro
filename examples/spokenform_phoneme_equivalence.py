from __future__ import annotations

from benchmarks.polynorm_eval import PyKokoroPhonemeHarness, semantic_phoneme_key

CASES = [
    ("I paid $12.50.", "I paid twelve dollars and fifty cents."),
    ("Meet Dr. Smith at 5:30.", "Meet Doctor Smith at five thirty."),
    ("The file is 42 MB.", "The file is forty two megabytes."),
]


def main() -> None:
    with PyKokoroPhonemeHarness("en-us", "kokorog2p") as harness:
        for original_text, normalized_text in CASES:
            original = harness.phonemize(original_text)
            normalized = harness.phonemize(normalized_text)
            semantic_match = semantic_phoneme_key(original.phonemes) == semantic_phoneme_key(
                normalized.phonemes
            )
            print(f"original:   {original_text}")
            print(f"expected:   {normalized_text}")
            print(f"phonemes:   {original.phonemes}")
            print(f"expected φ: {normalized.phonemes}")
            print(f"semantic match: {semantic_match}")
            print()


if __name__ == "__main__":
    main()
