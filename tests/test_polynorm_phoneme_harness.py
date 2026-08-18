from __future__ import annotations

import pytest

from benchmarks.polynorm_eval import PyKokoroPhonemeHarness


@pytest.mark.parametrize(
    ("original_text", "normalized_text"),
    [
        ("2", "two"),
        ("42 kg", "forty two kilograms"),
        ("Dr. Smith", "Doctor Smith"),
        ("$5", "five dollars"),
    ],
)
def test_plain_harness_matches_spokenform_equivalent_pairs(
    original_text: str,
    normalized_text: str,
) -> None:
    with PyKokoroPhonemeHarness("en-us", "kokorog2p") as harness:
        original = harness.phonemize(original_text)
        normalized = harness.phonemize(normalized_text)

    assert original.phonemes == normalized.phonemes
    assert original.tokens == normalized.tokens
    assert original.warnings == ()
    assert normalized.warnings == ()


def test_ssmd_harness_keeps_equivalent_output_without_onnx() -> None:
    with PyKokoroPhonemeHarness("en-us", "kokorog2p", ssmd=True) as harness:
        original = harness.phonemize("Meet Dr. Smith at 5:30.")
        normalized = harness.phonemize("Meet Doctor Smith at five thirty.")

    assert original.phonemes == normalized.phonemes
    assert original.tokens == normalized.tokens
