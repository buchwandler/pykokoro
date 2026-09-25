from __future__ import annotations

import pytest

pytest.importorskip("benchmarks.polynorm_eval")

from benchmarks.polynorm_eval import PyKokoroPhonemeHarness


def test_request_harness_phonemizes_plain_text_without_document_planning() -> None:
    text = "Meet Dr. Smith at 5:30."
    with PyKokoroPhonemeHarness("en-us", "kokorog2p") as harness:
        first = harness.phonemize(text)
        second = harness.phonemize(text)

    assert first.phonemes
    assert first.tokens
    assert first.segment_count == 1
    assert first.phonemes == second.phonemes
    assert first.tokens == second.tokens
    assert first.warnings == second.warnings
