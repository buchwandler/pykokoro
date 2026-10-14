from __future__ import annotations

import pytest
from kokorog2p import phonemes_to_ids

from pykokoro.prepared_g2p import PreparedG2PAdapter
from pykokoro.synthesis_config import SynthesisConfig
from pykokoro.synthesis_types import LinguisticToken, PronunciationOverride, SynthesisSegment
from pykokoro.synthesizer import KokoroSynthesizer


@pytest.mark.parametrize(
    "text",
    [
        "Wait, really?",
        "I can't believe it's already 2025.",
        "Dr. Watson met Misaki at 10:30.",
        'She said, "um, let us go," then left.',
        "The U.S. team arrived on time.",
        "Uh, I think we're ready.",
    ],
)
def test_reference_transcript_tokens_match_kokoro_text_cleaner(text: str) -> None:
    prepared = PreparedG2PAdapter().phonemize(
        SynthesisSegment("reference-parity", text, "en-us"), SynthesisConfig()
    )

    assert prepared.text == text
    assert prepared.token_ids == tuple(phonemes_to_ids(prepared.phonemes, model="1.0"))


def test_request_measure_uses_the_preparation_frontend_for_structured_inputs() -> None:

    request = SynthesisSegment(
        "structured-measure-parity",
        "Wait, really?",
        "en-us",
        pronunciation_overrides=(PronunciationOverride(0, 4, language="de"),),
        tokens=(LinguisticToken(0, 4, text="Wait", pos="VERB"),),
    )
    synthesizer = KokoroSynthesizer(g2p=PreparedG2PAdapter())

    prepared = synthesizer.prepare(request)
    measured = synthesizer.measure_request(request)

    assert measured.amount == len(prepared.token_ids)
