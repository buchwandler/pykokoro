import numpy as np
import pytest

from pykokoro.generation_config import GenerationConfig
from pykokoro.prepared_g2p import PreparedSynthesis
from pykokoro.synthesis_config import SynthesisConfig
from pykokoro.synthesis_types import RenderedSegment, SynthesisSegment
from pykokoro.synthesizer import KokoroSynthesizer


class FakeG2P:
    def __init__(self):
        self.requests = []

    def phonemize(self, segment, config):
        self.requests.append(segment)
        return PreparedSynthesis(
            request_id=segment.id,
            text=segment.text,
            language=segment.language,
            voice=segment.voice if isinstance(segment.voice, str) else None,
            phonemes="test",
            token_ids=(1, 2),
        )


class FakeRenderer:
    def __init__(self):
        self.requests = []

    def render(self, prepared, segment, config):
        self.requests.append(segment.id)
        return RenderedSegment(
            id=segment.id,
            audio=np.array([0.1, 0.2], dtype=np.float32),
            sample_rate=24000,
            text=segment.text,
            language=segment.language,
            voice=segment.voice if isinstance(segment.voice, str) else config.voice,
            phonemes=prepared.phonemes,
            token_ids=prepared.token_ids,
        )


def test_synthesize_text_sends_literal_prepared_text_without_document_parsing():
    g2p = FakeG2P()
    renderer = FakeRenderer()
    synthesizer = KokoroSynthesizer(
        SynthesisConfig(generation=GenerationConfig(lang="en-us")),
        g2p=g2p,
        renderer=renderer,
    )
    literal = '[Hello]{voice="guest"}'

    result = synthesizer.synthesize_text(literal, voice="af_sarah")

    assert g2p.requests[0].text == literal
    assert result.text == literal
    assert result.id.startswith("request-")
    assert result.voice == "af_sarah"


def test_synthesize_segments_preserves_order_and_returns_independent_results():
    renderer = FakeRenderer()
    synthesizer = KokoroSynthesizer(g2p=FakeG2P(), renderer=renderer)
    requests = (
        SynthesisSegment("a", "First.", "en-us", voice="af_sarah"),
        SynthesisSegment("b", "Second.", "en-us", voice="af_bella"),
    )

    results = list(synthesizer.synthesize_segments(requests))

    assert [result.id for result in results] == ["a", "b"]
    assert [result.voice for result in results] == ["af_sarah", "af_bella"]
    assert [len(result.audio) for result in results] == [2, 2]
    assert renderer.requests == ["a", "b"]


def test_synthesize_rejects_renderer_result_with_wrong_request_id():
    class WrongIdRenderer(FakeRenderer):
        def render(self, prepared, segment, config):
            result = super().render(prepared, segment, config)
            result.id = "not-the-request-id"
            return result

    synthesizer = KokoroSynthesizer(g2p=FakeG2P(), renderer=WrongIdRenderer())

    with pytest.raises(ValueError, match="ID must match"):
        synthesizer.synthesize(SynthesisSegment("expected", "Hello", "en-us"))
