import numpy as np
import pytest

from pykokoro.error_diagnostics import error_diagnostic
from pykokoro.exceptions import AlignmentError, BackendError, InvalidVoiceError, SynthesisStateError
from pykokoro.generation_config import GenerationConfig
from pykokoro.prepared_g2p import PreparedSynthesis
from pykokoro.synthesis_config import SynthesisConfig
from pykokoro.synthesis_types import RenderedSegment, SynthesisSegment
from pykokoro.synthesizer import KokoroSynthesizer


class FakeG2P:
    def __init__(self):
        self.requests = []
        self.configs = []

    def phonemize(self, segment, config):
        self.requests.append(segment)
        self.configs.append(config)
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

    with pytest.raises(AlignmentError, match="ID must match"):
        synthesizer.synthesize(SynthesisSegment("expected", "Hello", "en-us"))


def test_closed_synthesizer_raises_typed_state_error():
    synthesizer = KokoroSynthesizer(g2p=FakeG2P(), renderer=FakeRenderer())
    synthesizer.close()

    with pytest.raises(SynthesisStateError, match="closed"):
        synthesizer.synthesize(SynthesisSegment("line", "Hello", "en-us"))


def test_backend_failure_is_wrapped_as_typed_error() -> None:
    class BrokenRenderer(FakeRenderer):
        def render(self, prepared, segment, config):
            raise RuntimeError("inference failed")

    synthesizer = KokoroSynthesizer(g2p=FakeG2P(), renderer=BrokenRenderer())

    with pytest.raises(BackendError, match="Kokoro synthesis failed") as raised:
        synthesizer.synthesize(SynthesisSegment("line", "Hello", "en-us"))

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert raised.value.stage == "synthesis"
    assert raised.value.native_error_type == "RuntimeError"
    assert raised.value.native_error_message == "inference failed"
    assert "RuntimeError: inference failed" in str(raised.value)
    diagnostic = error_diagnostic(raised.value)
    assert diagnostic["cause"]["message"] == "inference failed"  # type: ignore[index]


def test_frontend_failure_is_structured_without_request_text() -> None:
    class BrokenG2P:
        def phonemize(self, segment, config):
            raise RuntimeError("phonemizer unavailable")

    synthesizer = KokoroSynthesizer(g2p=BrokenG2P(), renderer=FakeRenderer())

    with pytest.raises(BackendError, match="RuntimeError: phonemizer unavailable") as raised:
        synthesizer.prepare(SynthesisSegment("request-1", "secret request text", "en-us"))

    assert raised.value.stage == "frontend"
    assert raised.value.native_error_type == "RuntimeError"
    assert raised.value.native_error_message == "phonemizer unavailable"
    diagnostic = error_diagnostic(raised.value)
    assert diagnostic["details"]["request_id"] == "request-1"  # type: ignore[index]
    assert "secret request text" not in str(diagnostic)


class NoInferenceRenderer:
    def render(self, *args, **kwargs):
        raise AssertionError("measurement must not render audio")

    def enroll_voice(self, *args, **kwargs):
        raise AssertionError("measurement must not enroll a reference voice")

    def enroll_inno_voice(self, *args, **kwargs):
        raise AssertionError("measurement must not enroll an Inno voice")


def test_measure_request_matches_prepare_for_static_request() -> None:
    g2p = FakeG2P()
    synthesizer = KokoroSynthesizer(g2p=g2p, renderer=NoInferenceRenderer())
    request = SynthesisSegment("measure", "Hello world", "en-us", voice="af_sarah")

    prepared = synthesizer.prepare(request)
    measured = synthesizer.measure_request(request, maximum=2)

    assert measured.amount == len(prepared.token_ids) == 2
    assert measured.fits is True
    assert g2p.requests == [request, request]


def test_measure_request_allows_unenrolled_reference_target_only_for_measurement() -> None:
    g2p = FakeG2P()
    synthesizer = KokoroSynthesizer(
        SynthesisConfig(model_source="github", model_variant="en-akinvox-cloning-v1"),
        g2p=g2p,
        renderer=NoInferenceRenderer(),
    )
    request = SynthesisSegment("reference-measure", "Hello world", "en-us")

    measured = synthesizer.measure_request(request, maximum=1)

    assert measured.amount == 2
    assert measured.fits is False
    assert g2p.configs[-1].model_variant == "en-akinvox-cloning-v1"
    assert len(g2p.requests) == 1
    with pytest.raises(InvalidVoiceError, match="requires a ReferenceVoice"):
        synthesizer.prepare(request)
    with pytest.raises(InvalidVoiceError, match="requires a ReferenceVoice"):
        synthesizer.synthesize(request)
    assert len(g2p.requests) == 1


def test_measure_request_passes_structured_frontend_input_through_prepare_path() -> None:
    from pykokoro.synthesis_types import LinguisticToken, PronunciationOverride

    g2p = FakeG2P()
    synthesizer = KokoroSynthesizer(g2p=g2p, renderer=NoInferenceRenderer())
    request = SynthesisSegment(
        "structured-measure",
        "Hello world",
        "en-us",
        pronunciation_overrides=(PronunciationOverride(0, 5, language="de"),),
        tokens=(LinguisticToken(0, 5, text="Hello", pos="INTJ"),),
    )

    prepared = synthesizer.prepare(request)
    measured = synthesizer.measure_request(request)

    assert measured.amount == len(prepared.token_ids)
    assert g2p.requests == [request, request]
    assert g2p.requests[-1].pronunciation_overrides == request.pronunciation_overrides
    assert g2p.requests[-1].tokens == request.tokens
