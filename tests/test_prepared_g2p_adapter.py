from types import SimpleNamespace

from kokorog2p import OverrideSpan, TokenAnnotation

from pykokoro.language_routing import LanguageRoutingConfig
from pykokoro.prepared_g2p import PreparedG2PAdapter
from pykokoro.synthesis_config import SynthesisConfig
from pykokoro.synthesis_types import LinguisticToken, PronunciationOverride, SynthesisSegment
from pykokoro.tokenizer import TokenizerConfig


class FakeKokoroG2P:
    OverrideSpan = OverrideSpan
    TokenAnnotation = TokenAnnotation

    def __init__(self):
        self.calls = []
        self.g2p_creations = []

    def get_g2p(self, **kwargs):
        instance = object()
        self.g2p_creations.append((kwargs, instance))
        return instance

    def phonemize_prepared(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return SimpleNamespace(
            phonemes="hˈɛloʊ wɜːld",
            token_ids=(1, 2, 3),
            tokens=(
                {
                    "text": "Hello",
                    "phonemes": "hˈɛloʊ",
                    "char_start": 0,
                    "char_end": 5,
                },
                {
                    "text": "world",
                    "phonemes": "wɜːld",
                    "char_start": 6,
                    "char_end": 11,
                },
            ),
            warnings=(),
        )

    def phonemes_to_ids(self, phonemes, model):
        return [ord(char) for char in phonemes]


def _config():
    return SynthesisConfig(
        language_routing=LanguageRoutingConfig(mode="auto", languages=("en-us", "de")),
        tokenizer_config=TokenizerConfig(lexicons=(), lexicon_data_policy="installed-only"),
    )


def test_prepared_adapter_forwards_request_text_overrides_annotations_and_routing():
    module = FakeKokoroG2P()
    adapter = PreparedG2PAdapter(module)
    request = SynthesisSegment(
        id="line-1",
        text="Hello world",
        language="en-us",
        pronunciation_overrides=(
            PronunciationOverride(0, 5, phonemes="hˈɛloʊ"),
            PronunciationOverride(6, 11, language="de"),
        ),
        tokens=(
            LinguisticToken(0, 5, text="Hello", pos="INTJ", lemma="hello", morph="Number=Sing"),
        ),
    )
    config = _config()

    result = adapter.phonemize(request, config)

    text, kwargs = module.calls[0]
    assert text == request.text
    assert kwargs["language"] == "en-us"
    assert kwargs["language_routing"] == {"mode": "auto", "languages": ("en-us", "de-de")}
    assert kwargs["overlap"] == "snap"
    assert kwargs["target_model"] == "1.0"
    assert isinstance(kwargs["overrides"][0], OverrideSpan)
    assert kwargs["overrides"][0].attrs == {"ph": "hˈɛloʊ"}
    assert kwargs["overrides"][1].attrs == {"lang": "de-de"}
    assert isinstance(kwargs["annotations"][0], TokenAnnotation)
    assert kwargs["annotations"][0].pos == "INTJ"
    assert kwargs["annotations"][0].lemma == "hello"
    assert kwargs["annotations"][0].morph == "Number=Sing"
    assert module.g2p_creations[0][0]["use_spacy"] is False
    assert kwargs["g2p"].__class__ is object
    assert result.request_id == request.id
    assert result.alignment_tokens[1].char_start == 6
    assert result.alignment_tokens[1].char_end == 11


def test_prepared_adapter_reuses_g2p_instance_for_repeated_request():
    module = FakeKokoroG2P()
    adapter = PreparedG2PAdapter(module)
    request = SynthesisSegment(id="first", text="Hello", language="en-us")

    adapter.phonemize(request, _config())
    adapter.phonemize(
        SynthesisSegment(id="second", text="Hello again", language="en-us"), _config()
    )

    assert len(module.g2p_creations) == 1
    assert module.calls[0][1]["g2p"] is module.calls[1][1]["g2p"]


def test_whole_request_phonemes_bypass_text_phonemization():
    module = FakeKokoroG2P()
    adapter = PreparedG2PAdapter(module)
    request = SynthesisSegment(id="phonemes", text="Hello", language="en-us", phonemes="hˈɛloʊ")

    result = adapter.phonemize(request, _config())

    assert module.calls == []
    assert result.phonemes == "hˈɛloʊ"
    assert result.token_ids == tuple(ord(char) for char in "hˈɛloʊ")


def test_context_phonemization_uses_the_cached_prepared_g2p_instance():
    module = FakeKokoroG2P()
    adapter = PreparedG2PAdapter(module)

    result = adapter.phonemize_context("Wait, {segment}.", "en-us", _config())

    text, kwargs = module.calls[0]
    assert text == "Wait, {segment}."
    assert kwargs["alignment"] == "span"
    assert kwargs["target_model"] == "1.0"
    assert kwargs["overrides"] is None
    assert kwargs["annotations"] is None
    assert result.phonemes == "hˈɛloʊ wɜːld"
    assert len(module.g2p_creations) == 1
