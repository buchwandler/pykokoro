import hashlib
from pathlib import Path
from types import SimpleNamespace

import g2lex
import pytest
from kokorog2p import OverrideSpan, TokenAnnotation
from kokorog2p.lexicons.registry import FileLexiconSource

from pykokoro import G2LexFileRef
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


def test_prepared_adapter_forwards_local_g2lex_without_named_installation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installations = []
    monkeypatch.setattr(
        "pykokoro.lexicon_data.install_missing_lexphon_data",
        lambda language, lexicons: installations.append((language, lexicons)) or (),
    )
    ref = G2LexFileRef("custom/local.g2lex", sha256="a" * 64)
    config = SynthesisConfig(
        tokenizer_config=TokenizerConfig(
            lexicons=(),
            g2lex_files=(ref,),
            resolve_heteronyms=False,
        )
    )
    module = FakeKokoroG2P()
    adapter = PreparedG2PAdapter(module)

    adapter.phonemize(
        SynthesisSegment(id="local-source", text="Hello", language="en-us"),
        config,
    )

    kwargs = module.g2p_creations[0][0]
    assert kwargs["lexicons"] == ()
    assert kwargs["g2lex_files"] == (
        FileLexiconSource(
            path=ref.path,
            sha256=ref.sha256,
            logical_sha256=ref.logical_sha256,
        ),
    )
    assert kwargs["resolve_heteronyms"] is False
    assert installations == [("en-us", ())]


def test_alignment_preserves_contextual_heteronym_provenance() -> None:
    decision = {
        "source": "english-context-resolver",
        "context_rule": "lead.metal",
        "ipa": "lɛd",
        "reason": "A metal-object complement selects the element pronunciation.",
        "lexicon_id": "en-us:gold",
    }
    token = SimpleNamespace(
        text="lead",
        phonemes="lɛd",
        whitespace="",
        char_start=0,
        char_end=4,
        meta={
            "pronunciation_source": "contextual-heteronym",
            "heteronym_resolution": decision,
        },
    )

    alignment = PreparedG2PAdapter._alignment_token(token, FakeKokoroG2P(), "1.0")

    assert alignment.pronunciation_source == "contextual-heteronym"
    assert alignment.pronunciation_provider == "english-context-resolver"
    assert alignment.pronunciation_lexicon_id == "en-us:gold"
    assert alignment.pronunciation_source_ipa == "lɛd"
    assert alignment.heteronym_resolution == decision
    assert alignment.to_dict()["heteronym_resolution"] == decision


def test_alignment_preserves_local_g2lex_source_details() -> None:
    details = {
        "source_kind": "file",
        "selector": "editorial",
        "lexicon_id": "en-us:editorial",
        "source_path": "/tmp/editorial.g2lex",
        "sha256": "sha256:" + "a" * 64,
        "logical_sha256": "sha256:" + "b" * 64,
        "phoneme_encoding": "ipa",
    }
    token = SimpleNamespace(
        text="lead",
        phonemes="lɛd",
        whitespace="",
        char_start=0,
        char_end=4,
        meta={
            "pronunciation_source": "local-g2lex",
            "pronunciation_provider": "file",
            "pronunciation_lexicon_id": "en-us:editorial",
            "pronunciation_source_details": details,
        },
    )

    alignment = PreparedG2PAdapter._alignment_token(token, FakeKokoroG2P(), "1.0")

    assert alignment.pronunciation_source == "local-g2lex"
    assert alignment.pronunciation_provider == "file"
    assert alignment.pronunciation_lexicon_id == "en-us:editorial"
    assert alignment.pronunciation_source_details == details
    assert alignment.to_dict()["pronunciation_source_details"] == details


def test_local_g2lex_provenance_reaches_prepared_alignment(tmp_path: Path) -> None:
    source_json = tmp_path / "editorial.json"
    source_path = tmp_path / "editorial.g2lex"
    source_json.write_text(
        '{"read": {"DEFAULT": "riːd", "VERB": "riːd", "VBN": "rɛd"}}',
        encoding="utf-8",
    )
    g2lex.pack_file(
        source_json,
        source_path,
        input_format="kokoro-json",
        metadata={
            "id": "fixture:editorial",
            "name": "editorial",
            "language": "en-us",
            "phoneme_encoding": "ipa",
        },
    )
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    config = SynthesisConfig(
        tokenizer_config=TokenizerConfig(
            lexicons=(),
            g2lex_files=(G2LexFileRef(str(source_path), sha256=digest),),
            use_spacy=False,
            resolve_heteronyms=False,
        )
    )
    request = SynthesisSegment(
        id="local-g2lex",
        text="had read",
        language="en-us",
        tokens=(
            LinguisticToken(
                4,
                8,
                text="read",
                pos="VERB",
                tag="VBN",
                lemma="read",
                morph="VerbForm=Part",
            ),
        ),
    )

    result = PreparedG2PAdapter().phonemize(request, config)
    alignment = next(item for item in result.alignment_tokens if item.text == "read")

    assert alignment.phonemes == "ɹɛd"
    assert alignment.pronunciation_source == "local-g2lex"
    assert alignment.pronunciation_provider == "file"
    assert alignment.pronunciation_source_details is not None
    assert alignment.pronunciation_source_details["source_path"] == str(source_path.resolve())
    assert alignment.pronunciation_source_details["sha256"] == f"sha256:{digest}"


def test_prepared_adapter_preserves_tagged_live_annotation():
    module = FakeKokoroG2P()
    adapter = PreparedG2PAdapter(module)
    request = SynthesisSegment(
        id="live-verb",
        text="I live here.",
        language="en-us",
        tokens=(
            LinguisticToken(
                2,
                6,
                text="live",
                pos="VERB",
                tag="VBP",
                lemma="live",
                morph="Tense=Pres|VerbForm=Fin",
            ),
        ),
    )

    adapter.phonemize(request, _config())

    annotation = module.calls[0][1]["annotations"][0]
    assert isinstance(annotation, TokenAnnotation)
    assert (annotation.start, annotation.end, annotation.text) == (2, 6, "live")
    assert (annotation.pos, annotation.tag, annotation.lemma, annotation.morph) == (
        "VERB",
        "VBP",
        "live",
        "Tense=Pres|VerbForm=Fin",
    )


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
