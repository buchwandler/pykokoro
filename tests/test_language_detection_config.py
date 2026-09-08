from __future__ import annotations

import pytest

from pykokoro import LanguageDetectionConfig
from pykokoro.generation_config import GenerationConfig
from pykokoro.language_detection import resolve_language_detection
from pykokoro.pipeline import _coerce_pipeline_config, _merge_config
from pykokoro.pipeline_config import PipelineConfig


def test_language_detection_config_normalizes_candidates() -> None:
    config = LanguageDetectionConfig(mode="auto", languages=("de", "en", "en-us"))
    assert config.languages == ("de-de", "en-us")


def test_language_detection_config_requires_two_auto_candidates() -> None:
    with pytest.raises(ValueError, match="at least two"):
        LanguageDetectionConfig(mode="auto", languages=("de",))


def test_pipeline_mapping_coercion_and_explicit_off() -> None:
    config = _coerce_pipeline_config(
        {
            "language_detection": {"mode": "auto", "languages": ["de", "en"]},
        }
    )
    assert config.language_detection == LanguageDetectionConfig(
        mode="auto", languages=("de-de", "en-us")
    )
    assert PipelineConfig().language_detection is None
    disabled = _merge_config(config, {"language_detection": {"mode": "off"}})
    assert disabled.language_detection == LanguageDetectionConfig()


def test_language_detection_precedence_is_api_then_header_then_default() -> None:
    header = {"language_detection": {"mode": "auto", "languages": ["de", "en"]}}
    from_header = resolve_language_detection(None, header)
    assert from_header.source == "header"
    assert from_header.as_routing() == {"mode": "auto", "languages": ("de-de", "en-us")}

    from_api = resolve_language_detection(LanguageDetectionConfig(mode="off"), header)
    assert from_api.source == "pipeline"
    assert from_api.as_routing() is None
    assert resolve_language_detection(None, {}).source == "default"


def test_adapter_forwards_routing_and_split_overrides(monkeypatch) -> None:
    from types import SimpleNamespace

    from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter
    from pykokoro.stages.protocols import DocumentResult
    from pykokoro.types import AnnotationSpan, Segment, Trace

    calls = []

    class FakeModule:
        __version__ = "test"

        @staticmethod
        def phonemize_prepared(*args, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                phonemes="a", ids=[1], tokens=[], warnings=[], language_routes=[]
            )

        @staticmethod
        def phonemes_to_ids(phonemes, model=None):
            return [len(phonemes), model]

    text = "gecanceltt"
    segment = Segment("seg", text, 0, len(text), {"language": "de"})
    doc = DocumentResult(
        clean_text=text,
        annotation_spans=[AnnotationSpan(2, 8, {"language": "en", "scope": "pronunciation"})],
    )
    config = PipelineConfig(
        generation=GenerationConfig(lang="de"),
        language_detection=LanguageDetectionConfig(mode="auto", languages=("de", "en")),
    )
    adapter = KokoroG2PAdapter()
    monkeypatch.setattr(adapter, "_load", lambda: FakeModule())
    monkeypatch.setattr(adapter, "_get_g2p_instance", lambda lang, cfg: object())

    adapter.phonemize([segment], doc, config, Trace())

    assert calls[0]["overlap"] == "split"
    assert calls[0]["target_model"] == "1.0"
    assert calls[0]["language_routing"] == {"mode": "auto", "languages": ("de-de", "en-us")}
    override = calls[0]["overrides"][0]
    assert (override.char_start, override.char_end) == (2, 8)
    assert override.attrs == {"lang": "en"}


def test_routing_policy_is_in_g2p_key_but_not_acoustic_key() -> None:
    from pykokoro.pipeline import KokoroPipeline
    from pykokoro.runtime.cache import make_g2p_key

    base = {
        "text": "Hello",
        "lang": "de",
        "is_phonemes": False,
        "tokenizer_config": None,
        "phoneme_override": None,
    }
    assert make_g2p_key(
        **base, language_routing={"mode": "auto", "languages": ("de", "en")}
    ) != make_g2p_key(**base, language_routing=None)

    first = PipelineConfig(
        generation=GenerationConfig(lang="de"),
        language_detection=LanguageDetectionConfig(mode="auto", languages=("de", "en")),
    )
    second = PipelineConfig(
        generation=GenerationConfig(lang="de"),
        language_detection=LanguageDetectionConfig(mode="auto", languages=("de", "fr")),
    )
    assert KokoroPipeline(first)._kokoro_key(first) == KokoroPipeline(second)._kokoro_key(second)


def test_phoneme_input_does_not_resolve_foreign_g2p(monkeypatch) -> None:

    from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter
    from pykokoro.stages.protocols import DocumentResult
    from pykokoro.types import Segment, Trace

    class FakeModule:
        __version__ = "test"

        @staticmethod
        def phonemes_to_ids(phonemes, model=None):
            return [len(phonemes)]

    text = "a b"
    segment = Segment("seg", text, 0, len(text), {"language": "de"})
    doc = DocumentResult(clean_text=text)
    config = PipelineConfig(
        generation=GenerationConfig(lang="de", is_phonemes=True),
        language_detection=LanguageDetectionConfig(mode="auto", languages=("de", "en")),
    )
    adapter = KokoroG2PAdapter()
    monkeypatch.setattr(adapter, "_load", lambda: FakeModule())
    monkeypatch.setattr(
        adapter,
        "_get_g2p_instance",
        lambda lang, cfg: (_ for _ in ()).throw(AssertionError("foreign G2P was resolved")),
    )

    result = adapter.phonemize([segment], doc, config, Trace())
    assert result[0].phonemes == text
    assert doc.metadata["language_detection"]["mode"] == "auto"
