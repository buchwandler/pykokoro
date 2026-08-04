from __future__ import annotations

from types import SimpleNamespace

import pytest

from pykokoro import PipelineConfig
from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline import KokoroPipeline
from pykokoro.pipeline_config import PipelineConfig as PipelineConfigType
from pykokoro.spacy_models import (
    SpacyModelRequest,
    make_spacy_model_request,
)
from pykokoro.stages.audio_generation.noop import NoopAudioGenerationAdapter
from pykokoro.stages.audio_postprocessing.noop import NoopAudioPostprocessingAdapter
from pykokoro.stages.doc_parsers.plain import PhrasplitSentenceSplitter
from pykokoro.stages.doc_parsers.ssmd import SsmdDocumentParser
from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter
from pykokoro.stages.phoneme_processing.noop import NoopPhonemeProcessorAdapter
from pykokoro.stages.protocols import DocumentResult
from pykokoro.tokenizer import TokenizerConfig
from pykokoro.types import Segment, Trace


def test_default_request_is_highest_available_and_auto_is_unset():
    assert TokenizerConfig().spacy_model is None
    assert TokenizerConfig().spacy_model_size is None
    assert TokenizerConfig(spacy_model=" AuTo ").spacy_model is None
    assert make_spacy_model_request().mode == "highest_available"
    assert make_spacy_model_request(size="lg").mode == "size"
    assert make_spacy_model_request(model="en_core_web_sm").mode == "explicit"


def test_invalid_size_is_rejected():
    with pytest.raises(ValueError, match="spacy_model_size"):
        TokenizerConfig(spacy_model_size="xl")  # type: ignore[arg-type]


def test_explicit_model_wins_over_size_in_request():
    request = make_spacy_model_request(model="en_core_web_sm", size="lg")

    assert request == SpacyModelRequest(model="en_core_web_sm", size="lg")
    assert request.mode == "explicit"


def test_plain_parser_forwards_unset_request_and_records_selected_model(monkeypatch):
    captured: dict[str, object] = {}

    def fake_resolve(**kwargs: object) -> SimpleNamespace:
        captured["resolve"] = kwargs
        return SimpleNamespace(selected_model="en_core_web_lg", model_size="lg")

    def fake_split(text: str, **kwargs: object) -> list[SimpleNamespace]:
        captured["split"] = kwargs
        return [SimpleNamespace(text=text, start=0, end=len(text), paragraph=0, sentence=0)]

    fake_phrasplit = SimpleNamespace(
        resolve_spacy_model=fake_resolve,
        split_with_offsets=fake_split,
    )
    monkeypatch.setattr(
        "pykokoro.stages.doc_parsers.plain.importlib.import_module",
        lambda _name: fake_phrasplit,
    )

    doc = DocumentResult(clean_text="Hello.")
    segments = PhrasplitSentenceSplitter().split(doc, PipelineConfigType(), Trace())

    assert captured["split"] == {
        "mode": "sentence",
        "language_model": None,
        "language": "en",
        "model_size": None,
        "use_spacy": None,
        "apply_corrections": True,
    }
    assert doc.metadata["spacy_models"]["sentence"]["selected_model"] == "en_core_web_lg"
    assert segments[0].text == doc.clean_text[0:6]


def test_plain_parser_forwards_exact_model_and_size(monkeypatch):
    captured: dict[str, object] = {}

    def fake_split(text: str, **kwargs: object) -> list[SimpleNamespace]:
        captured.update(kwargs)
        return [SimpleNamespace(text=text, start=0, end=len(text), paragraph=0, sentence=0)]

    fake_phrasplit = SimpleNamespace(
        resolve_spacy_model=lambda **_kwargs: SimpleNamespace(
            selected_model="en_core_web_sm", model_size="sm"
        ),
        split_with_offsets=fake_split,
    )
    monkeypatch.setattr(
        "pykokoro.stages.doc_parsers.plain.importlib.import_module",
        lambda _name: fake_phrasplit,
    )
    cfg = PipelineConfigType(
        tokenizer_config=TokenizerConfig(
            spacy_model="en_core_web_sm",
            spacy_model_size="sm",
        )
    )

    PhrasplitSentenceSplitter().split(DocumentResult(clean_text="Hello."), cfg, Trace())

    assert captured["language_model"] == "en_core_web_sm"
    assert captured["model_size"] == "sm"


def test_plain_parser_keeps_explicit_spacy_requests_strict(monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_resolve(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        raise RuntimeError("model unavailable")

    fake_phrasplit = SimpleNamespace(
        resolve_spacy_model=fake_resolve,
        split_with_offsets=lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "pykokoro.stages.doc_parsers.plain.importlib.import_module",
        lambda _name: fake_phrasplit,
    )
    cfg = PipelineConfigType(
        tokenizer_config=TokenizerConfig(use_spacy=True, spacy_model="en_core_web_sm")
    )

    with pytest.raises(RuntimeError, match="model unavailable"):
        PhrasplitSentenceSplitter().split(DocumentResult(clean_text="Hello."), cfg, Trace())
    assert calls[0]["require"] is True


def test_plain_parser_auto_spacy_falls_back_without_a_local_model(monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_resolve(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        raise RuntimeError("no model installed")

    fake_phrasplit = SimpleNamespace(
        resolve_spacy_model=fake_resolve,
        split_with_offsets=lambda text, **_kwargs: [
            SimpleNamespace(text=text, start=0, end=len(text), paragraph=0, sentence=0)
        ],
    )
    monkeypatch.setattr(
        "pykokoro.stages.doc_parsers.plain.importlib.import_module",
        lambda _name: fake_phrasplit,
    )

    segments = PhrasplitSentenceSplitter().split(
        DocumentResult(clean_text="Hello."), PipelineConfigType(), Trace()
    )
    assert segments[0].text == "Hello."
    assert calls[0]["require"] is False


def test_ssmd_wrapper_preserves_exact_forwarding(monkeypatch):
    captured: dict[str, object] = {}

    def fake_body_parser(*args: object, **kwargs: object) -> tuple[float, list[object]]:
        captured.update(kwargs)
        return 0.0, []

    monkeypatch.setattr("pykokoro.ssmd_parser._parse_ssmd_body_to_segments", fake_body_parser)
    from pykokoro.ssmd_parser import parse_ssmd_document

    parse_ssmd_document(
        "Hello.",
        lang="en-us",
        spacy_model="en_core_web_sm",
        model_size="sm",
        use_spacy=True,
    )

    assert captured["spacy_model"] == "en_core_web_sm"
    assert captured["model_size"] == "sm"
    assert captured["use_spacy"] is True


def test_ssmd_stage_copies_sentence_diagnostics(monkeypatch):
    diagnostic = SimpleNamespace(selected_model="en_core_web_lg", selected_model_size="lg")
    parsed = SimpleNamespace(
        diagnostics=(),
        sentence_diagnostics=diagnostic,
        initial_pause=0.0,
        segments=(),
        pause_defaults=None,
        header={},
        body="Hello.",
    )
    monkeypatch.setattr(
        "pykokoro.stages.doc_parsers.ssmd.parse_ssmd_document",
        lambda *args, **kwargs: parsed,
    )

    doc = SsmdDocumentParser().parse("Hello.", PipelineConfigType(), Trace())

    assert doc.metadata["spacy_models"]["sentence"]["selected_model"] == "en_core_web_lg"


def test_g2p_forwards_request_and_records_concrete_selection():
    captured: list[dict[str, object]] = []
    adapter = KokoroG2PAdapter()
    adapter._load = lambda: SimpleNamespace(
        get_g2p=lambda **kwargs: (
            captured.append(kwargs) or SimpleNamespace(spacy_model="en_core_web_lg")
        )
    )
    cfg = PipelineConfigType(tokenizer_config=TokenizerConfig())
    instance = adapter._get_g2p_instance("en-us", cfg)
    doc = DocumentResult(clean_text="Hello.")
    adapter._record_selection(doc, "en-us", cfg, instance)

    assert captured[0]["spacy_model"] is None
    assert captured[0]["spacy_model_size"] is None
    assert doc.metadata["spacy_models"]["g2p"]["en-us"]["selected_model"] == "en_core_web_lg"


def test_g2p_cache_key_includes_model_request():
    calls: list[dict[str, object]] = []
    adapter = KokoroG2PAdapter()
    adapter._load = lambda: SimpleNamespace(
        get_g2p=lambda **kwargs: calls.append(kwargs) or SimpleNamespace(spacy_model=None)
    )

    adapter._get_g2p_instance("en-us", PipelineConfigType(tokenizer_config=TokenizerConfig()))
    adapter._get_g2p_instance("en-us", PipelineConfigType(tokenizer_config=TokenizerConfig()))
    adapter._get_g2p_instance(
        "en-us",
        PipelineConfigType(tokenizer_config=TokenizerConfig(spacy_model_size="sm")),
    )

    assert len(calls) == 2


def test_pipeline_result_retains_document_selection_metadata():
    class Parser:
        def parse(self, text: str, cfg: PipelineConfigType, trace: Trace) -> DocumentResult:
            return DocumentResult(
                clean_text=text,
                segments=[Segment("s0", text, 0, len(text), 0, 0, 0)],
                metadata={"spacy_models": {"sentence": {"selected_model": "en_core_web_lg"}}},
            )

    class G2P:
        def phonemize(self, segments, doc, cfg, trace):
            return []

    pipeline = KokoroPipeline(
        PipelineConfig(generation=GenerationConfig(lang="en-us")),
        doc_parser=Parser(),
        g2p=G2P(),
        phoneme_processing=NoopPhonemeProcessorAdapter(),
        audio_generation=NoopAudioGenerationAdapter(),
        audio_postprocessing=NoopAudioPostprocessingAdapter(),
    )

    result = pipeline.run("Hello.")

    assert result.document_metadata["spacy_models"]["sentence"]["selected_model"] == (
        "en_core_web_lg"
    )
