import sys
import types

import pytest

from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.stages.doc_parsers.plain import PlainTextDocumentParser
from pykokoro.stages.doc_parsers.ssmd import SsmdDocumentParser
from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter
from pykokoro.stages.protocols import DocumentResult
from pykokoro.types import Segment, Trace


def test_plain_parser_auto_sentence_boundaries(monkeypatch):
    text = "Hello world. Another sentence."

    class FakeSegment:
        def __init__(self, text: str, start: int, end: int, sentence_idx: int) -> None:
            self.text = text
            self.start = start
            self.end = end
            self.sentence = sentence_idx
            self.paragraph = 0

    first_end = len("Hello world.")
    second_start = text.index("Another")
    first = FakeSegment("Hello world.", 0, first_end, 0)
    second = FakeSegment("Another sentence.", second_start, len(text), 1)

    dummy_module = types.SimpleNamespace(
        split_with_offsets=lambda *_args, **_kwargs: [first, second]
    )
    monkeypatch.setitem(sys.modules, "phrasplit", dummy_module)

    cfg = PipelineConfig(generation=GenerationConfig(pause_mode="auto"))
    doc = PlainTextDocumentParser().parse(text, cfg, Trace())

    expected_pos = first_end - 1
    sentence_boundaries = [
        boundary
        for boundary in doc.boundary_events
        if boundary.kind == "pause" and boundary.attrs.get("strength") == "s"
    ]
    assert any(boundary.pos == expected_pos for boundary in sentence_boundaries)


def test_ssmd_parser_auto_sentence_boundaries():
    text = "Hello world. Another sentence."
    cfg = PipelineConfig(generation=GenerationConfig(pause_mode="auto"))
    doc = SsmdDocumentParser().parse(text, cfg, Trace())

    sentence_indices = [
        segment.sentence_idx
        for segment in doc.segments
        if segment.sentence_idx is not None
    ]
    assert sentence_indices
    first_sentence = min(sentence_indices)
    first_end = max(
        segment.char_end
        for segment in doc.segments
        if segment.sentence_idx == first_sentence
    )
    expected_pos = max(0, first_end - 1)
    sentence_boundaries = [
        boundary
        for boundary in doc.boundary_events
        if boundary.kind == "pause" and boundary.attrs.get("strength") == "s"
    ]
    assert any(boundary.pos == expected_pos for boundary in sentence_boundaries)


def test_auto_clause_pause_in_g2p(monkeypatch):
    fake_g2p = types.SimpleNamespace(__version__="0.0")

    def phonemes_to_ids(phonemes, model=None):
        _ = model
        return list(range(len(phonemes)))

    def ids_to_phonemes(ids, model=None):
        _ = model
        return "a" * len(ids)

    fake_g2p.phonemes_to_ids = phonemes_to_ids
    fake_g2p.ids_to_phonemes = ids_to_phonemes
    monkeypatch.setitem(sys.modules, "kokorog2p", fake_g2p)

    text = "a" * 300 + "," + "b" * 300 + "," + "c" * 300
    generation = GenerationConfig(
        is_phonemes=True, pause_mode="auto", pause_clause=0.25
    )
    cfg = PipelineConfig(generation=generation)
    doc = DocumentResult(clean_text=text)
    segments = [
        Segment(
            id="seg_0",
            text=text,
            char_start=0,
            char_end=len(text),
            paragraph_idx=0,
            sentence_idx=0,
            clause_idx=0,
        )
    ]

    phoneme_segments = KokoroG2PAdapter().phonemize(segments, doc, cfg, Trace())

    assert len(phoneme_segments) == 3
    assert phoneme_segments[0].pause_after == pytest.approx(generation.pause_clause)
    assert phoneme_segments[1].pause_after == pytest.approx(generation.pause_clause)
    assert phoneme_segments[2].pause_after == pytest.approx(0.0)
