from __future__ import annotations

import sys
from types import SimpleNamespace

from pykokoro.pipeline_config import PipelineConfig
from pykokoro.stages.doc_parsers.plain import PhrasplitSentenceSplitter
from pykokoro.stages.protocols import DocumentResult
from pykokoro.types import AnnotationSpan, Trace


def test_phrasplit_splits_on_phoneme_override(monkeypatch):
    text = "Hello world"
    doc = DocumentResult(
        clean_text=text,
        annotation_spans=[AnnotationSpan(char_start=0, char_end=5, attrs={"ph": "OVERRIDE"})],
    )
    fake_module = SimpleNamespace(split_with_offsets=lambda *_args, **_kwargs: [])
    monkeypatch.setitem(sys.modules, "phrasplit", fake_module)

    splitter = PhrasplitSentenceSplitter()
    segments = splitter.split(doc, PipelineConfig(), Trace())

    assert [(seg.char_start, seg.char_end) for seg in segments] == [(0, 5), (5, 11)]
    assert segments[0].text == "Hello"
