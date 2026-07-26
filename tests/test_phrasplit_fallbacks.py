from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace

from pykokoro.pipeline_config import PipelineConfig
from pykokoro.stages.doc_parsers.plain import PhrasplitSentenceSplitter
from pykokoro.stages.protocols import DocumentResult
from pykokoro.types import Trace


@dataclass
class FakeSplitSegment:
    text: str
    start: int | None
    end: int | None
    paragraph: int | None = None
    sentence: int | None = None
    clause: int | None = None


def test_phrasplit_fallback_uses_cursor(monkeypatch):
    text = "Hello world. Hello world."
    first = FakeSplitSegment(text="Hello world.", start=0, end=12)
    second = FakeSplitSegment(text="Hello world.", start=None, end=None)
    fake_module = SimpleNamespace(split_with_offsets=lambda *_args, **_kwargs: [first, second])
    monkeypatch.setitem(sys.modules, "phrasplit", fake_module)

    splitter = PhrasplitSentenceSplitter()
    cfg = PipelineConfig()
    doc = DocumentResult(clean_text=text)
    trace = Trace()

    segments = splitter.split(doc, cfg, trace)

    assert [segment.text for segment in segments] == ["Hello world.", "Hello world."]
    assert [segment.char_start for segment in segments] == [0, 13]
    assert [segment.char_end for segment in segments] == [12, 25]


def test_phrasplit_fallback_clamps_invalid_offsets(monkeypatch):
    text = "Hello world."
    fake_segment = FakeSplitSegment(text="Hello world.", start=0, end=100)
    fake_module = SimpleNamespace(split_with_offsets=lambda *_args, **_kwargs: [fake_segment])
    monkeypatch.setitem(sys.modules, "phrasplit", fake_module)

    splitter = PhrasplitSentenceSplitter()
    cfg = PipelineConfig()
    doc = DocumentResult(clean_text=text)
    trace = Trace()

    segments = splitter.split(doc, cfg, trace)

    assert len(segments) == 1
    assert segments[0].text == "Hello world."
    assert segments[0].char_start == 0
    assert segments[0].char_end == len(text)
