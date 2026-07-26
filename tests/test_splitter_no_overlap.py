from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType

from pykokoro.debug.segment_invariants import check_segment_invariants
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


class patch_sys_modules:
    def __init__(self, updates: dict[str, ModuleType]) -> None:
        self._updates = updates
        self._originals: dict[str, ModuleType | None] = {}

    def __enter__(self):
        for key, value in self._updates.items():
            self._originals[key] = sys.modules.get(key)
            sys.modules[key] = value
        return self

    def __exit__(self, exc_type, exc, tb):
        for key, original in self._originals.items():
            if original is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = original
        return False


def _run_split_with_segments(text: str, segments: list[FakeSplitSegment]):
    fake_module = ModuleType("phrasplit")
    fake_module.split_with_offsets = lambda *_args, **_kwargs: segments
    splitter = PhrasplitSentenceSplitter()
    cfg = PipelineConfig()
    doc = DocumentResult(clean_text=text)
    trace = Trace()

    with patch_sys_modules({"phrasplit": fake_module}):
        return splitter.split(doc, cfg, trace)


def test_splitter_invariants_with_valid_offsets():
    text = "Hello world. Hello again."
    first = FakeSplitSegment(text="Hello world.", start=0, end=12)
    assert first.end is not None
    second_start = text.find("Hello again.", first.end + 1)
    second = FakeSplitSegment(text="Hello again.", start=second_start, end=second_start + 12)

    segments = _run_split_with_segments(text, [first, second])
    result = check_segment_invariants(segments, text, report_fn=None)

    assert result.ok
    assert [segment.text for segment in segments] == ["Hello world.", "Hello again."]


def test_splitter_clamps_overlapping_offsets():
    text = "Hello world. Hello world."
    first = FakeSplitSegment(text="Hello world.", start=0, end=12)
    second = FakeSplitSegment(text="Hello world.", start=5, end=17)

    segments = _run_split_with_segments(text, [first, second])
    result = check_segment_invariants(segments, text, report_fn=None)

    assert result.ok
    assert [segment.char_start for segment in segments] == [0, 13]
    assert [segment.char_end for segment in segments] == [12, 25]


def test_splitter_recovers_mismatched_offsets():
    text = "Wait... what? Wait... what?"
    first = FakeSplitSegment(text="Wait... what?", start=1, end=14)
    second = FakeSplitSegment(text="Wait... what?", start=10, end=23)

    segments = _run_split_with_segments(text, [first, second])
    result = check_segment_invariants(segments, text, report_fn=None)

    expected_second_start = text.find("Wait... what?", 1)
    assert result.ok
    assert segments[0].char_start == 0
    assert segments[0].char_end == len("Wait... what?")
    assert segments[1].char_start == expected_second_start
    assert segments[1].char_end == expected_second_start + len("Wait... what?")
