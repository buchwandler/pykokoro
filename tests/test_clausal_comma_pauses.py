from __future__ import annotations

import sys
import types

import pytest
from pykokoro.stages.segmentation.phrasplit import PhrasplitSentenceSegmenter

from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.runtime.language_plan import LanguageRun
from pykokoro.runtime.linguistics import LinguisticRequestState, PreparedRunAnalysis
from pykokoro.runtime.spans import slice_boundaries
from pykokoro.stages.protocols import DocumentResult
from pykokoro.types import BoundaryEvent, Trace

TEXT = "It had picked up the sound of a explosion, direction suggested it was behind."
COMMA = TEXT.index(",")


def _boundary() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        text=",",
        char_start=COMMA,
        char_end=COMMA + 1,
        kind="clausal_comma",
        meta={},
    )


def _run_case(monkeypatch, generation: GenerationConfig, detected=None, *, existing=None):
    calls: list[tuple[str, str, object]] = []
    detector_calls = 0

    def split_with_offsets(text: str, *args, **kwargs):
        _ = args, kwargs
        return [(text, 0, len(text), 0, 0, 0)]

    def detect_clause_boundaries(text: str, *, language: str, doc: object):
        nonlocal detector_calls
        detector_calls += 1
        calls.append((text, language, doc))
        return [] if detected is None else detected

    fake_module = types.SimpleNamespace(
        split_with_offsets=split_with_offsets,
        detect_clause_boundaries=detect_clause_boundaries,
    )
    monkeypatch.setitem(sys.modules, "phrasplit", fake_module)

    run = LanguageRun(char_start=0, char_end=len(TEXT), language="en-us")
    analysis_doc = object()
    analysis = PreparedRunAnalysis(run=run, text=TEXT, doc=analysis_doc, annotations=())
    state = LinguisticRequestState(prepared_plan=(run,), prepared_analysis=[analysis])
    doc = DocumentResult(clean_text=TEXT, linguistic_state=state)
    if existing:
        doc.boundary_events.extend(existing)
    trace = Trace()

    segments = PhrasplitSentenceSegmenter().split(doc, PipelineConfig(generation=generation), trace)
    return doc, trace, segments, detector_calls, calls, analysis_doc


def test_auto_clausal_comma_creates_owned_pause_and_cut(monkeypatch):
    generation = GenerationConfig(lang="en-us", pause_mode="auto", pause_clause=0.25)
    doc, trace, segments, detector_calls, calls, analysis_doc = _run_case(
        monkeypatch, generation, [_boundary()]
    )

    assert detector_calls == 1
    assert calls == [(TEXT, "en-us", analysis_doc)]
    assert [segment.text for segment in segments] == [TEXT[: COMMA + 1], TEXT[COMMA + 1 :]]
    assert segments[0].char_end == COMMA + 1
    assert segments[1].char_start == COMMA + 1

    events = [
        event
        for event in doc.boundary_events
        if event.attrs.get("boundary_kind") == "clausal_comma"
    ]
    assert len(events) == 1
    event = events[0]
    assert event.pos == COMMA
    assert event.duration_s == pytest.approx(0.25)
    assert event.attrs == {
        "strength": "c",
        "anchor": "after",
        "source": "pipeline_default",
        "detector": "phrasplit",
        "boundary_kind": "clausal_comma",
    }
    diagnostic = next(item for item in trace.events if item.name == "clausal_comma_boundaries")
    assert diagnostic.details == {
        "detected": 1,
        "added": 1,
        "deduplicated": 0,
        "runs_with_analysis": 1,
        "runs_without_analysis": 0,
    }
    assert doc.metadata["segmentation"]["clausal_comma_boundaries"] == 1


def test_clausal_comma_pause_is_owned_by_preceding_g2p_segment(monkeypatch):
    generation = GenerationConfig(lang="en-us", pause_mode="auto", pause_clause=0.25)
    doc, _trace, segments, _detector_calls, _calls, _analysis_doc = _run_case(
        monkeypatch, generation, [_boundary()]
    )

    sliced = [
        slice_boundaries(
            doc.boundary_events, segment.char_start, segment.char_end, doc_end=len(TEXT)
        )
        for segment in segments
    ]
    assert [boundary.pos for boundary in sliced[0]] == [COMMA]
    assert sliced[1] == []
    from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter

    adapter = KokoroG2PAdapter()
    assert adapter._resolve_pauses(sliced[0], generation) == (0.0, pytest.approx(0.25))
    assert adapter._resolve_pauses(sliced[1], generation) == (0.0, 0.0)


@pytest.mark.parametrize("pause_mode", ["tts", "manual"])
def test_non_auto_modes_do_not_add_clausal_pause(monkeypatch, pause_mode):
    generation = GenerationConfig(lang="en-us", pause_mode=pause_mode, pause_clause=0.25)
    doc, _trace, segments, detector_calls, _calls, _analysis_doc = _run_case(
        monkeypatch, generation, [_boundary()]
    )

    assert detector_calls == 0
    assert len(segments) == 1
    assert not [event for event in doc.boundary_events if event.attrs.get("boundary_kind")]


def test_zero_clause_pause_does_not_split(monkeypatch):
    generation = GenerationConfig(lang="en-us", pause_mode="auto", pause_clause=0.0)
    doc, _trace, segments, detector_calls, _calls, _analysis_doc = _run_case(
        monkeypatch, generation, [_boundary()]
    )

    assert detector_calls == 0
    assert len(segments) == 1
    assert not doc.boundary_events


def test_explicit_pause_wins_at_comma(monkeypatch):
    explicit = BoundaryEvent(
        pos=COMMA,
        kind="pause",
        duration_s=0.9,
        attrs={"strength": "s", "source": "explicit"},
    )
    generation = GenerationConfig(lang="en-us", pause_mode="auto", pause_clause=0.25)
    doc, trace, segments, _detector_calls, _calls, _analysis_doc = _run_case(
        monkeypatch, generation, [_boundary()], existing=[explicit]
    )

    assert len(segments) == 2
    at_comma = [event for event in doc.boundary_events if event.pos == COMMA]
    assert at_comma == [explicit]
    diagnostic = next(item for item in trace.events if item.name == "clausal_comma_boundaries")
    assert diagnostic.details["deduplicated"] == 1


def test_no_analysis_skips_detector_without_crashing(monkeypatch):
    calls = 0

    def detect(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("detector must not run without a prepared document")

    fake_module = types.SimpleNamespace(
        split_with_offsets=lambda text, *args, **kwargs: [(text, 0, len(text), 0, 0, 0)],
        detect_clause_boundaries=detect,
    )
    monkeypatch.setitem(sys.modules, "phrasplit", fake_module)
    run = LanguageRun(0, len(TEXT), "en-us")
    analysis = PreparedRunAnalysis(run=run, text=TEXT, doc=None, annotations=())
    state = LinguisticRequestState(prepared_plan=(run,), prepared_analysis=[analysis])
    doc = DocumentResult(clean_text=TEXT, linguistic_state=state)
    trace = Trace()

    segments = PhrasplitSentenceSegmenter().split(
        doc, PipelineConfig(generation=GenerationConfig(lang="en-us", pause_mode="auto")), trace
    )

    assert calls == 0
    assert len(segments) == 1
    assert not doc.boundary_events
    diagnostic = next(item for item in trace.events if item.name == "clausal_comma_boundaries")
    assert diagnostic.details["runs_without_analysis"] == 1


def test_list_and_shared_subject_commas_remain_unsplit(monkeypatch):
    for text in (
        "It picked up sound, light, smoke, and debris.",
        "I opened the door, looked inside, and walked away.",
    ):
        run = LanguageRun(0, len(text), "en-us")
        analysis = PreparedRunAnalysis(run=run, text=text, doc=object(), annotations=())
        state = LinguisticRequestState(prepared_plan=(run,), prepared_analysis=[analysis])
        doc = DocumentResult(clean_text=text, linguistic_state=state)
        fake_module = types.SimpleNamespace(
            split_with_offsets=lambda value, *args, **kwargs: [(value, 0, len(value), 0, 0, 0)],
            detect_clause_boundaries=lambda *args, **kwargs: [],
        )
        monkeypatch.setitem(sys.modules, "phrasplit", fake_module)

        segments = PhrasplitSentenceSegmenter().split(
            doc,
            PipelineConfig(generation=GenerationConfig(lang="en-us", pause_mode="auto")),
            Trace(),
        )

        assert len(segments) == 1
        assert not doc.boundary_events


def test_malformed_boundary_is_ignored_with_warning(monkeypatch):
    malformed = types.SimpleNamespace(
        text="wrong",
        char_start=COMMA,
        char_end=COMMA + 1,
        kind="clausal_comma",
    )
    generation = GenerationConfig(lang="en-us", pause_mode="auto", pause_clause=0.25)
    doc, trace, segments, _detector_calls, _calls, _analysis_doc = _run_case(
        monkeypatch, generation, [malformed]
    )

    assert len(segments) == 1
    assert not doc.boundary_events
    assert any("mismatched clean_text slice" in warning for warning in trace.warnings)


def test_stale_phrasplit_without_detector_warns_and_falls_back(monkeypatch):
    fake_module = types.SimpleNamespace(
        split_with_offsets=lambda text, *args, **kwargs: [(text, 0, len(text), 0, 0, 0)]
    )
    monkeypatch.setitem(sys.modules, "phrasplit", fake_module)
    run = LanguageRun(0, len(TEXT), "en-us")
    analysis = PreparedRunAnalysis(run=run, text=TEXT, doc=object(), annotations=())
    state = LinguisticRequestState(prepared_plan=(run,), prepared_analysis=[analysis])
    doc = DocumentResult(clean_text=TEXT, linguistic_state=state)
    trace = Trace()

    segments = PhrasplitSentenceSegmenter().split(
        doc, PipelineConfig(generation=GenerationConfig(lang="en-us", pause_mode="auto")), trace
    )

    assert len(segments) == 1
    assert not doc.boundary_events
    assert any("detect_clause_boundaries is unavailable" in warning for warning in trace.warnings)
