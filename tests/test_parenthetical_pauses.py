from __future__ import annotations

import sys
import types

import pytest

from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.runtime.language_plan import LanguageRun
from pykokoro.runtime.linguistics import LinguisticRequestState, PreparedRunAnalysis
from pykokoro.runtime.spans import slice_boundaries
from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter
from pykokoro.stages.protocols import DocumentResult
from pykokoro.stages.segmentation.phrasplit import PhrasplitSentenceSegmenter
from pykokoro.types import BoundaryEvent, Segment, Trace

FINAL_TEXT = "They changed out their clothes (stained with blood)."
MEDIAL_TEXT = "They changed clothes (stained with blood) before leaving."


def _boundary(text: str, kind: str, position: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        text=text[position : position + 1],
        char_start=position,
        char_end=position + 1,
        kind=kind,
        meta={"confidence": 1.0, "reason": "test"},
    )


def _run_case(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    boundaries: list[object],
    *,
    analysis_doc: object | None = object(),
    existing: list[BoundaryEvent] | None = None,
    pause_parenthetical: float = 0.15,
):
    calls: list[tuple[str, str | None]] = []

    fake_module = types.SimpleNamespace(
        split_with_offsets=lambda value, *args, **kwargs: [(value, 0, len(value), 0, 0, 0)],
        detect_clause_boundaries=lambda *args, **kwargs: [],
    )

    def detect_parenthetical_boundaries(value: str, *, language: str | None = None):
        calls.append((value, language))
        return boundaries

    fake_module.detect_parenthetical_boundaries = detect_parenthetical_boundaries
    monkeypatch.setitem(sys.modules, "phrasplit", fake_module)

    run = LanguageRun(0, len(text), "en-us")
    analysis = PreparedRunAnalysis(run=run, text=text, doc=analysis_doc, annotations=())
    state = LinguisticRequestState(prepared_plan=(run,), prepared_analysis=[analysis])
    doc = DocumentResult(clean_text=text, linguistic_state=state)
    if existing:
        doc.boundary_events.extend(existing)
    trace = Trace()
    generation = GenerationConfig(
        lang="en-us",
        pause_mode="auto",
        pause_parenthetical=pause_parenthetical,
        pause_variance=0.0,
    )
    segments = PhrasplitSentenceSegmenter().split(doc, PipelineConfig(generation=generation), trace)
    return doc, trace, segments, calls, generation


def test_auto_parenthetical_opening_creates_owned_pause_and_cut(monkeypatch):
    position = FINAL_TEXT.index("(")
    doc, trace, segments, calls, _generation = _run_case(
        monkeypatch,
        FINAL_TEXT,
        [_boundary(FINAL_TEXT, "parenthetical_open", position)],
    )

    assert calls == [(FINAL_TEXT, "en-us")]
    assert [segment.text for segment in segments] == [FINAL_TEXT[:position], FINAL_TEXT[position:]]
    events = [
        event for event in doc.boundary_events if event.attrs.get("pause_kind") == "parenthetical"
    ]
    assert len(events) == 1
    event = events[0]
    assert event.pos == position
    assert event.duration_s == pytest.approx(0.15)
    assert event.attrs == {
        "strength": "w",
        "anchor": "before",
        "source": "pipeline_default",
        "detector": "phrasplit",
        "pause_kind": "parenthetical",
        "boundary_kind": "parenthetical_open",
        "confidence": "1.0",
        "reason": "test",
    }
    diagnostic = next(item for item in trace.events if item.name == "parenthetical_boundaries")
    assert diagnostic.details == {
        "detected": 1,
        "open_detected": 1,
        "close_detected": 0,
        "added": 1,
        "deduplicated": 0,
        "malformed_skipped": 0,
    }
    assert doc.metadata["segmentation"]["parenthetical_boundaries"] == 1


def test_final_parenthetical_has_no_generated_close_pause(monkeypatch):
    position = FINAL_TEXT.index("(")
    doc, _trace, _segments, _calls, _generation = _run_case(
        monkeypatch,
        FINAL_TEXT,
        [_boundary(FINAL_TEXT, "parenthetical_open", position)],
    )

    assert not [
        event
        for event in doc.boundary_events
        if event.attrs.get("boundary_kind") == "parenthetical_close"
    ]


def test_medial_parenthetical_maps_open_and_close_coordinates(monkeypatch):
    opening = MEDIAL_TEXT.index("(")
    closing = MEDIAL_TEXT.index(")")
    doc, _trace, segments, _calls, _generation = _run_case(
        monkeypatch,
        MEDIAL_TEXT,
        [
            _boundary(MEDIAL_TEXT, "parenthetical_open", opening),
            _boundary(MEDIAL_TEXT, "parenthetical_close", closing),
        ],
    )

    assert [segment.text for segment in segments] == [
        MEDIAL_TEXT[:opening],
        MEDIAL_TEXT[opening : closing + 1],
        MEDIAL_TEXT[closing + 1 :],
    ]
    assert [event.pos for event in doc.boundary_events] == [opening, closing + 1]
    assert [event.attrs["boundary_kind"] for event in doc.boundary_events] == [
        "parenthetical_open",
        "parenthetical_close",
    ]


def test_parenthetical_pause_is_owned_by_aside_and_resumed_host(monkeypatch):
    opening = MEDIAL_TEXT.index("(")
    closing = MEDIAL_TEXT.index(")")
    doc, _trace, segments, _calls, generation = _run_case(
        monkeypatch,
        MEDIAL_TEXT,
        [
            _boundary(MEDIAL_TEXT, "parenthetical_open", opening),
            _boundary(MEDIAL_TEXT, "parenthetical_close", closing),
        ],
    )

    sliced = [
        slice_boundaries(doc.boundary_events, segment.char_start, segment.char_end, doc_end=len(MEDIAL_TEXT))
        for segment in segments
    ]
    adapter = KokoroG2PAdapter()
    assert adapter._resolve_pauses(sliced[1], generation) == (pytest.approx(0.15), 0.0)
    assert adapter._resolve_pauses(sliced[2], generation) == (pytest.approx(0.15), 0.0)


@pytest.mark.parametrize("pause_mode", ["tts", "manual"])
def test_non_auto_modes_do_not_add_parenthetical_pause(monkeypatch, pause_mode):
    position = MEDIAL_TEXT.index("(")
    fake_module = types.SimpleNamespace(
        split_with_offsets=lambda value, *args, **kwargs: [(value, 0, len(value), 0, 0, 0)],
        detect_clause_boundaries=lambda *args, **kwargs: [],
        detect_parenthetical_boundaries=lambda *args, **kwargs: pytest.fail("detector must not run"),
    )
    monkeypatch.setitem(sys.modules, "phrasplit", fake_module)
    run = LanguageRun(0, len(MEDIAL_TEXT), "en-us")
    analysis = PreparedRunAnalysis(run=run, text=MEDIAL_TEXT, doc=None, annotations=())
    doc = DocumentResult(
        clean_text=MEDIAL_TEXT,
        linguistic_state=LinguisticRequestState(
            prepared_plan=(run,), prepared_analysis=[analysis]
        ),
    )
    segments = PhrasplitSentenceSegmenter().split(
        doc,
        PipelineConfig(
            generation=GenerationConfig(lang="en-us", pause_mode=pause_mode, pause_parenthetical=0.15)
        ),
        Trace(),
    )

    assert len(segments) == 1
    assert not doc.boundary_events
    assert position > 0


def test_zero_parenthetical_pause_preserves_other_auto_detection(monkeypatch):
    calls = 0

    def detect(*args, **kwargs):
        nonlocal calls
        calls += 1
        return pytest.fail("parenthetical detector must not run for zero duration")

    fake_module = types.SimpleNamespace(
        split_with_offsets=lambda value, *args, **kwargs: [(value, 0, len(value), 0, 0, 0)],
        detect_clause_boundaries=lambda *args, **kwargs: [],
        detect_parenthetical_boundaries=detect,
    )
    monkeypatch.setitem(sys.modules, "phrasplit", fake_module)
    run = LanguageRun(0, len(MEDIAL_TEXT), "en-us")
    analysis = PreparedRunAnalysis(run=run, text=MEDIAL_TEXT, doc=object(), annotations=())
    doc = DocumentResult(
        clean_text=MEDIAL_TEXT,
        linguistic_state=LinguisticRequestState(
            prepared_plan=(run,), prepared_analysis=[analysis]
        ),
    )

    segments = PhrasplitSentenceSegmenter().split(
        doc,
        PipelineConfig(generation=GenerationConfig(lang="en-us", pause_mode="auto", pause_parenthetical=0.0)),
        Trace(),
    )

    assert calls == 0
    assert len(segments) == 1


def test_no_spacy_analysis_still_detects_parenthetical(monkeypatch):
    opening = MEDIAL_TEXT.index("(")
    doc, _trace, segments, _calls, _generation = _run_case(
        monkeypatch,
        MEDIAL_TEXT,
        [_boundary(MEDIAL_TEXT, "parenthetical_open", opening)],
        analysis_doc=None,
    )

    assert len(segments) == 2
    assert doc.boundary_events[0].pos == opening


def test_explicit_pause_wins_at_parenthetical_boundary(monkeypatch):
    opening = MEDIAL_TEXT.index("(")
    explicit = BoundaryEvent(
        pos=opening,
        kind="pause",
        duration_s=0.9,
        attrs={"strength": "s", "source": "explicit"},
    )
    doc, trace, _segments, _calls, _generation = _run_case(
        monkeypatch,
        MEDIAL_TEXT,
        [_boundary(MEDIAL_TEXT, "parenthetical_open", opening)],
        existing=[explicit],
    )

    assert [event for event in doc.boundary_events if event.pos == opening] == [explicit]
    diagnostic = next(item for item in trace.events if item.name == "parenthetical_boundaries")
    assert diagnostic.details["deduplicated"] == 1


@pytest.mark.parametrize(
    "candidate",
    [
        types.SimpleNamespace(text="(", char_start=-1, char_end=0, kind="parenthetical_open"),
        types.SimpleNamespace(text="(", char_start=0, char_end=999, kind="parenthetical_open"),
        types.SimpleNamespace(text="(", char_start=2, char_end=2, kind="parenthetical_open"),
        types.SimpleNamespace(text=")", char_start=2, char_end=3, kind="parenthetical_open"),
        types.SimpleNamespace(text="(", char_start=2, char_end=3, kind="unknown"),
        types.SimpleNamespace(text="(", char_start="2", char_end=3, kind="parenthetical_open"),
    ],
)
def test_malformed_parenthetical_results_are_skipped(monkeypatch, candidate):
    doc, trace, segments, _calls, _generation = _run_case(monkeypatch, MEDIAL_TEXT, [candidate])

    assert len(segments) == 1
    assert not doc.boundary_events
    diagnostic = next(item for item in trace.events if item.name == "parenthetical_boundaries")
    assert diagnostic.details["malformed_skipped"] == 1
    assert any("malformed" in warning for warning in trace.warnings)


def test_automatic_collision_does_not_sum_pause_durations(monkeypatch):
    opening = MEDIAL_TEXT.index("(")
    explicit_auto = BoundaryEvent(
        pos=opening,
        kind="pause",
        duration_s=0.3,
        attrs={"strength": "c", "source": "pipeline_default"},
    )
    doc, trace, _segments, _calls, _generation = _run_case(
        monkeypatch,
        MEDIAL_TEXT,
        [_boundary(MEDIAL_TEXT, "parenthetical_open", opening)],
        existing=[explicit_auto],
    )

    assert [event for event in doc.boundary_events if event.pos == opening] == [explicit_auto]
    diagnostic = next(item for item in trace.events if item.name == "parenthetical_boundaries")
    assert diagnostic.details["deduplicated"] == 1


def test_parenthetical_pause_propagates_through_g2p_boundaries():
    text = "Host (aside) resumes."
    opening = text.index("(")
    closing = text.index(")")
    generation = GenerationConfig(
        lang="en-us",
        pause_mode="auto",
        pause_parenthetical=0.12,
    )
    doc = DocumentResult(
        clean_text=text,
        boundary_events=[
            BoundaryEvent(
                pos=opening,
                kind="pause",
                duration_s=generation.pause_parenthetical,
                attrs={"anchor": "before", "pause_kind": "parenthetical"},
            ),
            BoundaryEvent(
                pos=closing + 1,
                kind="pause",
                duration_s=generation.pause_parenthetical,
                attrs={"anchor": "before", "pause_kind": "parenthetical"},
            ),
        ],
    )
    segments = [
        Segment(id="host", text=text[:opening], char_start=0, char_end=opening),
        Segment(
            id="aside",
            text=text[opening : closing + 1],
            char_start=opening,
            char_end=closing + 1,
        ),
        Segment(
            id="resumed",
            text=text[closing + 1 :],
            char_start=closing + 1,
            char_end=len(text),
        ),
    ]
    sliced = [
        slice_boundaries(boundaries, segment.char_start, segment.char_end, doc_end=len(text))
        for segment, boundaries in zip(segments, [doc.boundary_events] * len(segments), strict=False)
    ]
    adapter = KokoroG2PAdapter()
    assert adapter._resolve_pauses(sliced[1], generation) == (pytest.approx(0.12), 0.0)
    assert adapter._resolve_pauses(sliced[2], generation) == (pytest.approx(0.12), 0.0)
