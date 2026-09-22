from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from utterplan import PlannerConfig, UtterancePlan, UtterancePlanner
from utterplan.units import make_units

from pykokoro.planning.utterplan_adapter import adapt_plan
from pykokoro.runtime.language_plan import LanguageRun
from pykokoro.runtime.linguistics import (
    LinguisticRequestState,
    PreparedRunAnalysis,
    TokenAnnotation,
)
from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter
from pykokoro.stages.protocols import DocumentResult
from pykokoro.types import Segment


def _annotated_plan(text: str, annotations: dict[str, dict[str, str]]) -> UtterancePlan:
    plan = UtterancePlanner(
        PlannerConfig(language="en-us", text_preparation="identity", document_format="ssmd")
    ).plan(text)
    tokens = tuple(replace(token, **annotations.get(token.text, {})) for token in plan.tokens)
    units = make_units(plan.segments, plan.markers, tokens, plan.units[0].kind)
    return replace(plan, tokens=tokens, units=units).with_identity()

def _target_segment(plan: UtterancePlan, text: str):
    rendered = adapt_plan(plan)
    segment = next(segment for segment in rendered.segments if segment.text == text)
    return rendered, segment


def test_plan_token_annotations_forward_with_local_offsets_and_provenance() -> None:
    plan = _annotated_plan(
        "Intro. I live here.",
        {
            "I": {"pos": "PRON", "tag": "PRP", "lemma": "I", "morph": "Case=Nom"},
            "live": {"pos": "VERB", "tag": "VBP", "lemma": "live", "morph": "Tense=Pres"},
            "here.": {"pos": "ADV", "tag": "RB", "lemma": "here", "morph": "PronType=Dem"},
        },
    )
    rendered, segment = _target_segment(plan, "I live here.")

    annotations = KokoroG2PAdapter._prepared_annotations(rendered.document, segment.to_segment())

    assert rendered.document.metadata["utterplan_linguistic_runs"] == plan.linguistic_runs
    assert [
        (
            item.start,
            item.end,
            item.text,
            item.pos,
            item.tag,
            item.lemma,
            item.language,
            item.morph,
        )
        for item in annotations
    ] == [
        (0, 1, "I", "PRON", "PRP", "I", "en-us", "Case=Nom"),
        (2, 6, "live", "VERB", "VBP", "live", "en-us", "Tense=Pres"),
        (7, 12, "here.", "ADV", "RB", "here", "en-us", "PronType=Dem"),
    ]


class _GrammarAwareG2P:
    def __init__(self) -> None:
        self.annotations = ()

    def phonemize_prepared(self, text: str, **kwargs: object) -> SimpleNamespace:
        self.annotations = tuple(kwargs["annotations"] or ())
        live = next(item for item in self.annotations if item.text == "live")
        phonemes = "GRAMMAR_VERB" if live.pos == "VERB" else "GRAMMAR_ADJECTIVE"
        return SimpleNamespace(phonemes=phonemes, ids=[1], tokens=[])


def test_grammar_annotations_change_prepared_g2p_output() -> None:
    cases = (
        (
            "I live here.",
            {
                "I": {"pos": "PRON", "tag": "PRP", "lemma": "I"},
                "live": {"pos": "VERB", "tag": "VBP", "lemma": "live"},
                "here.": {"pos": "ADV", "tag": "RB", "lemma": "here"},
            },
            "GRAMMAR_VERB",
        ),
        (
            "a live show",
            {
                "a": {"pos": "DET", "tag": "DT", "lemma": "a"},
                "live": {"pos": "ADJ", "tag": "JJ", "lemma": "live"},
                "show": {"pos": "NOUN", "tag": "NN", "lemma": "show"},
            },
            "GRAMMAR_ADJECTIVE",
        ),
    )

    outputs = []
    for text, fields, expected in cases:
        plan = _annotated_plan(text, fields)
        rendered, segment = _target_segment(plan, text)
        fake = _GrammarAwareG2P()
        result = KokoroG2PAdapter._phonemize_prepared(
            fake,
            segment.text,
            segment.language,
            [],
            KokoroG2PAdapter._prepared_annotations(rendered.document, segment.to_segment()),
            object(),
        )
        outputs.append(result.phonemes)
        assert result.phonemes == expected
        assert next(item for item in fake.annotations if item.text == "live").pos == fields["live"]["pos"]

    assert outputs == ["GRAMMAR_VERB", "GRAMMAR_ADJECTIVE"]


def test_plan_round_trip_preserves_token_annotations(tmp_path) -> None:
    plan = _annotated_plan(
        "a live show",
        {
            "a": {"pos": "DET", "tag": "DT", "lemma": "a", "morph": "Definite=Ind"},
            "live": {"pos": "ADJ", "tag": "JJ", "lemma": "live", "morph": "Degree=Pos"},
            "show": {"pos": "NOUN", "tag": "NN", "lemma": "show", "morph": "Number=Sing"},
        },
    )
    path = tmp_path / "annotated.utterplan.json"
    plan.save(path)
    loaded = UtterancePlan.load(path)

    assert loaded.tokens == plan.tokens
    rendered, segment = _target_segment(loaded, "a live show")
    annotations = KokoroG2PAdapter._prepared_annotations(rendered.document, segment.to_segment())
    assert [item.morph for item in annotations] == ["Definite=Ind", "Degree=Pos", "Number=Sing"]


def test_plan_annotations_do_not_request_spacy(monkeypatch) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise AssertionError("plan rendering requested spaCy")

    monkeypatch.setattr(
        "pykokoro.runtime.linguistics.LinguisticResourcePool.get_spacy_pipeline",
        fail,
    )
    plan = _annotated_plan(
        "a live show",
        {
            "a": {"pos": "DET"},
            "live": {"pos": "ADJ"},
            "show": {"pos": "NOUN"},
        },
    )
    rendered, segment = _target_segment(plan, "a live show")

    assert KokoroG2PAdapter._prepared_annotations(rendered.document, segment.to_segment())


def test_invalid_plan_token_index_is_a_hard_error() -> None:
    plan = _annotated_plan("a live show", {"live": {"pos": "ADJ"}})
    rendered, segment = _target_segment(plan, "a live show")
    segment_value = segment.to_segment()
    segment_value = replace(
        segment_value,
        meta={**segment_value.meta, "plan_token_indices": (999,)},
    )

    with pytest.raises(ValueError, match="invalid token"):
        KokoroG2PAdapter._prepared_annotations(rendered.document, segment_value)


def test_linguistic_state_annotations_preserve_morph() -> None:
    text = "live"
    doc = DocumentResult(clean_text=text)
    doc.linguistic_state = LinguisticRequestState(
        prepared_analysis=[
            PreparedRunAnalysis(
                run=LanguageRun(0, len(text), "en-us"),
                text=text,
                doc=None,
                annotations=(
                    TokenAnnotation(
                        start=0,
                        end=len(text),
                        text=text,
                        pos="VERB",
                        tag="VBP",
                        lemma="live",
                        language="en-us",
                        morph="Tense=Pres",
                    ),
                ),
            ),
        ]
    )
    segment = Segment(
        id="seg",
        text=text,
        char_start=0,
        char_end=len(text),
        paragraph_idx=0,
        sentence_idx=0,
        clause_idx=0,
    )

    annotations = KokoroG2PAdapter._prepared_annotations(doc, segment)

    assert annotations[0].morph == "Tense=Pres"
    assert annotations[0].pos == "VERB"
    assert annotations[0].tag == "VBP"
    assert annotations[0].lemma == "live"


def test_missing_prepared_api_rejects_annotations_but_keeps_empty_fallback() -> None:
    class LegacyG2P:
        def phonemize(self, *args: object, **kwargs: object) -> str:
            return "legacy"

    with pytest.raises(RuntimeError, match="prepared token annotations"):
        KokoroG2PAdapter._phonemize_prepared(
            LegacyG2P(), "live", "en-us", [], [object()], object()
        )

    assert (
        KokoroG2PAdapter._phonemize_prepared(LegacyG2P(), "live", "en-us", [], [], object())
        == "legacy"
    )
