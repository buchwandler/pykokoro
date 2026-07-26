import sys
import types

import pytest

from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter
from pykokoro.stages.protocols import DocumentResult
from pykokoro.types import BoundaryEvent, Segment, Trace


def test_pause_propagation_for_single_batch(monkeypatch):
    fake_g2p = types.SimpleNamespace(__version__="0.0")

    def phonemes_to_ids(phonemes, model=None):
        _ = phonemes, model
        return [1, 2]

    def ids_to_phonemes(ids, model=None):
        _ = ids, model
        return "a"

    fake_g2p.phonemes_to_ids = phonemes_to_ids
    fake_g2p.ids_to_phonemes = ids_to_phonemes

    monkeypatch.setitem(sys.modules, "kokorog2p", fake_g2p)

    generation = GenerationConfig(is_phonemes=True, pause_mode="manual", pause_paragraph=1.25)
    cfg = PipelineConfig(generation=generation)

    text = "aaaaa.\n\nbbbbb."
    doc = DocumentResult(
        clean_text=text,
        boundary_events=[
            BoundaryEvent(pos=0, kind="pause", duration_s=None, attrs={"strength": "p"}),
            BoundaryEvent(pos=5, kind="pause", duration_s=None, attrs={"strength": "p"}),
        ],
    )
    segments = [
        Segment(
            id="seg_0",
            text="aaaaa.",
            char_start=0,
            char_end=6,
            paragraph_idx=0,
            sentence_idx=0,
            clause_idx=0,
        ),
        Segment(
            id="seg_1",
            text="bbbbb.",
            char_start=8,
            char_end=14,
            paragraph_idx=1,
            sentence_idx=1,
            clause_idx=0,
        ),
    ]

    trace = Trace()
    phoneme_segments = KokoroG2PAdapter().phonemize(segments, doc, cfg, trace)

    first_segment = next(seg for seg in phoneme_segments if seg.segment_id == "seg_0")
    assert first_segment.pause_before == pytest.approx(generation.pause_paragraph)
    assert first_segment.pause_after == pytest.approx(generation.pause_paragraph)
