"""Regression coverage for kokorog2p 0.8's Spokenform integration boundary."""

from __future__ import annotations

import kokorog2p
from spokenform import prepare_for_kokorog2p

from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter
from pykokoro.stages.protocols import DocumentResult
from pykokoro.types import Segment, Trace

TEXT = (
    "Dr. Smith will see you at 10:30 on 05/20/2023. "
    "The box weighs 5 kg and costs $10.99. "
    "The temperature is 98.6°F. "
    "She finished in 1st place."
)


def _result_ids(result) -> list[int]:
    return list(getattr(result, "ids", None) or getattr(result, "token_ids", None) or [])


def test_kokorog2p_spokenform_semantics() -> None:
    prepared = prepare_for_kokorog2p(TEXT, language="en-us")
    result = kokorog2p.phonemize_prepared(
        prepared.spoken_text,
        language="en-us",
        use_spacy=False,
        return_phonemes=True,
        return_ids=True,
    )

    spoken = str(getattr(result, "extended_text", ""))

    assert "Doctor Smith" in spoken
    assert "ten thirty" in spoken
    assert "May twentieth" in spoken
    assert "five kilograms" in spoken
    assert "ten dollars" in spoken
    assert "ninety eight point six degrees Fahrenheit" in spoken
    assert "first place" in spoken
    assert result.phonemes
    assert _result_ids(result)
    assert not any(char.isdigit() for char in spoken)

def test_adapter_accepts_spokenform_rich_source_without_warnings() -> None:
    cfg = PipelineConfig(generation=GenerationConfig(lang="en-us"))
    doc = DocumentResult(clean_text=TEXT)
    segments = [
        Segment(
            id="seg_0",
            text=TEXT,
            char_start=0,
            char_end=len(TEXT),
            paragraph_idx=0,
            sentence_idx=0,
        )
    ]
    trace = Trace()

    out = KokoroG2PAdapter().phonemize(segments, doc, cfg, trace)

    assert out
    assert all(segment.phonemes for segment in out)
    assert all(segment.tokens for segment in out)
    assert not [warning for warning in trace.warnings if "[SPOKENFORM]" in warning.upper()]


def test_adapter_uses_prepared_entrypoint(monkeypatch) -> None:
    calls: list[str] = []
    prepared = kokorog2p.phonemize_prepared

    def spy_prepared(*args, **kwargs):
        calls.append("prepared")
        return prepared(*args, **kwargs)

    def forbidden_written(*args, **kwargs):
        raise AssertionError("written phonemize entrypoint was used")

    monkeypatch.setattr(kokorog2p, "phonemize_prepared", spy_prepared)
    monkeypatch.setattr(kokorog2p, "phonemize", forbidden_written)

    cfg = PipelineConfig(generation=GenerationConfig(lang="en-us"))
    doc = DocumentResult(clean_text="Hello world.")
    segment = Segment(
        id="seg",
        text=doc.clean_text,
        char_start=0,
        char_end=len(doc.clean_text),
        paragraph_idx=0,
        sentence_idx=0,
    )
    KokoroG2PAdapter().phonemize([segment], doc, cfg, Trace())
    assert calls == ["prepared"]
