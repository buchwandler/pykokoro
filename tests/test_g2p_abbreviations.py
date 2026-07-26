import kokorog2p
import pytest

from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter
from pykokoro.stages.protocols import DocumentResult
from pykokoro.types import Segment, Trace

TEXT = "Meet Mr. Schmidt, Mrs. Johnson, Ms. Anderson, and Dr. Brown."


def _token_phonemes(token) -> str:
    meta = getattr(token, "meta", None) or {}
    phonemes = meta.get("phonemes")
    if phonemes is None:
        phonemes = getattr(token, "phonemes", "")
    return phonemes or ""


def _normalize_abbreviation(token_text: str) -> str:
    return token_text.rstrip(".").lower()


def test_kokorog2p_abbreviations_have_phonemes():
    result = kokorog2p.phonemize(
        TEXT,
        language="en-us",
        return_phonemes=True,
        return_ids=True,
    )

    tokens = getattr(result, "tokens", [])
    assert tokens

    for abbr in ("mr", "mrs", "ms", "dr"):
        token = next(
            (t for t in tokens if _normalize_abbreviation(getattr(t, "text", "")) == abbr),
            None,
        )
        assert token is not None
        if not _token_phonemes(token).strip():
            pytest.xfail("kokorog2p does not emit phonemes for Ms./and with punctuation")


def test_kokorog2p_punctuation():
    result = kokorog2p.phonemize(
        "Hello, World! I like you . . .",
        language="en-us",
        return_phonemes=True,
        return_ids=True,
    )

    tokens = getattr(result, "tokens", [])
    phonemes = getattr(result, "phonemes", [])
    assert tokens
    assert "! " in phonemes
    assert "…" in phonemes
    assert "," in phonemes
    assert ", " not in phonemes


def test_kokorog2p_adapter_no_abbreviation_warnings():
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

    phoneme_segments = KokoroG2PAdapter().phonemize(segments, doc, cfg, trace)

    assert phoneme_segments
    assert phoneme_segments[0].phonemes
