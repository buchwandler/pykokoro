from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

from pykokoro.runtime.linguistics import LinguisticResourcePool


def test_spacy_selection_load_and_analysis_timings(monkeypatch, caplog) -> None:
    token = SimpleNamespace(idx=0, text="Hello", pos_="INTJ", tag_="UH", lemma_="hello")

    class FakePipeline:
        meta = {"name": "fake"}

        def __call__(self, text: str) -> list[object]:
            assert text == "Hello"
            return [token]

    pipeline = FakePipeline()
    monkeypatch.setitem(sys.modules, "spacy", SimpleNamespace(load=lambda model: pipeline))
    caplog.set_level(logging.DEBUG, logger="pykokoro.runtime.linguistics")

    resources = LinguisticResourcePool()
    analysis = resources.analyze("Hello", language="en-us", model="fake_model")
    resources.analyze("Hello", language="en-us", model="fake_model")

    assert analysis is not None
    messages = [record.getMessage() for record in caplog.records]
    assert any("linguistics.spacy.selection.finish" in message for message in messages)
    assert any("linguistics.spacy.import.finish" in message for message in messages)
    assert any("linguistics.spacy.load.start" in message for message in messages)
    assert any("linguistics.spacy.load.finish" in message for message in messages)
    assert any(
        "linguistics.spacy.analyze.finish" in message and "tokens=1" in message
        for message in messages
    )
    assert any("cache_hit=True" in message for message in messages)
