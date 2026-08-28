from __future__ import annotations

from types import SimpleNamespace

from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.runtime.cache import DiskCache
from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter
from pykokoro.stages.protocols import DocumentResult
from pykokoro.types import Segment, Trace


def test_thorsten_cleanup_retokenizes_uncached_and_cached_results(tmp_path, monkeypatch) -> None:
    segment = Segment(
        id="segment-0",
        text="Brücke",
        char_start=0,
        char_end=6,
        paragraph_idx=0,
        sentence_idx=0,
        clause_idx=0,
    )
    config = PipelineConfig(
        cache_dir=str(tmp_path),
        model_source="github",
        model_variant="de-thorsten",
        generation=GenerationConfig(lang="de"),
    )
    calls = {"phonemize": 0}

    class FakeG2PModule:
        __version__ = "test"

        @staticmethod
        def phonemize(*args, **kwargs):
            _ = args, kwargs
            calls["phonemize"] += 1
            return SimpleNamespace(
                phonemes="bʏkə",
                ids=[999],
                tokens=[{"text": "Brücke", "phonemes": "bʏkə", "whitespace": ""}],
                warnings=[],
            )

        @staticmethod
        def phonemes_to_ids(phonemes, model=None):
            _ = model
            return [ord(char) for char in phonemes]

        @staticmethod
        def ids_to_phonemes(tokens, model=None):
            _ = tokens, model
            return "bʏkə"

    adapter = KokoroG2PAdapter()
    monkeypatch.setattr(adapter, "_load", lambda: FakeG2PModule())
    monkeypatch.setattr(adapter, "_get_g2p_instance", lambda lang, cfg: object())
    doc = DocumentResult(clean_text=segment.text, segments=[segment])

    uncached = adapter.phonemize([segment], doc, config, Trace())
    cached = adapter.phonemize([segment], doc, config, Trace())

    assert calls["phonemize"] == 1
    assert uncached[0].phonemes == "bykə"
    assert "ʏ" not in uncached[0].phonemes
    assert uncached[0].tokens == [ord(char) for char in "bykə"]
    assert cached[0].phonemes == uncached[0].phonemes
    assert cached[0].tokens == uncached[0].tokens
    assert cached[0].alignment_tokens[0].phonemes == "bykə"
    assert cached[0].alignment_tokens[0].model_token_count == len("bykə")

    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 1
    payload = DiskCache(tmp_path).get(cache_files[0].stem)
    assert payload["phonemes"] == "bykə"
    assert payload["tokens"] == [ord(char) for char in "bykə"]
