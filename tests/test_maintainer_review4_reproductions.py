"""Regression reproductions for the maintainer review 4 state/config defects."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline import KokoroPipeline
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter
from pykokoro.stages.protocols import DocumentResult
from pykokoro.tokenizer import TokenizerConfig
from pykokoro.types import PhonemeSegment, Segment


class _DocParser:
    def parse(self, text: str, cfg: PipelineConfig, trace: Any) -> DocumentResult:
        _ = (cfg, trace)
        return DocumentResult(
            clean_text=text,
            segments=[
                Segment(
                    id="segment-0",
                    text=text,
                    char_start=0,
                    char_end=len(text),
                    paragraph_idx=0,
                    sentence_idx=0,
                    clause_idx=0,
                )
            ],
        )


class _RecordingG2P:
    def __init__(self) -> None:
        self.configs: list[PipelineConfig] = []

    def phonemize(
        self,
        segments: list[Segment],
        doc: DocumentResult,
        cfg: PipelineConfig,
        trace: Any,
    ) -> list[PhonemeSegment]:
        _ = trace
        self.configs.append(cfg)
        segment = segments[0]
        return [
            PhonemeSegment(
                id="phoneme-0",
                segment_id=segment.id,
                phoneme_id=0,
                text=doc.clean_text,
                phonemes="a",
                tokens=[],
                lang=cfg.generation.lang,
                char_start=segment.char_start,
                char_end=segment.char_end,
                paragraph_idx=segment.paragraph_idx,
                sentence_idx=segment.sentence_idx,
                clause_idx=segment.clause_idx,
            )
        ]


class _PassthroughStage:
    def process(self, phoneme_segments, cfg, trace):
        _ = (cfg, trace)
        return phoneme_segments

    def generate(self, phoneme_segments, cfg, trace):
        _ = (cfg, trace)
        return phoneme_segments

    def postprocess(self, phoneme_segments, cfg, trace):
        _ = (phoneme_segments, cfg, trace)
        return np.zeros(4, dtype=np.float32)


def _pipeline(g2p: _RecordingG2P) -> KokoroPipeline:
    stage = _PassthroughStage()
    return KokoroPipeline(
        PipelineConfig(),
        doc_parser=_DocParser(),
        g2p=g2p,
        phoneme_processing=stage,
        audio_generation=stage,
        audio_postprocessing=stage,
    )


def test_run_accepts_mapping_generation_with_lang() -> None:
    g2p = _RecordingG2P()
    pipeline = _pipeline(g2p)

    pipeline.run("hello", generation={"speed": 1.2}, lang="de")

    assert g2p.configs[-1].generation == GenerationConfig(speed=1.2, lang="de")


def test_g2p_instance_cache_key_includes_model_and_tokenizer_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[dict[str, object]] = []

    class FakeG2PModule:
        @staticmethod
        def get_g2p(**kwargs: object) -> object:
            created.append(kwargs)
            return object()

    adapter = KokoroG2PAdapter()
    monkeypatch.setattr(adapter, "_load", lambda: FakeG2PModule())
    first_cfg = PipelineConfig(
        model_variant="v1.0",
        tokenizer_config=TokenizerConfig(backend="kokorog2p"),
    )
    second_cfg = replace(
        first_cfg,
        model_variant="v1.1-zh",
        tokenizer_config=TokenizerConfig(backend="espeak"),
    )

    first = adapter._get_g2p_instance("en-us", first_cfg)
    second = adapter._get_g2p_instance("en-us", second_cfg)

    assert first is not second
    assert [kwargs["version"] for kwargs in created] == ["1.0", "1.1"]
    assert [kwargs["backend"] for kwargs in created] == ["kokorog2p", "espeak"]


def test_failed_backend_replacement_preserves_open_previous_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[Any] = []

    class FakeKokoro:
        def __init__(self, **kwargs: object) -> None:
            self.closed = False
            instances.append(self)
            if kwargs["model_variant"] == "v1.1-zh":
                raise RuntimeError("replacement failed")

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("pykokoro.onnx_backend.Kokoro", FakeKokoro)
    pipeline = KokoroPipeline(
        PipelineConfig(model_variant="v1.0", generation=GenerationConfig(lang="en-us"))
    )
    pipeline._ensure_kokoro(pipeline.config)

    with pytest.raises(RuntimeError, match="replacement failed"):
        pipeline._ensure_kokoro(replace(pipeline.config, model_variant="v1.1-zh"))

    assert instances[0].closed is False
    assert pipeline._kokoro is instances[0]


def test_backend_cache_key_snapshots_mutable_nested_configuration() -> None:
    pipeline = KokoroPipeline(PipelineConfig(generation=GenerationConfig(lang="en-us")))
    provider_options = {"execution": {"device_id": 0}}
    tokenizer_config = TokenizerConfig(lexicons=("gold",))
    cfg = replace(
        pipeline.config,
        provider_options=provider_options,
        tokenizer_config=tokenizer_config,
    )

    initial_key = pipeline._kokoro_key(cfg)
    provider_options["execution"]["device_id"] = 1
    provider_key = pipeline._kokoro_key(cfg)
    assert provider_key != initial_key
    object.__setattr__(tokenizer_config, "lexicons", ("gold", "de"))
    assert pipeline._kokoro_key(cfg) == provider_key


def test_backend_cache_key_snapshots_mutable_spokenform_sensitive_tokenizer_flags() -> None:
    pipeline = KokoroPipeline(PipelineConfig(generation=GenerationConfig(lang="en-us")))
    tokenizer_config = TokenizerConfig(load_gold=True, load_silver=True, use_espeak_fallback=True)
    cfg = replace(pipeline.config, tokenizer_config=tokenizer_config)

    initial_key = pipeline._kokoro_key(cfg)
    tokenizer_config.load_silver = False

    assert pipeline._kokoro_key(cfg) != initial_key
