from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from pykokoro.emphasis import apply_emphasis_policy, resolve_emphasis
from pykokoro.exceptions import CapabilityError
from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline import KokoroPipeline
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.prosody import apply_volume
from pykokoro.ssmd_config import SSMDRenderConfig
from pykokoro.ssmd_parser import parse_ssmd_document
from pykokoro.stages.protocols import DocumentResult
from pykokoro.types import PhonemeSegment, Segment, Trace


def _segment(
    emphasis: str,
    *,
    segment_id: str = "source-1",
    metadata: dict[str, object] | None = None,
) -> PhonemeSegment:
    values = {"emphasis": emphasis}
    if metadata:
        values.update(metadata)
    return PhonemeSegment(
        id=f"{segment_id}-{emphasis}",
        segment_id=segment_id,
        phoneme_id=0,
        text="important",
        phonemes="test",
        tokens=[],
        ssmd_metadata=values,
    )


def _cfg(mode: str, *, scale: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(
        ssmd=SSMDRenderConfig(emphasis_mode=mode, emphasis_gain_scale=scale),
        generation=SimpleNamespace(pause_mode="none"),
    )


@pytest.mark.parametrize(
    ("level", "scale", "expected"),
    [
        ("moderate", 0.5, "+1.5dB"),
        ("strong", 0.5, "+3dB"),
        ("reduced", 0.5, "-1.5dB"),
        ("moderate", 1.0, "+3dB"),
        ("strong", 1.0, "+6dB"),
        ("reduced", 1.0, "-3dB"),
        ("moderate", 1.5, "+4.5dB"),
        ("strong", 1.5, "+9dB"),
        ("reduced", 1.5, "-4.5dB"),
    ],
)
def test_approximate_gain_scale_uses_deterministic_db_format(
    level: str, scale: float, expected: str
) -> None:
    decision = resolve_emphasis(level, "approximate", gain_scale=scale)

    assert decision.volume == expected
    assert "e" not in decision.volume.lower()


def test_default_emphasis_gain_scale_preserves_released_behavior() -> None:
    assert SSMDRenderConfig().emphasis_gain_scale == 1.0
    assert resolve_emphasis("moderate", "approximate").volume == "+3dB"
    assert resolve_emphasis("strong", "approximate").volume == "+6dB"
    assert resolve_emphasis("reduced", "approximate").volume == "-3dB"


@pytest.mark.parametrize(
    "value",
    [-0.1, 2.1, math.nan, math.inf, -math.inf, True, False],
)
def test_invalid_emphasis_gain_scale_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="emphasis_gain_scale"):
        SSMDRenderConfig(emphasis_gain_scale=value)  # type: ignore[arg-type]


def test_zero_emphasis_gain_scale_is_accepted_without_semantic_remapping() -> None:
    segment = _segment("strong")

    apply_emphasis_policy([segment], _cfg("approximate", scale=0.0), Trace())

    assert segment.ssmd_metadata == {"emphasis": "strong", "prosody_volume": "0dB"}


@pytest.mark.parametrize("mode", ["plain", "approximate", "warn", "error"])
def test_none_is_silent_noop_in_every_mode(mode: str) -> None:
    segment = _segment("none")
    trace = Trace()

    apply_emphasis_policy([segment], _cfg(mode), trace)

    assert segment.ssmd_metadata == {"emphasis": "none"}
    assert trace.warnings == []


@pytest.mark.parametrize("level", ["none", "reduced", "moderate", "strong"])
def test_plain_preserves_all_levels_without_mutation(level: str) -> None:
    segment = _segment(level)
    original = dict(segment.ssmd_metadata or {})
    trace = Trace()

    apply_emphasis_policy([segment], _cfg("plain"), trace)

    assert segment.ssmd_metadata == original
    assert trace.warnings == []


def test_plain_produces_no_gain_regardless_of_scale() -> None:
    segment = _segment("strong")

    apply_emphasis_policy([segment], _cfg("plain", scale=1.5), Trace())

    assert segment.ssmd_metadata == {"emphasis": "strong"}


@pytest.mark.parametrize("level", ["reduced", "moderate", "strong"])
def test_approximate_levels_use_non_neutral_volume(level: str) -> None:
    segment = _segment(level)

    apply_emphasis_policy([segment], _cfg("approximate"), Trace())

    volume = (segment.ssmd_metadata or {}).get("prosody_volume")
    assert isinstance(volume, str)
    audio = np.ones(16, dtype=np.float32)
    adjusted = apply_volume(audio, volume)
    assert not np.allclose(adjusted, audio)


def test_explicit_prosody_wins_over_approximation() -> None:
    segment = _segment(
        "strong",
        metadata={"prosody_volume": "soft", "prosody_rate": "slow"},
    )

    apply_emphasis_policy([segment], _cfg("approximate"), Trace())

    assert segment.ssmd_metadata == {
        "emphasis": "strong",
        "prosody_volume": "soft",
        "prosody_rate": "slow",
    }


def test_approximate_gain_does_not_add_rate_or_pitch() -> None:
    segment = _segment("strong")

    apply_emphasis_policy([segment], _cfg("approximate", scale=1.5), Trace())

    assert segment.ssmd_metadata == {"emphasis": "strong", "prosody_volume": "+9dB"}
    assert "prosody_rate" not in segment.ssmd_metadata
    assert "prosody_pitch" not in segment.ssmd_metadata


def test_warn_emits_one_diagnostic_per_logical_source_segment() -> None:
    segments = [
        _segment("strong", segment_id="source-1"),
        _segment("strong", segment_id="source-1"),
        _segment("moderate", segment_id="source-2"),
    ]
    trace = Trace()

    apply_emphasis_policy(segments, _cfg("warn", scale=1.5), trace)

    assert trace.warnings == [
        "ssmd.emphasis_unsupported: using unmodified speech",
        "ssmd.emphasis_unsupported: using unmodified speech",
    ]
    assert all(
        segment.ssmd_metadata == {"emphasis": level}
        for segment, level in zip(segments, ["strong", "strong", "moderate"], strict=True)
    )


def test_error_rejects_before_generation() -> None:
    with pytest.raises(CapabilityError):
        apply_emphasis_policy([_segment("strong")], _cfg("error", scale=1.5), Trace())


def test_ssmd_parser_emphasis_reaches_policy_metadata() -> None:
    parsed = parse_ssmd_document('[important]{emphasis="moderate"}')
    assert parsed.segments[0].metadata.emphasis == "moderate"

    segment = _segment(parsed.segments[0].metadata.emphasis)
    apply_emphasis_policy([segment], _cfg("approximate"), Trace())

    assert segment.ssmd_metadata["prosody_volume"] == "+3dB"


def test_invalid_mode_is_rejected_by_configuration() -> None:
    with pytest.raises(ValueError, match="emphasis_mode"):
        SSMDRenderConfig(emphasis_mode="invalid")  # type: ignore[arg-type]


class _Doc:
    def parse(self, text, cfg, trace):
        _ = cfg, trace
        return DocumentResult(
            clean_text=text,
            segments=[Segment("source-1", text, 0, len(text))],
        )


class _G2P:
    def phonemize(self, segments, doc, cfg, trace):
        _ = doc, trace
        return [_segment("strong", segment_id=segments[0].id)]


class _Processor:
    def process(self, segments, cfg, trace):
        _ = cfg, trace
        return segments


class _Generator:
    def __init__(self):
        self.calls = 0

    def generate(self, segments, cfg, trace):
        _ = cfg, trace
        self.calls += 1
        return segments


class _Postprocessor:
    def postprocess(self, segments, cfg, trace):
        _ = segments, cfg, trace
        return np.zeros(1, dtype=np.float32)


def test_error_policy_runs_before_audio_generation() -> None:
    generator = _Generator()
    pipeline = KokoroPipeline(
        PipelineConfig(
            generation=GenerationConfig(lang="en-us"),
            ssmd=SSMDRenderConfig(emphasis_mode="error"),
        ),
        doc_parser=_Doc(),
        g2p=_G2P(),
        phoneme_processing=_Processor(),
        audio_generation=generator,
        audio_postprocessing=_Postprocessor(),
    )

    with pytest.raises(CapabilityError):
        pipeline.run("important")

    assert generator.calls == 0
