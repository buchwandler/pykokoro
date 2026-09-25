"""Public contracts for the request-centric breaking engine boundary."""

from __future__ import annotations

import inspect

import numpy as np

import pykokoro
from pykokoro import (
    GenerationConfig,
    KokoroSynthesizer,
    LanguageRoutingConfig,
    LinguisticToken,
    LongTextSplitMode,
    PronunciationOverride,
    RenderedSegment,
    SynthesisConfig,
    SynthesisIdentity,
    SynthesisInputTooLongError,
    SynthesisRequest,
    SynthesisSegment,
    VoiceBlend,
    VoiceLevelApplication,
    VoiceLevelConfig,
    build_synthesis_identity,
)
from pykokoro.types import PhonemeSegment


def test_public_api_exports_request_and_result_types() -> None:
    expected = {
        "AssetProgressEvent",
        "AssetProgressCallback",
        "ConsoleAssetProgress",
        "ModelCapabilities",
        "ModelDiscoveryResult",
        "VoiceCapabilities",
        "discover_models",
        "LexiconCapabilities",
        "LexiconDiscoveryResult",
        "discover_lexicons",
        "GenerationConfig",
        "LanguageRoutingConfig",
        "SynthesisConfig",
        "SynthesisInputTooLongError",
        "PyKokoroError",
        "KokoroError",
        "SynthesisError",
        "ConfigurationError",
        "InvalidRequestError",
        "EmptyTextError",
        "InvalidLanguageError",
        "InvalidVoiceError",
        "InvalidModelError",
        "InvalidPronunciationError",
        "InvalidLinguisticTokensError",
        "UnsupportedFeatureError",
        "CapabilityError",
        "SynthesisStateError",
        "AlignmentError",
        "BackendError",
        "LongTextSplitMode",
        "SynthesisIdentity",
        "build_synthesis_identity",
        "VoiceLevelConfig",
        "VoiceLevelApplication",
        "VoiceBlend",
        "SynthesisSegment",
        "SynthesisRequest",
        "RenderedSegment",
        "PronunciationOverride",
        "LinguisticToken",
        "WordTiming",
        "TokenizerConfig",
        "EspeakConfig",
        "ShortSentenceConfig",
        "KokoroSynthesizer",
        "__version__",
        "__version_tuple__",
    }
    assert expected == set(pykokoro.__all__)
    assert all(
        value is not None
        for value in (
            KokoroSynthesizer,
            SynthesisSegment,
            SynthesisRequest,
            RenderedSegment,
            SynthesisConfig,
            SynthesisIdentity,
            build_synthesis_identity,
            LongTextSplitMode,
            SynthesisInputTooLongError,
            GenerationConfig,
            VoiceLevelConfig,
            VoiceLevelApplication,
            LanguageRoutingConfig,
            PronunciationOverride,
            LinguisticToken,
            VoiceBlend,
        )
    )


def test_document_and_composition_apis_are_not_kept_as_aliases() -> None:
    removed_names = {
        "KokoroPipeline",
        "PipelineConfig",
        "build_pipeline",
        "AudioUnitResult",
        "AudioResult",
        "SSMDRenderConfig",
        "ProsodyConfig",
        "LoudnessConfig",
        "PreparedG2PAdapter",
        "PreparedSynthesis",
        "OnnxRequestRenderer",
        "PhonemeSegment",
        "Trace",
        "TraceEvent",
        "VoiceCalibrationKey",
        "VoiceLevelCalibration",
        "resolve_synthesis_config",
    }
    assert removed_names.isdisjoint(pykokoro.__all__)
    assert all(not hasattr(pykokoro, name) for name in removed_names)


def test_phoneme_segment_serialization_has_only_engine_fields() -> None:
    fields = set(inspect.signature(PhonemeSegment).parameters)
    assert not fields.intersection(
        {"sentence_idx", "paragraph_idx", "pause_before", "pause_after", "ssmd_metadata"}
    )
    segment = PhonemeSegment(
        id="request:phoneme:0",
        segment_id="request",
        phoneme_id=0,
        text="hello",
        phonemes="həlˈoʊ",
        tokens=[1, 2, 3],
    )
    payload = segment.to_dict()
    assert "engine_metadata" not in payload
    assert "ssmd_metadata" not in payload
    assert "pause_before" not in payload
    restored = PhonemeSegment.from_dict(payload)
    assert restored.to_dict() == payload


def test_rendered_segment_owns_one_mono_result() -> None:
    result = RenderedSegment(
        id="request-1",
        audio=np.zeros(4, dtype=np.float32),
        sample_rate=24_000,
        text="Hello",
        language="en-us",
        voice="af_heart",
        phonemes="həlˈoʊ",
        token_ids=(1, 2, 3),
    )
    assert result.audio.shape == (4,)
    assert result.id == "request-1"
