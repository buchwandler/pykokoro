from __future__ import annotations

import json

import numpy as np
import pytest
from audiosig import apply_gain_db

from pykokoro import LoudnessConfig
from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.stages.audio_generation.onnx import OnnxAudioGenerationAdapter
from pykokoro.types import PhonemeSegment, Trace
from pykokoro.voice_level import (
    VoiceCalibrationCatalog,
    VoiceCalibrationKey,
    VoiceLevelCalibration,
    apply_voice_level_calibration,
    default_voice_calibration,
    load_voice_calibrations,
)


def _key() -> VoiceCalibrationKey:
    return VoiceCalibrationKey("github", "v1.0", "fp32", "af_bella")


def _catalog(gain_db: float = -2.0) -> VoiceCalibrationCatalog:
    key = _key()
    return VoiceCalibrationCatalog(
        schema=1,
        method="bs1770",
        corpus="test",
        reference_lufs=-18.0,
        generated_with={},
        voices={key: VoiceLevelCalibration(gain_db=gain_db, samples=8)},
    )


def test_fixed_calibration_gain_and_disabled_mode() -> None:
    audio = np.ones(8, dtype=np.float32)
    key = _key()
    np.testing.assert_array_equal(
        apply_voice_level_calibration(
            audio, LoudnessConfig(voice_leveling="calibrated"), key, catalog=_catalog()
        ),
        apply_gain_db(audio, -2.0),
    )
    np.testing.assert_array_equal(
        apply_voice_level_calibration(audio, LoudnessConfig(), key, catalog=_catalog()), audio
    )


def test_override_replaces_registry_gain_and_missing_is_safe() -> None:
    audio = np.ones(8, dtype=np.float32)
    config = LoudnessConfig(voice_leveling="calibrated", voice_gain_db=3.0)
    np.testing.assert_array_equal(
        apply_voice_level_calibration(audio, config, _key(), catalog=_catalog()),
        apply_gain_db(audio, 3.0),
    )
    np.testing.assert_array_equal(
        apply_voice_level_calibration(
            audio, LoudnessConfig(voice_leveling="calibrated"), None, catalog=_catalog()
        ),
        audio,
    )


def test_external_audio_bypasses_voice_calibration() -> None:
    audio = np.ones(8, dtype=np.float32)
    config = LoudnessConfig(voice_leveling="calibrated", voice_gain_db=3.0)
    np.testing.assert_array_equal(
        apply_voice_level_calibration(
            audio, config, _key(), external_audio=True, catalog=_catalog()
        ),
        audio,
    )


def test_calibration_trace_is_static_metadata() -> None:
    trace = Trace()
    apply_voice_level_calibration(
        np.ones(4, dtype=np.float32),
        LoudnessConfig(voice_leveling="calibrated"),
        _key(),
        catalog=_catalog(),
        trace=trace,
        segment_id="s1",
    )
    assert trace.model["voice_leveling"][0]["gain_db"] == -2.0
    assert trace.model["voice_leveling"][0]["segment_id"] == "s1"

    assert trace.model["voice_leveling"][0]["reason"] == "calibrated"


def test_calibration_trace_distinguishes_disabled_and_missing() -> None:
    audio = np.ones(4, dtype=np.float32)
    disabled_trace = Trace()
    apply_voice_level_calibration(audio, LoudnessConfig(), _key(), trace=disabled_trace)
    assert disabled_trace.model["voice_leveling"][0]["reason"] == "disabled"
    missing_trace = Trace()
    apply_voice_level_calibration(
        audio,
        LoudnessConfig(voice_leveling="calibrated"),
        VoiceCalibrationKey("github", "v1.0", "fp16", "af_bella"),
        catalog=_catalog(),
        trace=missing_trace,
    )
    assert missing_trace.model["voice_leveling"][0]["reason"] == "calibration_not_found"


def test_calibration_schema_validation(tmp_path) -> None:
    path = tmp_path / "calibration.json"
    data = {
        "schema": 1,
        "method": "bs1770",
        "corpus": "test",
        "reference_lufs": -18.0,
        "generated_with": {},
        "voices": {str(_key()): {"gain_db": -2.0, "samples": 8}},
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_voice_calibrations(path).voices[_key()].gain_db == -2.0
    data["schema"] = 2
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_voice_calibrations(path)


def test_segment_retains_render_key() -> None:
    segment = PhonemeSegment(
        id="s",
        segment_id="s",
        phoneme_id=0,
        text="x",
        phonemes="x",
        tokens=[],
        render_voice_key=_key(),
    )
    restored = PhonemeSegment.from_dict(segment.to_dict())
    assert restored.render_voice_key == _key()


class _GenerationBackend:
    def get_voice_style(self, voice_name):
        return np.zeros(256, dtype=np.float32)

    def resolve_voice_style(self, voice):
        return np.zeros(256, dtype=np.float32)

    def generate_raw_audio_segments(
        self, segments, voice_style, speed, voice_resolver, *, default_voice_name=None, trace=None
    ):
        return segments


def test_generation_adapter_records_default_render_identity() -> None:
    segment = PhonemeSegment(
        id="s", segment_id="s", phoneme_id=0, text="x", phonemes="x", tokens=[]
    )
    config = PipelineConfig(
        voice="af_bella",
        generation=GenerationConfig(lang="en-us"),
        model_source="github",
        model_variant="v1.0",
        model_quality="fp32",
    )
    OnnxAudioGenerationAdapter(_GenerationBackend()).generate([segment], config, Trace())
    assert segment.render_voice_key == _key()


def test_packaged_calibration_catalog_has_reviewed_fp32_records() -> None:
    catalog = default_voice_calibration()
    assert catalog.schema == 1
    assert catalog.method == "bs1770"
    assert catalog.corpus == "pykokoro-count-1-to-10-v2"
    assert catalog.reference_lufs == -24.0
    assert len(catalog.voices) == 216
    assert all(key.quality == "fp32" for key in catalog.voices)
    assert all(record.samples == 3 for record in catalog.voices.values())
    assert all(record.method == "bs1770" for record in catalog.voices.values())
    assert all(
        record.corpus_version == "pykokoro-count-1-to-10-v2" for record in catalog.voices.values()
    )


def test_packaged_calibration_has_exact_values_and_identity_boundaries() -> None:
    catalog = default_voice_calibration()
    examples = {
        "github:v1.0:fp32:af_bella": (-22.35060810577404, -1.6493918942259604),
        "github:ru-zaakirio-base:fp32:sveta": (-30.08524955204526, 6.0852495520452585),
        "github:vi-anphunl:fp32:ngoc_huyen": (-12.768391563685354, -11.231608436314646),
    }
    for raw_key, (measured_lufs, gain_db) in examples.items():
        record = catalog.voices[VoiceCalibrationKey.parse(raw_key)]
        assert record.measured_lufs == measured_lufs
        assert record.gain_db == gain_db
    assert VoiceCalibrationKey.parse("github:sv-joakim:fp32:Alice") not in catalog.voices
    assert VoiceCalibrationKey.parse("github:v1.0:fp16:af_bella") not in catalog.voices
