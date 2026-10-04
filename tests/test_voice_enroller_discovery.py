from __future__ import annotations

import pytest

from pykokoro import discovery
from pykokoro.exceptions import InvalidModelError, UnsupportedFeatureError
from pykokoro.model_profiles import resolve_voice_enroller
from pykokoro.model_registry import ModelRegistry, VoiceEnrollerSpec


class RegistryClientStub:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def load(self, *, offline: bool = False, refresh: bool = False) -> ModelRegistry:
        return self.registry


def test_discovery_exposes_voice_enroller_capabilities(monkeypatch) -> None:
    enroller = {
        "id": "inno-v0.2",
        "kind": "zero-shot-tuner",
        "transcript_required": False,
        "min_seconds": 3.0,
        "max_seconds": 30.0,
        "recommended_seconds": 10.0,
        "output_format": "kokoro-voicepack-v1",
    }
    registry = ModelRegistry(
        {
            "schema": 1,
            "runtime_contract": 1,
            "models": {
                "v1.0": {
                    "runtime": {
                        "voice_mode": "static",
                        "voices": ["af_heart"],
                        "default_voice": "af_heart",
                        "frontend": "pykokoro-native-v1",
                        "language_codes": ["en"],
                        "voice_enrollers": [enroller],
                    },
                    "runtime_available": False,
                    "model_source": "github",
                    "distributions": [],
                },
                "en-akinvox-cloning-v1": {
                    "runtime": {"voice_mode": "reference", "voices": []},
                    "runtime_available": False,
                    "model_source": "github",
                    "distributions": [],
                },
            },
        },
        "fixture",
    )
    monkeypatch.setattr(discovery, "RegistryClient", lambda: RegistryClientStub(registry))

    result = discovery.discover_models(offline=True)
    inno_model = next(model for model in result.models if model.model_id == "v1.0")
    akinvox_model = next(
        model for model in result.models if model.model_id == "en-akinvox-cloning-v1"
    )

    assert inno_model.voice_enrollers == (
        VoiceEnrollerSpec(
            id="inno-v0.2",
            kind="zero-shot-tuner",
            transcript_required=False,
            min_seconds=3.0,
            max_seconds=30.0,
            recommended_seconds=10.0,
            output_format="kokoro-voicepack-v1",
        ),
    )
    assert inno_model.supports_voice_enrollment is True
    assert akinvox_model.voice_enrollers == ()
    assert akinvox_model.supports_reference_enrollment is True
    assert akinvox_model.supports_voice_enrollment is True


def test_inno_model_resolution_defaults_to_v1_and_checks_capability() -> None:
    enroller = {
        "id": "inno-v0.2",
        "kind": "zero-shot-tuner",
        "transcript_required": False,
        "min_seconds": 3.0,
        "max_seconds": 30.0,
        "recommended_seconds": 10.0,
        "output_format": "kokoro-voicepack-v1",
    }
    registry = ModelRegistry(
        {
            "schema": 1,
            "runtime_contract": 1,
            "models": {
                "v1.0": {
                    "runtime": {"voice_enrollers": [enroller]},
                    "distributions": [],
                    "runtime_available": False,
                },
                "other-model": {
                    "runtime": {"voice_enrollers": []},
                    "distributions": [],
                    "runtime_available": False,
                },
            },
        },
        "fixture",
    )

    model_id, spec = resolve_voice_enroller(None, "inno-v0.2", registry=registry)
    assert model_id == "v1.0"
    assert spec.id == "inno-v0.2"
    assert resolve_voice_enroller("v1.0", "inno-v0.2", registry=registry)[0] == "v1.0"
    with pytest.raises(UnsupportedFeatureError, match="does not support voice enroller"):
        resolve_voice_enroller("other-model", "inno-v0.2", registry=registry)
    with pytest.raises(InvalidModelError, match="Unknown enrollment model"):
        resolve_voice_enroller("missing-model", "inno-v0.2", registry=registry)
