from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import pykokoro.discovery as discovery
from pykokoro.model_registry import ModelRegistry


class RegistryClientStub:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self.calls: list[dict[str, bool]] = []

    def load(self, *, offline: bool = False, refresh: bool = False) -> ModelRegistry:
        self.calls.append({"offline": offline, "refresh": refresh})
        return self.registry


def _distribution(distribution_id: str = "github") -> dict:
    return {
        "id": distribution_id,
        "provider": "github-release" if distribution_id == "github" else "huggingface",
        "transport": "https",
        "runtime_ready": True,
        "artifacts": [
            {
                "id": f"{distribution_id}-model",
                "role": "model",
                "format": "onnx",
                "url": "https://example.test/model.onnx",
                "local_name": "model.onnx",
                "size": 1,
                "sha256": "0" * 64,
                "quality": "fp32",
            },
            {
                "id": f"{distribution_id}-voices",
                "role": "voices",
                "format": "numpy-npz",
                "url": "https://example.test/voices.npz",
                "local_name": "voices.npz",
                "size": 1,
                "sha256": "0" * 64,
            },
        ],
    }


def _model(
    model_id: str,
    *,
    frontend: str,
    languages: list[str],
    voices: list[str],
    default_voice: str,
    runtime_available: bool = True,
    redistribution_allowed: bool = True,
    distributions: list[dict] | None = None,
    source: str = "github",
) -> dict:
    return {
        "runtime": {
            "frontend": frontend,
            "language_codes": languages,
            "voices": voices,
            "default_voice": default_voice,
            "layout": "single-onnx-v1",
            "sample_rate": 24000,
            "max_tokens": 510,
        },
        "runtime_available": runtime_available,
        "redistribution_allowed": redistribution_allowed,
        "model_source": source,
        "distributions": distributions if distributions is not None else [_distribution()],
    }


def _registry() -> ModelRegistry:
    return ModelRegistry(
        {
            "schema": 1,
            "runtime_contract": 1,
            "models": {
                "z-unsupported": _model(
                    "z-unsupported",
                    frontend="future-frontend-v2",
                    languages=["EN_us"],
                    voices=["future"],
                    default_voice="future",
                ),
                "de-crane": _model(
                    "de-crane",
                    frontend="german-ipa-v1",
                    languages=["de_DE"],
                    voices=["df_kerstin"],
                    default_voice="df_kerstin",
                ),
                "de-thorsten": _model(
                    "de-thorsten",
                    frontend="kokorog2p-de-thorsten-v1",
                    languages=["DE", "de_AT"],
                    voices=["thorsten"],
                    default_voice="thorsten",
                ),
                "restricted": _model(
                    "restricted",
                    frontend="pykokoro-native-v1",
                    languages=["en"],
                    voices=["voice"],
                    default_voice="voice",
                    redistribution_allowed=False,
                ),
                "unavailable": _model(
                    "unavailable",
                    frontend="pykokoro-native-v1",
                    languages=["fr"],
                    voices=["voice"],
                    default_voice="voice",
                    runtime_available=False,
                    distributions=[],
                ),
            },
        },
        "fixture-cache",
    )


def _use_fixture(monkeypatch: pytest.MonkeyPatch) -> RegistryClientStub:
    client = RegistryClientStub(_registry())
    monkeypatch.setattr(discovery, "RegistryClient", lambda: client)
    return client


def test_discover_models_is_public() -> None:
    from pykokoro import ModelCapabilities, ModelDiscoveryResult, discover_models

    assert callable(discover_models)
    assert ModelCapabilities.__module__ == "pykokoro.discovery"
    assert ModelDiscoveryResult.__module__ == "pykokoro.discovery"


def test_discovery_returns_complete_sorted_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_fixture(monkeypatch)

    result = discovery.discover_models(offline=True)
    assert [model.model_id for model in result.models] == [
        "de-crane",
        "de-thorsten",
        "restricted",
        "unavailable",
        "z-unsupported",
    ]
    assert result.registry_source == "fixture-cache"
    assert result.cache_fallback is False
    assert result.offline is True

    thorsten = next(model for model in result.models if model.model_id == "de-thorsten")
    assert thorsten.source == "github"
    assert thorsten.languages == ("de", "de-at")
    assert thorsten.default_voice == "thorsten"
    assert thorsten.voices == ("thorsten",)
    assert thorsten.qualities == ("fp32",)
    assert thorsten.g2p_backend == "kokorog2p"
    assert thorsten.lexicons == ("gold", "crane")
    assert thorsten.frontend == "kokorog2p-de-thorsten-v1"
    assert thorsten.status == "ready"
    assert thorsten.experimental is False
    assert thorsten.runtime_available is True
    assert thorsten.redistribution_allowed is True
    assert thorsten.distribution_id == "github"
    assert thorsten.provider == "github-release"
    assert thorsten.sample_rate == 24000
    assert thorsten.max_tokens == 510

    crane = next(model for model in result.models if model.model_id == "de-crane")
    assert crane.voices == ("default",)
    assert crane.default_voice == "default"
    assert crane.status == "experimental"
    assert crane.experimental is True

    assert (
        next(model for model in result.models if model.model_id == "restricted").status
        == "restricted"
    )
    assert next(model for model in result.models if model.model_id == "unavailable").status == (
        "registry-unavailable"
    )
    assert next(model for model in result.models if model.model_id == "z-unsupported").status == (
        "unsupported-frontend"
    )

    with pytest.raises(FrozenInstanceError):
        thorsten.status = "changed"  # type: ignore[misc]


def test_discovery_uses_selected_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _registry()
    registry.data["models"]["de-thorsten"]["distributions"] = [
        _distribution("github"),
        _distribution("hf"),
    ]
    client = RegistryClientStub(registry)
    monkeypatch.setattr(discovery, "RegistryClient", lambda: client)

    model = next(
        item
        for item in discovery.discover_models(preference="huggingface").models
        if item.model_id == "de-thorsten"
    )
    assert model.source == "huggingface"
    assert model.provider == "huggingface"
    assert model.distribution_id == "hf"


def test_offline_refresh_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _use_fixture(monkeypatch)

    with pytest.raises(ValueError, match="offline and refresh cannot be combined"):
        discovery.discover_models(offline=True, refresh=True)
    assert client.calls == []


def test_refresh_provenance_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _use_fixture(monkeypatch)

    result = discovery.discover_models(refresh=True)

    assert client.calls == [{"offline": False, "refresh": True}]
    assert result.refreshed is True


def test_discovery_never_downloads_assets_or_imports_onnx(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_fixture(monkeypatch)
    monkeypatch.setattr(
        "pykokoro.model_registry.download_artifact",
        lambda *args, **kwargs: pytest.fail("discovery downloaded an artifact"),
    )

    result = discovery.discover_models(offline=True)

    assert result.models
    assert "onnxruntime" not in discovery.__dict__
