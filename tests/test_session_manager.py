"""Tests for OnnxSessionManager compatibility facade."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from pykokoro.exceptions import ConfigurationError
from pykokoro.onnx_session import (
    OnnxSessionManager,
    ProviderType,
    normalize_execution_provider,
    resolve_execution_provider,
)


class TestProviderSelection:
    """Test provider selection logic."""

    def test_invalid_provider_raises_error(self) -> None:
        invalid_provider = cast(ProviderType, "invalid")
        with pytest.raises(ConfigurationError, match="Requested.*invalid"):
            normalize_execution_provider(invalid_provider, available=("CPUExecutionProvider",))

    @pytest.mark.parametrize(
        ("requested", "expected"),
        [
            ("nnapi", "NnapiExecutionProvider"),
            ("NNAPI", "NnapiExecutionProvider"),
            (" nnapiexecutionprovider ", "NnapiExecutionProvider"),
            ("xnnpack", "XnnpackExecutionProvider"),
            ("XnnpackExecutionProvider", "XnnpackExecutionProvider"),
            ("cpu", "CPUExecutionProvider"),
        ],
    )
    def test_normalize_provider_aliases_and_runtime_names(
        self, requested: str, expected: str
    ) -> None:
        available = (
            "NnapiExecutionProvider",
            "XnnpackExecutionProvider",
            "CPUExecutionProvider",
        )
        assert normalize_execution_provider(requested, available=available) == expected

    @pytest.mark.parametrize("requested", ["", "unknown", "UnknownExecutionProvider"])
    def test_normalize_provider_errors_include_context(self, requested: str) -> None:
        available = ("NnapiExecutionProvider", "CPUExecutionProvider")
        with pytest.raises(ConfigurationError, match="Requested.*available providers"):
            normalize_execution_provider(requested, available=available)

    def test_resolve_provider_auto_and_legacy_gpu(self) -> None:
        available = (
            "NnapiExecutionProvider",
            "XnnpackExecutionProvider",
            "CPUExecutionProvider",
        )
        assert resolve_execution_provider("auto", available=available) == "NnapiExecutionProvider"
        assert (
            resolve_execution_provider(None, use_gpu=True, available=available)
            == "NnapiExecutionProvider"
        )
        assert resolve_execution_provider(None, available=("CPUExecutionProvider",)) == (
            "CPUExecutionProvider"
        )

    def test_resolve_provider_environment_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        available = (
            "NnapiExecutionProvider",
            "XnnpackExecutionProvider",
            "CPUExecutionProvider",
        )
        monkeypatch.setenv("ONNX_PROVIDER", "XNNPACK")
        assert resolve_execution_provider("cpu", available=available) == "XnnpackExecutionProvider"
        monkeypatch.setenv("ONNX_PROVIDER", "XnnpackExecutionProvider")
        assert resolve_execution_provider("cpu", available=available) == "XnnpackExecutionProvider"

    def test_auto_selects_xnnpack_when_nnapi_missing(self) -> None:
        available = ("XnnpackExecutionProvider", "CPUExecutionProvider")
        assert resolve_execution_provider("auto", available=available) == (
            "XnnpackExecutionProvider"
        )

    def test_unavailable_provider_raises_error(self) -> None:
        available = ("CPUExecutionProvider",)
        with pytest.raises(ConfigurationError, match="Requested.*available providers"):
            normalize_execution_provider("cuda", available=available)

    def test_env_override_invalid_raises_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ONNX_PROVIDER", "UnknownExecutionProvider")
        with pytest.raises(ConfigurationError, match="ONNX_PROVIDER"):
            resolve_execution_provider(
                "cpu", available=("CPUExecutionProvider",), respect_environment=True
            )


class TestProviderOptions:
    """Test provider options handling."""

    def test_get_default_openvino_options(self) -> None:
        manager = OnnxSessionManager(model_quality="fp32")
        defaults = manager._get_default_provider_options("OpenVINOExecutionProvider")
        assert "device_type" in defaults
        assert "cache_dir" in defaults
        assert "precision" in defaults
        assert defaults["precision"] == "FP32"

    def test_get_default_openvino_options_fp16(self) -> None:
        manager = OnnxSessionManager(model_quality="fp16")
        defaults = manager._get_default_provider_options("OpenVINOExecutionProvider")
        assert defaults["precision"] == "FP16"

    def test_get_default_cuda_options(self) -> None:
        manager = OnnxSessionManager()
        defaults = manager._get_default_provider_options("CUDAExecutionProvider")
        assert "device_id" in defaults
        assert defaults["device_id"] == "0"

    def test_get_default_directml_options(self) -> None:
        manager = OnnxSessionManager()
        defaults = manager._get_default_provider_options("DmlExecutionProvider")
        assert "device_id" in defaults
        assert defaults["device_id"] == "0"

    def test_get_provider_specific_options_passes_through(self) -> None:
        manager = OnnxSessionManager()
        all_options = {"device_id": "1", "num_threads": 4}
        opts = manager._get_provider_specific_options("CUDAExecutionProvider", all_options)
        assert opts["device_id"] == "1"
        assert opts["num_threads"] == "4"

    def test_get_provider_specific_options_converts_to_string(self) -> None:
        manager = OnnxSessionManager()
        opts = manager._get_provider_specific_options("CUDAExecutionProvider", {"device_id": 0})
        assert opts["device_id"] == "0"
        assert isinstance(opts["device_id"], str)

    @pytest.mark.parametrize("provider", ["NnapiExecutionProvider", "XnnpackExecutionProvider"])
    def test_platform_provider_has_no_default_options(self, provider: str) -> None:
        manager = OnnxSessionManager()
        assert manager._get_default_provider_options(provider) == {}


class TestSessionOptions:
    """Test session options creation."""

    def test_create_session_options_returns_none_by_default(self) -> None:
        manager = OnnxSessionManager()
        assert manager._create_session_options() is None

    def test_create_session_options_returns_user_provided(self) -> None:
        user_opts = MagicMock()
        manager = OnnxSessionManager(session_options=user_opts)
        assert manager._create_session_options() is user_opts


class TestSessionCreation:
    """Test OnnxVoice delegation."""

    def test_create_session_delegates_to_onnxvoice(self, tmp_path) -> None:
        model_path = tmp_path / "model.onnx"
        mock_session = MagicMock()
        with patch("onnxvoice.OnnxSession", return_value=mock_session):
            manager = OnnxSessionManager(provider="cpu")
            result = manager.create_session(model_path)
            assert result is mock_session


class TestPublicAPI:
    """Test public API compliance."""

    def test_manager_has_required_public_names(self) -> None:
        from pykokoro import onnx_session

        assert hasattr(onnx_session, "OnnxSessionManager")
        assert hasattr(onnx_session, "get_available_execution_providers")
        assert hasattr(onnx_session, "normalize_execution_provider")
        assert hasattr(onnx_session, "resolve_execution_provider")

    def test_manager_init_defaults(self) -> None:
        manager = OnnxSessionManager()
        assert manager._use_gpu is False
        assert manager._provider is None
        assert manager._session_options is None
        assert manager._provider_options == {}
        assert manager._model_quality == "fp32"

    def test_manager_init_with_parameters(self) -> None:
        prov_opts = {"device_id": "0"}
        manager = OnnxSessionManager(
            use_gpu=True,
            provider="cuda",
            session_options=MagicMock(),
            provider_options=prov_opts,
            model_quality="fp16",
        )
        assert manager._use_gpu is True
        assert manager._provider == "cuda"
        assert manager._provider_options == prov_opts
        assert manager._model_quality == "fp16"
