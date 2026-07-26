"""Tests for OnnxSessionManager."""

import logging
from typing import cast
from unittest.mock import MagicMock, patch

import onnxruntime as rt
import pytest

from pykokoro.exceptions import ConfigurationError
from pykokoro.onnx_session import OnnxSessionManager, ProviderType


@pytest.fixture
def mock_model_path(tmp_path):
    """Create a mock model path (doesn't need to exist for testing)."""
    return tmp_path / "model.onnx"


class TestOnnxSessionManagerInit:
    """Test OnnxSessionManager initialization."""

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        manager = OnnxSessionManager()
        assert manager._use_gpu is False
        assert manager._provider is None
        assert manager._session_options is None
        assert manager._provider_options == {}
        assert manager._model_quality == "fp32"

    def test_init_with_parameters(self):
        """Test initialization with custom parameters."""
        sess_opts = rt.SessionOptions()
        prov_opts = {"device_id": "0"}

        manager = OnnxSessionManager(
            use_gpu=True,
            provider="cuda",
            session_options=sess_opts,
            provider_options=prov_opts,
            model_quality="fp16",
        )

        assert manager._use_gpu is True
        assert manager._provider == "cuda"
        assert manager._session_options == sess_opts
        assert manager._provider_options == prov_opts
        assert manager._model_quality == "fp16"


class TestProviderSelection:
    """Test provider selection logic."""

    def test_select_cpu_provider(self):
        """Test CPU provider selection."""
        manager = OnnxSessionManager(provider="cpu")
        providers = manager._select_providers("cpu", False)
        assert "CPUExecutionProvider" in providers

    def test_select_auto_provider_cpu(self):
        """Test auto provider defaults to CPU when no GPU."""
        manager = OnnxSessionManager(provider="auto")
        providers = manager._select_providers("auto", False)
        # Should select best available or CPU
        assert isinstance(providers, list)
        assert len(providers) > 0

    def test_select_providers_with_use_gpu_legacy(self):
        """Test legacy use_gpu parameter."""
        manager = OnnxSessionManager(use_gpu=True)
        providers = manager._select_providers(None, True)
        # Should auto-select (could be CPU if no GPU available)
        assert isinstance(providers, list)

    def test_invalid_provider_raises_error(self):
        """Test that invalid provider raises ValueError."""
        invalid_provider = cast(ProviderType, "invalid")
        manager = OnnxSessionManager(provider=invalid_provider)
        with pytest.raises(ValueError, match="Unknown provider"):
            manager._select_providers(invalid_provider, False)

    def test_unavailable_provider_raises_error(self):
        """Test that unavailable provider raises RuntimeError."""
        available = rt.get_available_providers()

        # Find a provider that's NOT available
        test_providers = ["cuda", "openvino", "directml", "coreml"]

        for prov in test_providers:
            provider_map = {
                "cuda": "CUDAExecutionProvider",
                "openvino": "OpenVINOExecutionProvider",
                "directml": "DmlExecutionProvider",
                "coreml": "CoreMLExecutionProvider",
            }

            if provider_map[prov] not in available:
                provider_value = cast(ProviderType, prov)
                manager = OnnxSessionManager(provider=provider_value)
                with pytest.raises(RuntimeError, match="provider requested but not available"):
                    manager._select_providers(provider_value, False)
                return

        pytest.skip("All providers are available on this system")

    def test_env_override(self, monkeypatch):
        """Test ONNX_PROVIDER environment variable override."""
        monkeypatch.setenv("ONNX_PROVIDER", "CPUExecutionProvider")
        manager = OnnxSessionManager(provider="cpu")
        providers = manager._select_providers("cpu", False)
        assert "CPUExecutionProvider" in providers
        assert len(providers) == 1

    def test_env_override_invalid(self, monkeypatch):
        """Test invalid ONNX_PROVIDER override raises error."""
        monkeypatch.setenv("ONNX_PROVIDER", "UnknownExecutionProvider")
        manager = OnnxSessionManager(provider="cpu")

        with pytest.raises(ConfigurationError, match="ONNX_PROVIDER"):
            manager._select_providers("cpu", False)


class TestProviderOptions:
    """Test provider options handling."""

    def test_get_default_openvino_options(self, tmp_path):
        """Test default options for OpenVINO provider."""
        manager = OnnxSessionManager(model_quality="fp32")
        defaults = manager._get_default_provider_options("OpenVINOExecutionProvider")

        assert "device_type" in defaults
        assert "cache_dir" in defaults
        assert "precision" in defaults
        assert defaults["precision"] == "FP32"

    def test_get_default_openvino_options_fp16(self):
        """Test default OpenVINO options with FP16 model."""
        manager = OnnxSessionManager(model_quality="fp16")
        defaults = manager._get_default_provider_options("OpenVINOExecutionProvider")

        assert defaults["precision"] == "FP16"

    def test_get_default_cuda_options(self):
        """Test default options for CUDA provider."""
        manager = OnnxSessionManager()
        defaults = manager._get_default_provider_options("CUDAExecutionProvider")

        assert "device_id" in defaults
        assert "arena_extend_strategy" in defaults
        assert defaults["device_id"] == "0"

    def test_get_default_directml_options(self):
        """Test default options for DirectML provider."""
        manager = OnnxSessionManager()
        defaults = manager._get_default_provider_options("DmlExecutionProvider")

        assert "device_id" in defaults
        assert defaults["device_id"] == "0"

    def test_get_provider_specific_options(self):
        """Test extracting provider-specific options."""
        manager = OnnxSessionManager()
        all_options = {
            "device_id": "1",
            "gpu_mem_limit": "2GB",
            "num_threads": 4,  # SessionOptions attribute, should be filtered
        }

        cuda_opts = manager._get_provider_specific_options("CUDAExecutionProvider", all_options)

        assert "device_id" in cuda_opts
        assert "gpu_mem_limit" in cuda_opts
        assert "num_threads" not in cuda_opts  # Filtered out
        assert cuda_opts["device_id"] == "1"

    def test_get_provider_specific_options_converts_to_string(self):
        """Test that provider options are converted to strings."""
        manager = OnnxSessionManager()
        all_options = {"device_id": 0}  # Integer value

        opts = manager._get_provider_specific_options("CUDAExecutionProvider", all_options)

        assert opts["device_id"] == "0"  # Should be string
        assert isinstance(opts["device_id"], str)

    def test_unknown_option_warning(self, caplog):
        """Test that unknown options generate a warning."""
        caplog.set_level(logging.WARNING)
        manager = OnnxSessionManager()
        all_options = {"unknown_option": "value"}

        manager._get_provider_specific_options("CUDAExecutionProvider", all_options)

        assert any("Unknown option" in record.message for record in caplog.records)


class TestSessionOptions:
    """Test session options creation."""

    def test_create_session_options_defaults(self):
        """Test creating SessionOptions with defaults."""
        manager = OnnxSessionManager()
        sess_opt = manager._create_session_options()

        assert isinstance(sess_opt, rt.SessionOptions)
        assert sess_opt.graph_optimization_level == rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        assert sess_opt.execution_mode == rt.ExecutionMode.ORT_SEQUENTIAL

    def test_create_session_options_user_provided(self):
        """Test using user-provided SessionOptions."""
        user_opts = rt.SessionOptions()
        user_opts.intra_op_num_threads = 8

        manager = OnnxSessionManager(session_options=user_opts)
        sess_opt = manager._create_session_options()

        assert sess_opt == user_opts
        assert sess_opt.intra_op_num_threads == 8

    def test_apply_provider_options_to_session(self):
        """Test applying provider options to SessionOptions."""
        manager = OnnxSessionManager()
        sess_opt = rt.SessionOptions()
        options = {
            "num_threads": 4,
            "enable_profiling": True,
        }

        manager._apply_provider_options(sess_opt, options)

        assert sess_opt.intra_op_num_threads == 4
        assert sess_opt.enable_profiling is True

    def test_apply_provider_options_aliases(self):
        """Test that option aliases work correctly."""
        manager = OnnxSessionManager()
        sess_opt = rt.SessionOptions()

        # Test 'threads' alias for 'intra_op_num_threads'
        manager._apply_provider_options(sess_opt, {"threads": 2})
        assert sess_opt.intra_op_num_threads == 2

        # Test 'num_threads' alias
        manager._apply_provider_options(sess_opt, {"num_threads": 4})
        assert sess_opt.intra_op_num_threads == 4


class TestSessionCreation:
    """Test ONNX session creation."""

    @pytest.mark.skipif(
        "CPUExecutionProvider" not in rt.get_available_providers(),
        reason="CPU provider not available",
    )
    def test_create_session_cpu(self, tmp_path):
        """Test creating a session with CPU provider (requires valid model)."""
        # This test would need an actual ONNX model file
        # For now, we'll test the logic with mocking
        pytest.skip("Requires actual ONNX model file")

    def test_create_session_with_fallback(self, tmp_path):
        """Test session creation with fallback enabled."""
        # Mock the session creation
        with patch("onnxruntime.InferenceSession") as mock_session:
            mock_instance = MagicMock()
            mock_instance.get_providers.return_value = ["CPUExecutionProvider"]
            mock_session.return_value = mock_instance

            manager = OnnxSessionManager(provider="cpu")
            model_path = tmp_path / "model.onnx"
            model_path.touch()  # Create empty file

            session = manager.create_session(model_path, allow_fallback=True)

            assert session is not None
            assert mock_session.called

    def test_create_session_fallback_disabled(self, tmp_path):
        """Test that fallback can be disabled."""
        with patch("onnxruntime.InferenceSession") as mock_session:
            # Simulate failure on first attempt
            mock_session.side_effect = [Exception("Provider failed"), MagicMock()]

            manager = OnnxSessionManager(provider="cuda")
            model_path = tmp_path / "model.onnx"
            model_path.touch()

            # Accept either error: provider not available or session creation failed
            with pytest.raises(
                RuntimeError,
                match="(Failed to create ONNX session|CUDA provider requested but not available)",
            ):
                manager.create_session(model_path, allow_fallback=False)

    def test_create_session_logs_fallback(self, caplog, tmp_path):
        """Test that fallback is logged."""
        caplog.set_level(logging.WARNING)

        with patch("pykokoro.onnx_session.rt.InferenceSession") as mock_session:
            # First call (with non-CPU provider) fails,
            # second call (CPU fallback) succeeds
            mock_instance = MagicMock()
            mock_instance.get_providers.return_value = ["CPUExecutionProvider"]
            mock_session.side_effect = [
                Exception("Provider failed"),
                mock_instance,
            ]

            manager = OnnxSessionManager(provider="auto")

            # Mock _select_providers to return a non-CPU provider first
            # This ensures fallback happens even in CI where auto might be CPU-only
            with patch.object(
                manager,
                "_select_providers",
                return_value=[
                    ("CUDAExecutionProvider", {}),
                    "CPUExecutionProvider",
                ],
            ):
                model_path = tmp_path / "model.onnx"
                model_path.touch()

                session = manager.create_session(model_path, allow_fallback=True)

                assert session is not None
                # Should have warning about fallback
                assert any("fell back to" in record.message for record in caplog.records)


class TestProviderMergeLogic:
    """Test provider option merging logic."""

    def test_user_options_override_defaults(self):
        """Test that user options override default options."""
        # Skip if CUDA is not available
        if "CUDAExecutionProvider" not in rt.get_available_providers():
            pytest.skip("CUDA provider not available")

        manager = OnnxSessionManager(
            provider="cuda",
            provider_options={"device_id": "2"},
        )

        providers = manager._select_providers("cuda", False)

        # Check that the provider list contains user options
        # Find the CUDA provider entry
        for prov in providers:
            if isinstance(prov, tuple) and prov[0] == "CUDAExecutionProvider":
                assert prov[1]["device_id"] == "2"
                break

    def test_defaults_used_when_no_user_options(self):
        """Test that defaults are used when no user options provided."""
        # Skip if CUDA is not available
        if "CUDAExecutionProvider" not in rt.get_available_providers():
            pytest.skip("CUDA provider not available")

        manager = OnnxSessionManager(provider="cuda")

        providers = manager._select_providers("cuda", False)

        # Should have default options
        for prov in providers:
            if isinstance(prov, tuple) and prov[0] == "CUDAExecutionProvider":
                assert "device_id" in prov[1]
                assert prov[1]["device_id"] == "0"  # Default
                break


class TestLogging:
    """Test logging behavior."""

    def test_provider_selection_logged(self, caplog):
        """Test that provider selection is logged."""
        caplog.set_level(logging.INFO)
        manager = OnnxSessionManager(provider="cpu")
        manager._select_providers("cpu", False)

        assert any("CPU" in record.message for record in caplog.records)

    def test_auto_selection_logged(self, caplog):
        """Test that auto-selection is logged."""
        caplog.set_level(logging.INFO)
        manager = OnnxSessionManager(provider="auto")
        manager._select_providers("auto", False)

        assert any("Auto" in record.message or "CPU" in record.message for record in caplog.records)

    def test_session_options_application_logged(self, caplog):
        """Test that applying session options is logged."""
        caplog.set_level(logging.INFO)
        manager = OnnxSessionManager(provider_options={"num_threads": 4})
        manager._create_session_options()

        assert any("Applying provider options" in record.message for record in caplog.records)
