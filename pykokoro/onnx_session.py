"""ONNX Runtime session management for PyKokoro."""

import logging
import os
from pathlib import Path
from typing import Any, Literal

import onnxruntime as rt

from .exceptions import ConfigurationError
from .provider_config import ProviderConfigManager
from .utils import get_user_cache_path

logger = logging.getLogger(__name__)

# Provider type
ProviderType = Literal["auto", "cpu", "cuda", "openvino", "directml", "coreml"]

# Model quality type
ModelQuality = Literal[
    "fp32", "fp16", "fp16-gpu", "q8", "q8f16", "q4", "q4f16", "uint8", "uint8f16"
]


class OnnxSessionManager:
    """Manages ONNX Runtime session creation and provider selection.

    This class handles:
    - Execution provider selection (CPU, CUDA, OpenVINO, DirectML, CoreML)
    - Session options configuration
    - Provider-specific options
    - Automatic fallback to CPU when requested provider fails

    Args:
        use_gpu: Legacy flag for GPU usage (for backward compatibility)
        provider: Execution provider to use ('auto', 'cpu', 'cuda', etc.)
        session_options: Pre-configured SessionOptions (overrides other options)
        provider_options: Provider-specific configuration options
        model_quality: Model quality affects default provider settings
    """

    def __init__(
        self,
        use_gpu: bool = False,
        provider: ProviderType | None = None,
        session_options: rt.SessionOptions | None = None,
        provider_options: dict[str, Any] | None = None,
        model_quality: ModelQuality = "fp32",
    ):
        """Initialize the session manager."""
        self._use_gpu = use_gpu
        self._provider: ProviderType | None = provider
        self._session_options = session_options
        self._provider_options = provider_options or {}
        self._model_quality: ModelQuality = model_quality

    def create_session(
        self,
        model_path: Path,
        allow_fallback: bool = True,
    ) -> rt.InferenceSession:
        """Create an ONNX Runtime inference session.

        Args:
            model_path: Path to the ONNX model file
            allow_fallback: If True, fallback to CPU if primary provider fails

        Returns:
            Configured InferenceSession

        Raises:
            RuntimeError: If session creation fails
        """
        sess_options = self._create_session_options()
        providers = self._select_providers(self._provider, self._use_gpu)

        # Try to load ONNX model with automatic fallback
        last_error = None

        for attempt, provider_list in enumerate([providers, ["CPUExecutionProvider"]]):
            # Skip second attempt if we already tried CPU or
            # if explicit provider was requested
            if attempt == 1:
                if not allow_fallback:
                    break  # User disabled fallback
                if providers == ["CPUExecutionProvider"]:
                    break  # Already tried CPU
                if self._provider and self._provider != "auto":
                    break  # User explicitly requested a provider, don't fallback
                # Check if primary provider was already CPU (handle both str and tuple)
                primary_provider = providers[0]
                if isinstance(primary_provider, tuple):
                    primary_provider = primary_provider[0]
                if primary_provider == "CPUExecutionProvider":
                    break  # Primary was already CPU

            try:
                session = rt.InferenceSession(
                    str(model_path),
                    sess_options=sess_options,
                    providers=provider_list,
                )

                # Log what was actually loaded
                actual_providers = session.get_providers()
                logger.info(f"Loaded ONNX session with providers: {actual_providers}")

                # Warn if we had to fallback
                if attempt == 1:
                    logger.warning(
                        f"Primary provider {providers[0]} failed, "
                        f"fell back to CPUExecutionProvider"
                    )

                return session

            except Exception as e:
                last_error = e
                if attempt == 0:
                    logger.warning(
                        f"Failed to load with {provider_list}: {e}. "
                        f"Will try CPU fallback..."
                    )

        # If we get here, all attempts failed
        if last_error:
            raise RuntimeError(
                f"Failed to create ONNX session with {providers}: {last_error}"
            ) from last_error
        else:
            raise RuntimeError(f"Failed to create ONNX session with {providers}")

    def _create_session_options(self) -> rt.SessionOptions:
        """Create SessionOptions with user configuration and sensible defaults.

        Priority:
        1. User-provided SessionOptions object (self._session_options)
        2. User-provided provider_options dict (self._provider_options)
        3. Sensible defaults

        Returns:
            Configured SessionOptions instance
        """
        # If user provided a SessionOptions object, use it directly
        if self._session_options is not None:
            logger.info("Using user-provided SessionOptions")
            return self._session_options

        # Create new SessionOptions with defaults
        sess_opt = rt.SessionOptions()

        # Sensible defaults - let ONNX Runtime decide thread count
        # Only set these if user doesn't override
        sess_opt.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opt.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL

        # Apply user provider_options if provided
        if self._provider_options:
            logger.info(f"Applying provider options: {self._provider_options}")
            self._apply_provider_options(sess_opt, self._provider_options)

        return sess_opt

    def _select_providers(
        self,
        provider: ProviderType | None,
        use_gpu: bool,
    ) -> list[str | tuple[str, dict[str, str]]]:
        """Select ONNX Runtime execution providers based on preference.

        Args:
            provider: Explicit provider ('auto', 'cpu', 'cuda', 'openvino', etc.)
            use_gpu: Legacy GPU flag (for backward compatibility)

        Returns:
            List of providers in priority order. Can be simple strings or
            tuples of (provider_name, options_dict) for provider-specific options.

        Raises:
            RuntimeError: If requested provider is not available
            ValueError: If provider name is invalid
        """
        available = rt.get_available_providers()

        def _provider_key(
            entry: str | tuple[str, dict[str, str]],
        ) -> str:
            return entry[0] if isinstance(entry, tuple) else entry

        def _dedupe_providers(
            providers: list[str | tuple[str, dict[str, str]]],
        ) -> list[str | tuple[str, dict[str, str]]]:
            seen: set[str] = set()
            deduped: list[str | tuple[str, dict[str, str]]] = []
            for entry in providers:
                key = _provider_key(entry)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(entry)
            return deduped

        # Helper function to create provider list with options
        def _make_provider_list(prov: str) -> list[str | tuple[str, dict[str, str]]]:
            """Create provider list, adding options if needed."""
            # Get default options for this provider
            default_opts = self._get_default_provider_options(prov)

            # Get user-provided provider-specific options
            provider_opts = {}
            if self._provider_options:
                provider_opts = self._get_provider_specific_options(
                    prov, self._provider_options
                )

            # Merge defaults with user options (user options take precedence)
            merged_opts = {**default_opts, **provider_opts}

            if merged_opts:
                logger.info(f"Using {prov} with options: {merged_opts}")
                return [(prov, merged_opts)]

            return [prov]

        # Environment variable override (highest priority)
        env_provider = os.getenv("ONNX_PROVIDER")
        if env_provider:
            env_provider_clean = env_provider.strip()
            normalized_env = None
            provider_map = {
                "cpu": "CPUExecutionProvider",
                "cuda": "CUDAExecutionProvider",
                "openvino": "OpenVINOExecutionProvider",
                "directml": "DmlExecutionProvider",
                "coreml": "CoreMLExecutionProvider",
            }

            if env_provider_clean.lower() in provider_map:
                normalized_env = provider_map[env_provider_clean.lower()]
            else:
                for available_provider in available:
                    if available_provider.lower() == env_provider_clean.lower():
                        normalized_env = available_provider
                        break

            if normalized_env is None:
                raise ConfigurationError(
                    "ONNX_PROVIDER must match an available provider name. "
                    f"Got '{env_provider_clean}'. Available: {available}"
                )

            logger.info(f"Using provider from ONNX_PROVIDER env: {normalized_env}")
            return _dedupe_providers(_make_provider_list(normalized_env))

        # Auto-selection logic
        if provider == "auto" or (provider is None and use_gpu):
            # Priority: CUDA > OpenVINO > CoreML > DirectML
            for prov in [
                "CUDAExecutionProvider",
                "OpenVINOExecutionProvider",
                "CoreMLExecutionProvider",
                "DmlExecutionProvider",
            ]:
                if prov in available:
                    logger.info(f"Auto-selected provider: {prov}")
                    return _dedupe_providers(_make_provider_list(prov))
            logger.info("Auto-selection: No accelerators found, using CPU")
            return ["CPUExecutionProvider"]

        # Default to CPU if no provider specified and use_gpu=False
        if provider is None:
            logger.info("Using CPU provider")
            return ["CPUExecutionProvider"]

        # Explicit provider selection
        provider_map = {
            "cpu": "CPUExecutionProvider",
            "cuda": "CUDAExecutionProvider",
            "openvino": "OpenVINOExecutionProvider",
            "directml": "DmlExecutionProvider",
            "coreml": "CoreMLExecutionProvider",
        }

        selected = provider_map.get(provider.lower())
        if not selected:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Valid options: {list(provider_map.keys())}"
            )

        if selected not in available:
            # Provide helpful installation message
            install_hints = {
                "CUDAExecutionProvider": "pip install pykokoro[gpu]",
                "OpenVINOExecutionProvider": "pip install pykokoro[openvino]",
                "DmlExecutionProvider": "pip install pykokoro[directml]",
                "CoreMLExecutionProvider": "pip install pykokoro[coreml]",
            }
            hint = install_hints.get(
                selected, f"install the required package for {selected}"
            )
            raise RuntimeError(
                f"{provider.upper()} provider requested but not available.\n"
                f"Install with: {hint}\n"
                f"Available providers: {available}"
            )

        logger.info(f"Using explicitly selected provider: {selected}")
        return _dedupe_providers(_make_provider_list(selected))

    def _get_default_provider_options(self, provider: str) -> dict[str, str]:
        """Get sensible default options for a provider.

        Uses PyKokoro cache path and model quality for smart defaults.

        Args:
            provider: Provider name (e.g., "OpenVINOExecutionProvider")

        Returns:
            Dictionary of default provider options (string values)
        """
        cache_path = get_user_cache_path()
        return ProviderConfigManager.get_default_provider_options(
            provider=provider,
            model_quality=self._model_quality,
            cache_path=cache_path,
        )

    def _get_provider_specific_options(
        self,
        provider: str,
        all_options: dict[str, Any],
    ) -> dict[str, str]:
        """Extract provider-specific options for the given provider.

        Filters out SessionOptions attributes and converts values to strings
        as required by ONNX Runtime.

        Args:
            provider: Provider name (e.g., "OpenVINOExecutionProvider")
            all_options: Dictionary of all options (mixed session and provider options)

        Returns:
            Dictionary of provider-specific options with string values
        """
        return ProviderConfigManager.get_provider_specific_options(
            provider=provider,
            all_options=all_options,
        )

    def _apply_provider_options(
        self,
        sess_opt: rt.SessionOptions,
        options: dict[str, Any],
    ) -> None:
        """Apply provider options to SessionOptions.

        Handles both SessionOptions attributes and provider-specific configs.

        Args:
            sess_opt: SessionOptions to modify
            options: Dictionary of options to apply
        """
        # Map of common option names to SessionOptions attributes
        session_option_attrs: dict[str, str] = {
            "intra_op_num_threads": "intra_op_num_threads",
            "inter_op_num_threads": "inter_op_num_threads",
            "num_threads": "intra_op_num_threads",  # Alias
            "threads": "intra_op_num_threads",  # Alias
            "graph_optimization_level": "graph_optimization_level",
            "execution_mode": "execution_mode",
            "enable_profiling": "enable_profiling",
            "enable_mem_pattern": "enable_mem_pattern",
            "enable_cpu_mem_arena": "enable_cpu_mem_arena",
            "enable_mem_reuse": "enable_mem_reuse",
            "log_severity_level": "log_severity_level",
            "log_verbosity_level": "log_verbosity_level",
        }

        # Apply SessionOptions attributes
        for opt_name, value in options.items():
            if opt_name in session_option_attrs:
                attr_name = session_option_attrs[opt_name]
                setattr(sess_opt, attr_name, value)
                logger.debug(f"Set SessionOptions.{attr_name} = {value}")
