"""Provider configuration utilities for ONNX Runtime.

This module provides shared utilities for managing execution provider options
across different PyKokoro components.
"""

import logging
from pathlib import Path
from typing import Any, ClassVar, Literal

logger = logging.getLogger(__name__)

# Model quality type
ModelQuality = Literal[
    "fp32", "fp16", "fp16-gpu", "q8", "q8f16", "q4", "q4f16", "uint8", "uint8f16"
]


class ProviderConfigManager:
    """Manages ONNX Runtime provider configuration.

    This class provides utilities for:
    - Getting default provider options based on provider type and model quality
    - Extracting provider-specific options from mixed configuration dictionaries
    - Separating SessionOptions attributes from provider-specific options
    """

    # Known provider-specific options for each provider
    PROVIDER_OPTIONS_MAP: ClassVar[dict[str, list[str]]] = {
        "OpenVINOExecutionProvider": [
            "device_type",
            "precision",
            "num_of_threads",
            "cache_dir",
            "enable_opencl_throttling",
        ],
        "CUDAExecutionProvider": [
            "device_id",
            "gpu_mem_limit",
            "arena_extend_strategy",
            "cudnn_conv_algo_search",
            "do_copy_in_default_stream",
        ],
        "DmlExecutionProvider": ["device_id", "disable_metacommands"],
        "CoreMLExecutionProvider": [
            "MLComputeUnits",
            "EnableOnSubgraphs",
            "ModelFormat",
        ],
    }

    # SessionOptions attributes (not provider-specific)
    SESSION_OPTION_ATTRS: ClassVar[set[str]] = {
        "intra_op_num_threads",
        "inter_op_num_threads",
        "num_threads",
        "threads",
        "graph_optimization_level",
        "execution_mode",
        "enable_profiling",
        "enable_mem_pattern",
        "enable_cpu_mem_arena",
        "enable_mem_reuse",
        "log_severity_level",
        "log_verbosity_level",
    }

    @staticmethod
    def get_default_provider_options(
        provider: str,
        model_quality: ModelQuality = "fp32",
        cache_path: Path | None = None,
    ) -> dict[str, str]:
        """Get sensible default options for a provider.

        Args:
            provider: Provider name (e.g., "OpenVINOExecutionProvider")
            model_quality: Model quality affects default provider settings
            cache_path: Optional cache directory path for providers that need it

        Returns:
            Dictionary of default provider options (string values)
        """
        defaults: dict[str, str] = {}

        if provider == "OpenVINOExecutionProvider":
            # Use provided cache dir or default
            if cache_path:
                cache_dir = cache_path / "openvino_cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                defaults["cache_dir"] = str(cache_dir)

            defaults.update(
                {
                    "device_type": "CPU_FP32",
                    "enable_opencl_throttling": "false",
                }
            )

            # Auto-set precision based on model_quality
            if model_quality in ["fp16", "fp16-gpu"]:
                defaults["precision"] = "FP16"
            elif model_quality == "fp32":
                defaults["precision"] = "FP32"
            else:
                # For quantized models, use FP32 precision in OpenVINO
                defaults["precision"] = "FP32"

        elif provider == "CUDAExecutionProvider":
            defaults = {
                "device_id": "0",
                "arena_extend_strategy": "kNextPowerOfTwo",
            }

        elif provider == "DmlExecutionProvider":
            defaults = {
                "device_id": "0",
            }

        return defaults

    @staticmethod
    def get_provider_specific_options(
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
        known_options = ProviderConfigManager.PROVIDER_OPTIONS_MAP.get(provider, [])

        # Extract only provider-specific options
        provider_opts: dict[str, str] = {}
        for key, value in all_options.items():
            if key in ProviderConfigManager.SESSION_OPTION_ATTRS:
                continue  # Skip SessionOptions attributes

            if known_options and key not in known_options:
                logger.warning(
                    f"Unknown option '{key}' for {provider}. Known options: {known_options}"
                )
                continue

            # Convert to string as required by ONNX Runtime
            provider_opts[key] = str(value)

        return provider_opts

    @staticmethod
    def merge_provider_options(
        defaults: dict[str, str],
        user_options: dict[str, str],
    ) -> dict[str, str]:
        """Merge default and user-provided provider options.

        User options take precedence over defaults.

        Args:
            defaults: Default provider options
            user_options: User-provided provider options

        Returns:
            Merged dictionary with user options overriding defaults
        """
        return {**defaults, **user_options}
