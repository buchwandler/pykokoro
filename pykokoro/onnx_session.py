"""Deprecated provider compatibility helpers backed by OnnxVoice."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from .config_types import DEFAULT_MODEL_QUALITY, ModelQuality, ProviderType
from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

PROVIDER_ALIASES: Final[dict[str, str]] = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "directml": "DmlExecutionProvider",
    "dml": "DmlExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "nnapi": "NnapiExecutionProvider",
    "xnnpack": "XnnpackExecutionProvider",
}

AUTO_PROVIDER_PRIORITY: Final[tuple[str, ...]] = (
    "CUDAExecutionProvider",
    "NnapiExecutionProvider",
    "OpenVINOExecutionProvider",
    "CoreMLExecutionProvider",
    "DmlExecutionProvider",
    "XnnpackExecutionProvider",
    "CPUExecutionProvider",
)


def get_available_execution_providers() -> tuple[str, ...]:
    """Return provider names reported by OnnxVoice."""
    from onnxvoice import available_providers

    return tuple(str(provider) for provider in available_providers())


def normalize_execution_provider(
    requested: str,
    *,
    available: Sequence[str] | None = None,
) -> str:
    """Resolve a legacy provider alias to the runtime spelling."""
    requested_clean = requested.strip()
    available_providers = tuple(available or get_available_execution_providers())
    if not requested_clean:
        raise ConfigurationError(
            "Execution provider cannot be empty. "
            f"Requested: {requested!r}; available providers: {available_providers}"
        )
    candidate = PROVIDER_ALIASES.get(requested_clean.casefold(), requested_clean)
    for provider in available_providers:
        if provider.casefold() == candidate.casefold():
            return provider
    raise ConfigurationError(
        "Requested execution provider is unavailable. "
        f"Requested: {requested_clean}; available providers: {available_providers}"
    )


def resolve_execution_provider(
    provider: str | None = None,
    *,
    use_gpu: bool = False,
    available: Sequence[str] | None = None,
    respect_environment: bool = True,
) -> str:
    """Resolve the legacy provider API without creating an ONNX session."""
    available_providers = tuple(available or get_available_execution_providers())
    requested = provider
    environment_provider = os.getenv("ONNX_PROVIDER") if respect_environment else None
    if environment_provider and environment_provider.strip():
        requested = environment_provider
    if requested is None:
        requested = "auto" if use_gpu else "cpu"
    if requested.strip().casefold() == "auto":
        for candidate in AUTO_PROVIDER_PRIORITY:
            if candidate.casefold() in {value.casefold() for value in available_providers}:
                return next(
                    value
                    for value in available_providers
                    if value.casefold() == candidate.casefold()
                )
        raise ConfigurationError(
            f"No supported execution provider is available: {available_providers}"
        )
    try:
        return normalize_execution_provider(requested, available=available_providers)
    except ConfigurationError as exc:
        if environment_provider:
            raise ConfigurationError(f"ONNX_PROVIDER: {exc}") from exc
        raise


class OnnxSessionManager:
    """Deprecated compatibility facade; OnnxVoice owns session creation."""

    def __init__(
        self,
        use_gpu: bool = False,
        provider: ProviderType | None = None,
        session_options: Any | None = None,
        provider_options: dict[str, Any] | None = None,
        model_quality: ModelQuality = DEFAULT_MODEL_QUALITY,
    ) -> None:
        self._use_gpu = use_gpu
        self._provider = provider
        self._session_options = session_options
        self._provider_options = provider_options or {}
        self._model_quality = model_quality

    def _select_providers(self, provider: str | None, use_gpu: bool) -> list[str]:
        selected = resolve_execution_provider(provider, use_gpu=use_gpu)
        return [selected]

    def create_session(self, model_path: Path, allow_fallback: bool = True) -> Any:
        del allow_fallback
        from onnxvoice import OnnxSession

        provider = self._provider or ("auto" if self._use_gpu else "cpu")
        return OnnxSession(
            model_path,
            providers=provider,
            provider_options=self._provider_options,
            session_options=self._session_options,
        )

    def _create_session_options(self) -> Any | None:
        return self._session_options

    def _get_default_provider_options(self, provider: str) -> dict[str, Any]:
        if provider.casefold() == "openvinoexecutionprovider":
            return {"device_type": "CPU", "cache_dir": "", "precision": self._model_quality.upper()}
        if provider.casefold() in {"cudaexecutionprovider", "dmlexecutionprovider"}:
            return {"device_id": "0", "arena_extend_strategy": "kSameAsRequested"}
        return {}

    def _get_provider_specific_options(
        self, provider: str, options: dict[str, Any]
    ) -> dict[str, str]:
        del provider
        return {str(key): str(value) for key, value in options.items()}


__all__ = [
    "AUTO_PROVIDER_PRIORITY",
    "OnnxSessionManager",
    "PROVIDER_ALIASES",
    "get_available_execution_providers",
    "normalize_execution_provider",
    "resolve_execution_provider",
]
