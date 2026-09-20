"""PyKokoro's narrow integration boundary for the OnnxVoice runtime."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .asset_progress import AssetProgressEvent, AssetProgressPhase
from .exceptions import BackendError, ConfigurationError, KokoroError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ResolvedKokoroModel:
    """PyKokoro's read-only view of an OnnxVoice installation."""

    ref: str | None
    model_id: str | None

    quality: str | None
    distribution: str | None
    storage_id: str | None

    installation: Any | None
    metadata: Mapping[str, Any]

    sample_rate: int
    model_paths: tuple[Path, ...]
    voices_path: Path | None
    config_path: Path | None


def _onnxvoice() -> Any:
    try:
        import onnxvoice
    except ModuleNotFoundError as exc:
        raise ConfigurationError(
            "OnnxVoice is required for Kokoro model loading. Install pykokoro[cpu], "
            "pykokoro[gpu], or another provider extra."
        ) from exc
    return onnxvoice


def normalize_kokoro_ref(ref: str) -> str:
    """Normalize a Kokoro model id to an explicit OnnxVoice system reference."""
    if not isinstance(ref, str) or not ref.strip():
        raise ConfigurationError("Kokoro model reference must not be empty")
    value = ref.strip()
    if ":" not in value:
        return f"kokoro:{value}"
    system, model_id = value.split(":", 1)
    if system.casefold() != "kokoro" or not model_id:
        raise ConfigurationError(f"Not a Kokoro model reference: {ref!r}")
    return f"kokoro:{model_id}"


# Mapping from PyKokoro's ModelSource to OnnxVoice distribution IDs.
# OnnxVoice uses specific distribution IDs that may differ from PyKokoro's source names.
_DISTRIBUTION_MAP: dict[tuple[str, str], str] = {
    ("github", "v1.0"): "github-model-files-v1.0-timestamped-r4",
    ("github", "v1.1-zh"): "github-model-files-v1.1",
}


def onnxvoice_distribution_for(*, source: str, variant: str) -> str:
    """Translate PyKokoro's ModelSource and variant to an exact OnnxVoice distribution ID.

    Raises ConfigurationError if the explicit source/variant pair has no mapping.
    """
    try:
        return _DISTRIBUTION_MAP[(source, variant)]
    except KeyError as exc:
        raise ConfigurationError(
            f"No OnnxVoice distribution mapping for source={source!r}, variant={variant!r}"
        ) from exc


def _provider_name(value: Any) -> tuple[str, dict[str, Any]]:
    if hasattr(value, "name"):
        return str(value.name), dict(getattr(value, "options", {}) or {})
    if isinstance(value, tuple):
        name, options = value
        return str(name), dict(options or {})
    return str(value), {}


def normalize_provider_request(
    providers: Sequence[Any] | str | None,
    provider_options: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> tuple[str | Sequence[str] | None, Any | None]:
    """Translate PyKokoro provider compatibility values to OnnxVoice values."""
    requested = os.getenv("ONNX_PROVIDER") if providers is None else providers
    if requested is None:
        return "cpu", _normalize_provider_options("cpu", provider_options)
    if isinstance(requested, str):
        name = {"gpu": "cuda", "dml": "directml"}.get(requested.strip().casefold(), requested)
        return name, _normalize_provider_options(name, provider_options)

    names: list[str] = []
    options: list[dict[str, Any]] = []
    has_explicit_options = False
    for value in requested:
        name, value_options = _provider_name(value)
        name = {"gpu": "cuda", "dml": "directml"}.get(name.casefold(), name)
        names.append(name)
        options.append(value_options)
        has_explicit_options = has_explicit_options or bool(value_options)
    if not names:
        raise ConfigurationError("providers must not be empty")
    if has_explicit_options:
        return names, options
    return names, _normalize_provider_options(names, provider_options)


def _normalize_provider_options(
    providers: str | Sequence[str],
    provider_options: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> Any | None:
    if provider_options is None or isinstance(provider_options, (list, tuple)):
        return provider_options
    values = dict(provider_options)
    provider_names = [providers] if isinstance(providers, str) else list(providers)
    if any(name in values for name in provider_names):
        return {str(name): dict(values.get(name, {})) for name in provider_names}
    return [values.copy() for _ in provider_names]


def adapt_progress(
    callback: Callable[[AssetProgressEvent], None] | None,
) -> Callable[[Any], None] | None:
    """Adapt OnnxVoice asset events to PyKokoro asset progress events."""
    if callback is None:
        return None

    def emit(event: Any) -> None:
        phase_map: dict[str, AssetProgressPhase] = {
            "download_started": "download-start",
            "download_progress": "download-progress",
            "verify_started": "verify-start",
            "verify_completed": "verify-complete",
            "download_completed": "download-complete",
            "artifact_cached": "cache-hit",
            "artifact_installed": "artifact-installed",
            "install_completed": "install-complete",
            "install_failed": "install-failed",
        }
        raw_phase = str(getattr(event, "phase", ""))
        phase = phase_map.get(raw_phase)
        if phase is None:
            logger.debug("Ignoring unsupported OnnxVoice progress phase %r", raw_phase)
            return
        ref = getattr(event, "ref", None)
        model_id = ref.split(":", 1)[1] if isinstance(ref, str) and ":" in ref else str(ref or "")
        artifact = getattr(event, "artifact", None)
        role = getattr(event, "role", None)
        completed = getattr(event, "completed", None)
        total = getattr(event, "total", None)
        callback(
            AssetProgressEvent(
                phase=phase,
                model_id=model_id,
                distribution_id="",
                artifact_id=str(artifact or ""),
                role=str(role or "artifact"),
                filename=str(artifact or ""),
                bytes_done=int(completed) if completed is not None else 0,
                bytes_total=int(total) if total is not None else None,
                target=str(getattr(event, "target", "") or ""),
                message=getattr(event, "message", None),
            )
        )

    return emit


def _map_error(exc: Exception, *, operation: str) -> KokoroError:
    name = type(exc).__name__
    if name in {
        "OptionalDependencyError",
        "ImportError",
        "RuntimeContractError",
        "CapabilityError",
    }:
        return ConfigurationError(str(exc))
    if name in {"AssetNotFoundError", "NotInstalledError", "CatalogError"}:
        return ConfigurationError(str(exc))
    if name in {"IntegrityError", "ManifestError", "UnsafePathError", "LockError"}:
        return BackendError(f"Kokoro model asset integrity failure: {exc}")
    if operation == "infer":
        return BackendError("Kokoro model inference failed")
    if operation == "open":
        return BackendError(f"Could not open Kokoro runtime: {exc}")
    return BackendError(str(exc))


def _call(operation: str, function: Callable[[], Any]) -> Any:
    try:
        return function()
    except (FileNotFoundError, ValueError, TypeError, ConfigurationError):
        raise
    except Exception as exc:
        mapped = _map_error(exc, operation=operation)
        raise mapped from exc


def _installation_info(installation: Any, *, ref: str | None = None) -> ResolvedKokoroModel:
    """Extract PyKokoro's resolved view from an OnnxVoice Installation."""
    # Use semantic artifact helpers where available.
    # Fall back to manual scanning for test mocks that lack these methods.
    if hasattr(installation, "artifacts_for"):
        model_paths = tuple(Path(artifact.path) for artifact in installation.artifacts_for("model"))
    else:
        artifacts = tuple(getattr(installation, "artifacts", ()) or ())
        model_paths = tuple(
            Path(artifact.path)
            for artifact in artifacts
            if getattr(artifact, "role", None) == "model"
        )

    if hasattr(installation, "require_artifact"):
        try:
            voices_path = Path(installation.require_artifact("voices").path)
        except Exception:
            voices_path = None
    else:
        artifacts = tuple(getattr(installation, "artifacts", ()) or ())
        voices_path = next(
            (
                Path(artifact.path)
                for artifact in artifacts
                if getattr(artifact, "role", None) == "voices"
            ),
            None,
        )

    # Vocabulary/config: prefer artifact_path for unique roles.
    config_path = None
    if hasattr(installation, "artifact_path"):
        try:
            config_path = Path(installation.artifact_path("vocab"))
        except Exception:
            try:
                config_path = Path(installation.artifact_path("config"))
            except Exception:
                config_path = None
    else:
        artifacts = tuple(getattr(installation, "artifacts", ()) or ())
        config_path = next(
            (
                Path(artifact.path)
                for artifact in artifacts
                if getattr(artifact, "role", None) in {"config", "vocab"}
            ),
            None,
        )

    # Use typed properties from the Installation, with metadata fallback.
    selected_quality = getattr(installation, "selected_quality", None)
    if selected_quality is None:
        metadata = dict(getattr(installation, "metadata", {}) or {})
        selected_quality = metadata.get("selected_quality")
    else:
        metadata = dict(getattr(installation, "metadata", {}) or {})

    selected_distribution = getattr(installation, "selected_distribution", None)
    if selected_distribution is None:
        selected_distribution = metadata.get("selected_distribution")

    storage_id = getattr(installation, "storage_id", None)
    if storage_id is None:
        storage_id = metadata.get("storage_id")

    return ResolvedKokoroModel(
        ref=ref or getattr(installation, "ref", None),
        model_id=getattr(installation, "id", None),
        quality=str(selected_quality) if selected_quality is not None else None,
        distribution=str(selected_distribution) if selected_distribution is not None else None,
        storage_id=str(storage_id) if storage_id is not None else None,
        installation=installation,
        metadata=metadata,
        sample_rate=int(getattr(installation, "sample_rate", None) or 24000),
        model_paths=model_paths,
        voices_path=voices_path,
        config_path=config_path,
    )


def install_kokoro_model(
    ref: str,
    *,
    quality: str | None = None,
    source: str | None = None,
    distribution: str | None = None,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    refresh: bool = False,
    force: bool = False,
    progress: Callable[[AssetProgressEvent], None] | None = None,
) -> ResolvedKokoroModel:
    """Install a Kokoro model via OnnxVoice.

    Either *source* (PyKokoro compatibility alias like 'github') or
    *distribution* (exact OnnxVoice distribution ID) may be supplied.
    When *source* is given, it is translated to an OnnxVoice distribution
    using the internal mapping.
    """
    normalized = normalize_kokoro_ref(ref)
    if source is not None:
        variant = normalized.split(":", 1)[1]
        onnxvoice_distribution = onnxvoice_distribution_for(source=source, variant=variant)
    else:
        onnxvoice_distribution = distribution
    module = _onnxvoice()
    manager = module.OnnxVoice(cache_dir=cache_dir, offline=offline)
    installation = _call(
        "install",
        lambda: manager.install(
            normalized,
            quality=quality,
            distribution=onnxvoice_distribution,
            refresh=refresh,
            force=force,
            progress=adapt_progress(progress),
        ),
    )
    return _installation_info(installation, ref=normalized)


def resolve_kokoro_model(
    ref: str,
    *,
    quality: str | None = None,
    source: str | None = None,
    distribution: str | None = None,
    cache_dir: str | Path | None = None,
    offline: bool = False,
) -> ResolvedKokoroModel:
    """Resolve a locally installed Kokoro model via OnnxVoice.

    Either *source* (PyKokoro compatibility alias) or
    *distribution* (exact OnnxVoice distribution ID) may be supplied.
    """
    normalized = normalize_kokoro_ref(ref)
    if source is not None:
        variant = normalized.split(":", 1)[1]
        onnxvoice_distribution = onnxvoice_distribution_for(source=source, variant=variant)
    else:
        onnxvoice_distribution = distribution
    manager = _onnxvoice().OnnxVoice(cache_dir=cache_dir, offline=offline)
    installation = _call(
        "resolve",
        lambda: manager.resolve(normalized, quality=quality, distribution=onnxvoice_distribution),
    )
    return _installation_info(installation, ref=normalized)


def open_local_kokoro(
    *,
    model: str | Path | None = None,
    artifacts: Mapping[str, str | Path] | None = None,
    config: str | Path | None = None,
    voices: str | Path | None = None,
    runtime: Mapping[str, Any] | None = None,
    sample_rate: int | None = None,
    providers: Sequence[Any] | str | None = None,
    provider_options: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    session_options: Any | None = None,
) -> Any:
    requested, options = normalize_provider_request(providers, provider_options)
    runtime_adapter = _call(
        "open",
        lambda: _onnxvoice().open_local(
            system="kokoro",
            model=model,
            artifacts=artifacts,
            config=config,
            voices=voices,
            runtime=runtime,
            sample_rate=sample_rate,
            providers=requested,
            provider_options=options,
            session_options=session_options,
        ),
    )
    resolved = _installation_info(getattr(runtime_adapter, "installation", None))
    return KokoroRuntimeAdapter(runtime_adapter, resolved)


def open_installed_kokoro(
    resolved: ResolvedKokoroModel,
    *,
    providers: Sequence[Any] | str | None = None,
    provider_options: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    session_options: Any | None = None,
) -> Any:
    requested, options = normalize_provider_request(providers, provider_options)
    manager = _onnxvoice().OnnxVoice()
    runtime_adapter = _call(
        "open",
        lambda: manager.open(
            resolved.installation,
            providers=requested,
            provider_options=options,
            session_options=session_options,
        ),
    )
    return KokoroRuntimeAdapter(runtime_adapter, resolved)


def runtime_diagnostics(runtime: Any) -> dict[str, Any]:
    diagnostic = runtime.diagnostics()
    sessions = getattr(diagnostic, "sessions", ())
    return {
        "system": getattr(diagnostic, "system", "kokoro"),
        "ref": getattr(diagnostic, "ref", None),
        "layout": getattr(diagnostic, "layout", "single"),
        "sessions": tuple(
            {
                "component": getattr(session, "component", None),
                "model_path": str(getattr(session, "model_path", "")),
                "providers_requested": tuple(getattr(session, "providers_requested", ())),
                "providers_active": tuple(getattr(session, "providers_active", ())),
                "inputs": tuple(getattr(session, "inputs", ())),
                "outputs": tuple(getattr(session, "outputs", ())),
            }
            for session in sessions
        ),
    }


def summarize_inference(result: Any) -> tuple[Any, dict[str, Any]]:
    """Return trace-safe timing and output shape summaries for an inference result."""
    import numpy as np

    def summary(value: Any) -> dict[str, Any]:
        array = np.asarray(value)
        return {"shape": [int(size) for size in array.shape], "dtype": str(array.dtype)}

    timing = getattr(result, "timings", None)
    outputs = {
        str(name): summary(value)
        for name, value in dict(getattr(result, "outputs", {}) or {}).items()
    }
    return (summary(timing) if timing is not None else None), outputs


def close_runtime(runtime: Any | None) -> None:
    if runtime is not None:
        close = getattr(runtime, "close", None)
        if close is not None:
            close()


def _declared_timing_support(
    resolved: ResolvedKokoroModel,
    runtime: Any,
) -> bool | None:
    """Resolve declared timing support without conflating unknown with false."""
    runtime_value = getattr(runtime, "supports_timings", None)
    if isinstance(runtime_value, bool):
        return runtime_value

    installation = resolved.installation
    timing_output = getattr(installation, "timing_output", None)
    if isinstance(timing_output, bool):
        return timing_output
    if isinstance(timing_output, str) and timing_output.strip():
        return True

    metadata = resolved.metadata
    runtime_metadata = metadata.get("runtime")
    if isinstance(runtime_metadata, Mapping):
        value = runtime_metadata.get("timings_output")
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip():
            return True

    onnx_contract = metadata.get("onnx_contract")
    if isinstance(onnx_contract, Mapping):
        timing = onnx_contract.get("timing")
        if isinstance(timing, Mapping):
            value = timing.get("output")
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.strip():
                return True

    return None


def _timing_output_name(resolved: ResolvedKokoroModel) -> str | None:
    """Return the declared timing output name for diagnostics, if available."""
    installation_output = getattr(resolved.installation, "timing_output", None)
    if isinstance(installation_output, str) and installation_output.strip():
        return installation_output
    runtime_metadata = resolved.metadata.get("runtime")
    if isinstance(runtime_metadata, Mapping):
        value = runtime_metadata.get("timings_output")
        if isinstance(value, str) and value.strip():
            return value
    onnx_contract = resolved.metadata.get("onnx_contract")
    if isinstance(onnx_contract, Mapping):
        timing = onnx_contract.get("timing")
        if isinstance(timing, Mapping):
            value = timing.get("output")
            if isinstance(value, str) and value.strip():
                return value
    return None


class KokoroRuntimeAdapter:
    """Adapt an OnnxVoice Kokoro adapter to PyKokoro's runtime protocol."""

    def __init__(self, runtime: Any, resolved: ResolvedKokoroModel) -> None:
        self._runtime = runtime
        self.resolved = resolved
        self.sample_rate = resolved.sample_rate
        self.supports_timings = _declared_timing_support(resolved, runtime)
        logger.debug(
            "Kokoro runtime created timestamp_support=%s timing_output=%s",
            "declared" if self.supports_timings is not None else "unknown",
            _timing_output_name(resolved),
        )
        # Managed installations use storage_id for cache identity.
        # Local unmanaged models use path-sensitive identity.
        if resolved.storage_id is not None:
            self.cache_identity: tuple[Any, ...] = (
                "managed",
                resolved.ref,
                resolved.storage_id,
            )
        else:
            self.cache_identity = (
                "local",
                tuple(str(path.resolve()) for path in resolved.model_paths),
                str(resolved.voices_path.resolve()) if resolved.voices_path is not None else None,
                str(resolved.config_path.resolve()) if resolved.config_path is not None else None,
            )
        self._closed = False

    def infer(
        self,
        token_ids: Sequence[int],
        *,
        style: Any,
        speed: float,
        seed: int | None = None,
    ) -> Any:
        if self._closed:
            raise RuntimeError("Kokoro runtime is closed")
        return _call(
            "infer",
            lambda: self._runtime.infer(token_ids, style=style, speed=speed, seed=seed),
        )

    def diagnostics(self) -> Any:
        return self._runtime.diagnostics()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close_runtime(self._runtime)

    @property
    def installation(self) -> Any:
        return self.resolved.installation


__all__ = [
    "ResolvedKokoroModel",
    "KokoroRuntimeAdapter",
    "adapt_progress",
    "close_runtime",
    "install_kokoro_model",
    "normalize_kokoro_ref",
    "normalize_provider_request",
    "onnxvoice_distribution_for",
    "open_installed_kokoro",
    "open_local_kokoro",
    "resolve_kokoro_model",
    "runtime_diagnostics",
    "summarize_inference",
]
