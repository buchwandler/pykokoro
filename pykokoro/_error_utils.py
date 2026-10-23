from __future__ import annotations

from collections.abc import Mapping

from .exceptions import BackendError

_MAX_NATIVE_SUMMARY_LENGTH = 8 * 1024
_TRUNCATED_SUFFIX = "…[truncated]"


def native_error_fields(exc: Exception) -> dict[str, str]:
    """Return the native exception's stable type and message fields."""
    return {
        "native_error_type": type(exc).__name__,
        "native_error_message": str(exc),
    }


def with_native_summary(summary: str, exc: Exception) -> str:
    """Append the native error type and a bounded, one-line message."""
    fields = native_error_fields(exc)
    native_message = " ".join(fields["native_error_message"].split())
    if len(native_message) > _MAX_NATIVE_SUMMARY_LENGTH:
        native_message = (
            native_message[: _MAX_NATIVE_SUMMARY_LENGTH - len(_TRUNCATED_SUFFIX)]
            + _TRUNCATED_SUFFIX
        )
    return f"{summary}: {fields['native_error_type']}: {native_message}"


def backend_error_from_exception(
    summary: str,
    exc: Exception,
    *,
    stage: str,
    details: Mapping[str, object],
) -> BackendError:
    """Build an actionable typed wrapper while preserving the native cause fields."""
    return BackendError(
        with_native_summary(summary, exc),
        stage=stage,
        details=details,
        **native_error_fields(exc),
    )
