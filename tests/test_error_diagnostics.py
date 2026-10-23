from __future__ import annotations

import json

import pytest

from pykokoro.error_diagnostics import (
    ERROR_DIAGNOSTIC_SCHEMA,
    ERROR_DIAGNOSTICS_VERSION,
    error_diagnostic,
)
from pykokoro.exceptions import PyKokoroError


def test_error_diagnostic_serializes_single_error_and_structured_fields() -> None:
    error = PyKokoroError(
        "operation failed",
        stage="inference",
        details={"operation": "infer", "token_count": 3},
        native_error_type="RuntimeError",
        native_error_message="provider unavailable",
    )

    diagnostic = error_diagnostic(error)

    assert ERROR_DIAGNOSTICS_VERSION == 1
    assert diagnostic == {
        "schema": ERROR_DIAGNOSTIC_SCHEMA,
        "type": "PyKokoroError",
        "module": "pykokoro.exceptions",
        "message": "operation failed",
        "stage": "inference",
        "details": {"operation": "infer", "token_count": 3},
        "native_error_type": "RuntimeError",
        "native_error_message": "provider unavailable",
    }
    assert json.loads(json.dumps(diagnostic)) == diagnostic


def test_error_diagnostic_serializes_explicit_cause_before_context() -> None:
    cause = RuntimeError("native cause")
    context = ValueError("implicit context")
    error = PyKokoroError("wrapper")
    error.__cause__ = cause
    error.__context__ = context

    diagnostic = error_diagnostic(error)

    assert diagnostic["cause"] == {
        "type": "RuntimeError",
        "module": "builtins",
        "message": "native cause",
    }
    assert "context" not in diagnostic


def test_error_diagnostic_uses_unsuppressed_context_as_fallback() -> None:
    error = PyKokoroError("wrapper")
    error.__context__ = ValueError("implicit cause")

    diagnostic = error_diagnostic(error)

    assert diagnostic["context"]["message"] == "implicit cause"  # type: ignore[index]


def test_error_diagnostic_omits_suppressed_context() -> None:
    error = PyKokoroError("wrapper")
    error.__context__ = ValueError("suppressed")
    error.__suppress_context__ = True

    diagnostic = error_diagnostic(error)

    assert "context" not in diagnostic


def test_error_diagnostic_detects_exception_cycles() -> None:
    first = RuntimeError("first")
    second = ValueError("second")
    first.__cause__ = second
    second.__cause__ = first

    diagnostic = error_diagnostic(first)

    cycle = diagnostic["cause"]["cause"]  # type: ignore[index]
    assert cycle["message"] == "first"  # type: ignore[index]
    assert cycle["cycle"] is True  # type: ignore[index]
    json.dumps(diagnostic)


def test_error_diagnostic_marks_max_depth_truncation() -> None:
    first = RuntimeError("first")
    second = ValueError("second")
    third = LookupError("third")
    first.__cause__ = second
    second.__cause__ = third

    diagnostic = error_diagnostic(first, max_depth=2)

    assert diagnostic["cause"]["message"] == "second"  # type: ignore[index]
    assert diagnostic["cause"]["truncated"] is True  # type: ignore[index]
    assert "cause" not in diagnostic["cause"]  # type: ignore[operator]


def test_error_diagnostic_limits_long_messages() -> None:
    diagnostic = error_diagnostic(RuntimeError("x" * 20_000))

    assert len(diagnostic["message"]) <= 8 * 1024
    assert diagnostic["message"].endswith("…[truncated]")  # type: ignore[union-attr]


def test_error_diagnostic_filters_sensitive_details_and_describes_unsupported_values() -> None:
    class NoRepr:
        def __repr__(self) -> str:
            raise AssertionError("repr must not be called")

    error = PyKokoroError(
        "failed",
        details={
            "request_text": "private request",
            "reference_transcript": "private transcript",
            "audio_samples": [0.1, 0.2],
            "token_ids": [1, 2, 3],
            "reference_text_sha256": "abc123",
            "token_count": 3,
            "unsupported": NoRepr(),
        },
    )

    diagnostic = error_diagnostic(error)
    serialized = json.dumps(diagnostic)

    assert "private request" not in serialized
    assert "private transcript" not in serialized
    assert "audio_samples" not in serialized
    assert "token_ids" not in serialized
    assert diagnostic["details"]["reference_text_sha256"] == "abc123"  # type: ignore[index]
    assert diagnostic["details"]["token_count"] == 3  # type: ignore[index]
    assert diagnostic["details"]["unsupported"] == {"type": "NoRepr", "module": __name__}  # type: ignore[index]


def test_error_diagnostic_does_not_capture_traceback_or_frame_locals() -> None:
    def raise_error() -> BaseException:
        secret_local = "private local"
        try:
            raise RuntimeError("visible error")
        except RuntimeError as error:
            assert secret_local
            return error

    diagnostic = error_diagnostic(raise_error())
    serialized = json.dumps(diagnostic)

    assert "traceback" not in diagnostic
    assert "private local" not in serialized
    assert "visible error" in serialized


def test_error_diagnostic_bounds_detail_collections_and_nested_values() -> None:
    nested: object = "leaf"
    for _ in range(10):
        nested = {"level": nested}
    error = PyKokoroError(
        "bounded",
        details={
            "items": list(range(150)),
            "mapping": {str(index): index for index in range(150)},
            "nested": nested,
        },
    )

    diagnostic = error_diagnostic(error)
    details = diagnostic["details"]

    assert len(details["items"]) == 100  # type: ignore[index]
    assert len(details["mapping"]) == 100  # type: ignore[arg-type,index]
    nested_value = details["nested"]  # type: ignore[index]
    for _ in range(6):
        nested_value = nested_value["level"]  # type: ignore[index]
    assert nested_value == {"type": "dict", "module": "builtins"}


def test_error_diagnostic_rejects_invalid_max_depth() -> None:

    with pytest.raises(ValueError, match="positive integer"):
        error_diagnostic(RuntimeError("invalid depth"), max_depth=0)
