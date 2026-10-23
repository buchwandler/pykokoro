from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TypeAlias, cast

from .exceptions import PyKokoroError

ERROR_DIAGNOSTICS_VERSION = 1
ERROR_DIAGNOSTIC_SCHEMA = "pykokoro.error-diagnostic.v1"

_MAX_MESSAGE_LENGTH = 8 * 1024
_MAX_KEY_LENGTH = 256
_MAX_COLLECTION_ITEMS = 100
_MAX_DETAIL_DEPTH = 6
_TRUNCATED_SUFFIX = "…[truncated]"
_SAFE_DETAIL_KEYS = {
    "textlength",
    "textsha256",
    "referencetextsha256",
    "transcriptsha256",
    "audiosha256",
    "referenceaudiosha256",
    "tokencount",
    "maxtokens",
}
_SENSITIVE_DETAIL_PARTS = (
    "transcript",
    "phoneme",
    "audio",
    "waveform",
    "sample",
    "tokenid",
    "tokenarray",
    "tokensequence",
    "requesttext",
    "referencetext",
    "prompt",
)

JsonValue: TypeAlias = None | bool | int | float | str | list[object] | dict[str, object]


def error_diagnostic(
    error: BaseException,
    *,
    max_depth: int = 8,
) -> dict[str, object]:
    """Serialize an exception and its safe cause chain for downstream diagnostics.

    The result deliberately excludes traceback frames and arbitrary exception or
    detail-object representations. ``max_depth`` counts the top-level exception.
    """
    if not isinstance(error, BaseException):
        raise TypeError("error must be an exception")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1:
        raise ValueError("max_depth must be a positive integer")

    diagnostic = _serialize_error(error, depth=0, max_depth=max_depth, seen=set())
    return {"schema": ERROR_DIAGNOSTIC_SCHEMA, **diagnostic}


def _serialize_error(
    error: BaseException,
    *,
    depth: int,
    max_depth: int,
    seen: set[int],
) -> dict[str, object]:
    identity = id(error)
    node = _exception_fields(error)
    if identity in seen:
        node["cycle"] = True
        return node

    seen.add(identity)
    try:
        linked_error: BaseException | None = None
        link_name: str | None = None
        if error.__cause__ is not None:
            linked_error = error.__cause__
            link_name = "cause"
        elif not error.__suppress_context__ and error.__context__ is not None:
            linked_error = error.__context__
            link_name = "context"

        if linked_error is not None and link_name is not None:
            if depth + 1 >= max_depth:
                node["truncated"] = True
            else:
                node[link_name] = _serialize_error(
                    linked_error,
                    depth=depth + 1,
                    max_depth=max_depth,
                    seen=seen,
                )
        return node
    finally:
        seen.remove(identity)


def _exception_fields(error: BaseException) -> dict[str, object]:
    exception_type = type(error)
    type_name = _class_attribute(exception_type, "__name__", "BaseException")
    module_name = _class_attribute(exception_type, "__module__", "builtins")
    node: dict[str, object] = {
        "type": type_name,
        "module": module_name,
        "message": _bounded_message(error),
    }

    if isinstance(error, PyKokoroError):
        if isinstance(error.stage, str) and error.stage:
            node["stage"] = _bounded_text(error.stage, _MAX_KEY_LENGTH)
        details = _json_safe_details(error.details)
        if details:
            node["details"] = details
        if isinstance(error.native_error_type, str):
            node["native_error_type"] = _bounded_text(error.native_error_type, _MAX_KEY_LENGTH)
        if isinstance(error.native_error_message, str):
            node["native_error_message"] = _bounded_text(error.native_error_message)
    return node


def _bounded_message(error: BaseException) -> str:
    try:
        message = str(error)
    except Exception:
        message = "<message unavailable>"
    return _bounded_text(message)


def _bounded_text(value: str, limit: int = _MAX_MESSAGE_LENGTH) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - len(_TRUNCATED_SUFFIX)] + _TRUNCATED_SUFFIX


def _class_attribute(cls: type[object], name: str, fallback: str) -> str:
    value = getattr(cls, name, fallback)
    if isinstance(value, str):
        return _bounded_text(value, _MAX_KEY_LENGTH)
    return fallback


def _json_safe_details(details: Mapping[str, object]) -> dict[str, object]:
    safe: dict[str, object] = {}
    values = cast(dict[str, object], details) if type(details) is dict else {}
    for index, (key, value) in enumerate(values.items()):
        if index >= _MAX_COLLECTION_ITEMS:
            break
        if type(key) is not str or _is_sensitive_detail_key(key):
            continue
        normalized_key = _bounded_text(key, _MAX_KEY_LENGTH)
        if _normalized_detail_key(key) in _SAFE_DETAIL_KEYS and type(value) not in {
            bool,
            int,
            float,
            str,
            type(None),
        }:
            continue
        safe[normalized_key] = _json_safe(value, depth=0, containers=set())
    return safe


def _normalized_detail_key(key: str) -> str:
    return "".join(character for character in key.casefold() if character.isalnum())


def _is_sensitive_detail_key(key: str) -> bool:
    normalized = _normalized_detail_key(key)
    if normalized in _SAFE_DETAIL_KEYS or normalized in {"requestid", "modelid", "modelvariant"}:
        return False
    return any(part in normalized for part in _SENSITIVE_DETAIL_PARTS)


def _json_safe(value: object, *, depth: int, containers: set[int]) -> JsonValue:
    value_type = type(value)
    if value is None:
        return None
    if value_type is bool:
        return cast(bool, value)
    if value_type is str:
        return _bounded_text(cast(str, value))
    if value_type is int:
        integer = cast(int, value)
        return integer if integer.bit_length() <= 128 else _type_description(integer)
    if value_type is float:
        floating = cast(float, value)
        return floating if math.isfinite(floating) else _type_description(floating)
    if depth >= _MAX_DETAIL_DEPTH:
        return _type_description(value)

    identity = id(value)
    if identity in containers:
        return _type_description(value)

    if value_type is list:
        items = cast(list[object], value)
        containers.add(identity)
        try:
            return [
                _json_safe(item, depth=depth + 1, containers=containers)
                for item in items[:_MAX_COLLECTION_ITEMS]
            ]
        finally:
            containers.remove(identity)
    if value_type is dict:
        mapping = cast(dict[object, object], value)
        containers.add(identity)
        try:
            safe: dict[str, object] = {}
            for index, (key, item) in enumerate(mapping.items()):
                if index >= _MAX_COLLECTION_ITEMS:
                    break
                if type(key) is str and not _is_sensitive_detail_key(key):
                    safe[_bounded_text(key, _MAX_KEY_LENGTH)] = _json_safe(
                        item,
                        depth=depth + 1,
                        containers=containers,
                    )
            return safe
        finally:
            containers.remove(identity)
    return _type_description(value)


def _type_description(value: object) -> dict[str, object]:
    value_type = type(value)
    return {
        "type": _class_attribute(value_type, "__name__", "object"),
        "module": _class_attribute(value_type, "__module__", "builtins"),
    }
