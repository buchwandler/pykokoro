"""Dispatch resolved registry assets to their layout-specific runtime."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..model_registry import ModelRegistryError
from .model_assets import ResolvedRuntimeAssets
from .thai_wayu import ThaiWayuRuntime


class UnsupportedRuntimeLayout(ModelRegistryError):
    """Raised when a registry layout has no PyKokoro implementation."""


def create_runtime(
    assets: ResolvedRuntimeAssets,
    *,
    session_factory: Callable[[Path], Any] | None = None,
) -> ThaiWayuRuntime | None:
    """Create a nonstandard runtime, or return ``None`` for single-ONNX assets."""
    if assets.layout == "single-onnx-v1":
        return None
    if assets.layout == "split-onnx-v1":
        return ThaiWayuRuntime(assets, session_factory=session_factory)
    raise UnsupportedRuntimeLayout(
        f"Runtime layout {assets.layout!r} is not implemented for {assets.model_id!r}"
    )
