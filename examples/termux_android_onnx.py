#!/usr/bin/env python3
"""Inspect ONNX Runtime providers from a Termux/Android Python install.

Run this after installing ``onnxruntime`` and PyKokoro::

    python examples/termux_android_onnx.py

ONNX Runtime may print an ``Unsupported platform (android)`` warning on
Termux. That warning is expected for the current package build; the provider
list is still useful for selecting a PyKokoro execution provider. It is independent
of model downloads. When HuggingFace downloads are unavailable, select the
self-contained GitHub v1.0 assets explicitly with::

    PipelineConfig(
        model_source="github", model_variant="v1.0", model_quality="fp32"
    )

GitHub v1.0 uses the embedded standard vocabulary and PyKokoro never silently changes
the configured model source.
"""

from __future__ import annotations

from pykokoro.exceptions import ConfigurationError
from pykokoro.onnx_session import (
    get_available_execution_providers,
    resolve_execution_provider,
)


def main() -> None:
    """Print runtime providers and PyKokoro's provider resolutions."""
    import onnxruntime as ort

    runtime_providers = tuple(ort.get_available_providers())
    print("ONNX Runtime:", getattr(ort, "version", ort.__version__))
    print("Providers:", list(runtime_providers))
    print("PyKokoro providers:", list(get_available_execution_providers()))

    print("\nTermux model source example:")
    print('  PipelineConfig(model_source="github", model_variant="v1.0", model_quality="fp32")')
    print("  GitHub v1.0 is explicit and does not require HuggingFace config.json.")

    print("\nRequested provider resolutions:")
    for requested in ("nnapi", "xnnpack", "cpu"):
        try:
            selected = resolve_execution_provider(
                requested,
                available=runtime_providers,
                respect_environment=False,
            )
        except ConfigurationError as exc:
            print(f"  {requested}: unavailable ({exc})")
        else:
            print(f"  {requested}: {selected}")

    try:
        automatic = resolve_execution_provider(
            "auto",
            available=runtime_providers,
            respect_environment=False,
        )
    except ConfigurationError as exc:
        print(f"  auto: unavailable ({exc})")
    else:
        print(f"  auto: {automatic}")


if __name__ == "__main__":
    main()
