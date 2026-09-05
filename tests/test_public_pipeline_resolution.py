from __future__ import annotations

import socket
import subprocess
import sys
import textwrap

import pytest

import pykokoro
from pykokoro import GenerationConfig, PipelineConfig, resolve_pipeline_config
from pykokoro.pipeline_config import resolve_model_defaults


def test_resolve_pipeline_config_is_public() -> None:
    assert "resolve_pipeline_config" in pykokoro.__all__
    assert callable(pykokoro.resolve_pipeline_config)


def test_public_resolver_matches_internal_policy() -> None:
    cfg = PipelineConfig(generation=GenerationConfig(lang="de"))

    assert resolve_pipeline_config(cfg) == resolve_model_defaults(cfg)


def test_public_resolver_resolves_concrete_automatic_values() -> None:
    requested = PipelineConfig(generation=GenerationConfig(lang="de"))

    resolved = resolve_pipeline_config(requested)

    assert resolved.model_source is not None
    assert resolved.model_variant is not None
    assert resolved.model_quality is not None
    assert resolved.voice is not None
    assert requested.model_variant is None
    assert requested.model_source is None
    assert requested.model_quality is None
    assert requested.voice is None


def test_public_resolver_preserves_explicit_values() -> None:
    requested = PipelineConfig(
        voice="martin",
        model_source="github",
        model_variant="v1.2-de-martin",
        model_quality="fp32",
        generation=GenerationConfig(lang="de"),
    )

    resolved = resolve_pipeline_config(requested)

    assert resolved.voice == "martin"
    assert resolved.model_source == "github"
    assert resolved.model_variant == "v1.2-de-martin"
    assert resolved.model_quality == "fp32"


def test_public_resolver_is_idempotent() -> None:
    resolved = resolve_pipeline_config(PipelineConfig(generation=GenerationConfig(lang="de")))

    assert resolve_pipeline_config(resolved) == resolved


def test_public_resolver_requires_document_language() -> None:
    with pytest.raises(ValueError, match="document language"):
        resolve_pipeline_config(PipelineConfig())


def test_public_resolver_does_not_require_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access is not part of built-in resolution")

    monkeypatch.setattr(socket.socket, "connect", fail_network)

    resolved = resolve_pipeline_config(PipelineConfig(generation=GenerationConfig(lang="de")))

    assert resolved.model_variant == "v1.2-de-martin"


def test_public_resolver_does_not_import_runtime_modules() -> None:
    code = textwrap.dedent(
        """
        import sys

        import pykokoro
        from pykokoro import GenerationConfig, PipelineConfig, resolve_pipeline_config

        assert "onnxruntime" not in sys.modules
        assert "pykokoro.pipeline" not in sys.modules

        resolved = resolve_pipeline_config(
            PipelineConfig(generation=GenerationConfig(lang="de"))
        )

        assert resolved.model_variant == "v1.2-de-martin"
        assert "onnxruntime" not in sys.modules
        assert "pykokoro.pipeline" not in sys.modules
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)
