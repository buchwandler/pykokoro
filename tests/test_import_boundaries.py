"""Tests for dependency-light imports and capability diagnostics."""

from __future__ import annotations

import subprocess
import sys

import pytest

from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter


def test_lightweight_imports_do_not_require_onnxruntime() -> None:
    code = (
        "import builtins\n"
        "real_import = builtins.__import__\n"
        "def blocked(name, *args, **kwargs):\n"
        "    if name == 'onnxruntime' or name.startswith('onnxruntime.'):\n"
        "        raise ModuleNotFoundError('blocked onnxruntime', name='onnxruntime')\n"
        "    return real_import(name, *args, **kwargs)\n"
        "builtins.__import__ = blocked\n"
        "import pykokoro.runtime.cache\n"
        "import pykokoro.generation_config\n"
        "import pykokoro.model_assets\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_custom_stage_pipeline_does_not_require_onnxruntime() -> None:
    code = (
        "import builtins\n"
        "real_import = builtins.__import__\n"
        "def blocked(name, *args, **kwargs):\n"
        "    if name == 'onnxruntime' or name.startswith('onnxruntime.'):\n"
        "        raise ModuleNotFoundError('blocked onnxruntime', name='onnxruntime')\n"
        "    return real_import(name, *args, **kwargs)\n"
        "builtins.__import__ = blocked\n"
        "from pykokoro.pipeline import KokoroPipeline\n"
        "from pykokoro.pipeline_config import PipelineConfig\n"
        "assert KokoroPipeline(PipelineConfig())\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_default_pipeline_reports_missing_onnxruntime_extra() -> None:
    code = (
        "import builtins\n"
        "real_import = builtins.__import__\n"
        "def blocked(name, *args, **kwargs):\n"
        "    if name == 'onnxruntime' or name.startswith('onnxruntime.'):\n"
        "        raise ModuleNotFoundError('blocked onnxruntime', name='onnxruntime')\n"
        "    return real_import(name, *args, **kwargs)\n"
        "builtins.__import__ = blocked\n"
        "from pykokoro.pipeline import KokoroPipeline\n"
        "from pykokoro.pipeline_config import PipelineConfig\n"
        "try:\n"
        "    KokoroPipeline(PipelineConfig()).run('Hello.', lang='en-us')\n"
        "except RuntimeError as exc:\n"
        "    assert 'pykokoro[cpu]' in str(exc)\n"
        "else:\n"
        "    raise AssertionError('default backend unexpectedly succeeded')\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_missing_kokorog2p_has_distinct_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = KokoroG2PAdapter()
    real_import = __import__

    def blocked_import(name, *args, **kwargs):
        if name == "kokorog2p":
            raise ModuleNotFoundError("missing kokorog2p", name="kokorog2p")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked_import)
    with pytest.raises(RuntimeError, match="not installed"):
        adapter._load()
