"""Request-engine imports do not depend on document and composition packages."""

from __future__ import annotations

import subprocess
import sys


def test_request_engine_imports_without_document_extras() -> None:
    code = """
import importlib.abc
import sys
blocked = {"audiocompose", "utterplan", "ssmd"}
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise ModuleNotFoundError("blocked optional document dependency", name=fullname)
        return None
sys.meta_path.insert(0, Blocker())
import pykokoro
from pykokoro import SynthesisConfig, SynthesisSegment, RenderedSegment, KokoroSynthesizer
assert SynthesisConfig and SynthesisSegment and RenderedSegment and KokoroSynthesizer
assert not hasattr(pykokoro, "KokoroPipeline")
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
