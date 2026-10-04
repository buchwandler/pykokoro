"""Regression tests for the dependency-light public request API contract."""

from __future__ import annotations

import subprocess
import sys

import pykokoro
from pykokoro.api_contract import REQUEST_API_VERSION, RequestApiContract, request_api_contract


def test_request_api_version_is_one() -> None:
    assert REQUEST_API_VERSION == 1
    assert pykokoro.REQUEST_API_VERSION == 1


def test_request_api_contract_declares_stable_capabilities() -> None:
    contract = request_api_contract()

    assert contract == RequestApiContract(
        version=1,
        request_type="SynthesisRequest",
        result_type="RenderedSegment",
        synthesizer_type="KokoroSynthesizer",
        prepare_method="prepare",
        synthesize_method="synthesize",
        supports_linguistic_tokens=True,
        supports_pronunciation_overrides=True,
        supports_whole_request_phonemes=True,
        supports_word_timings=True,
        supports_voice_level=True,
        caller_owns_text_boundaries=True,
    )
    assert pykokoro.RequestApiContract is RequestApiContract
    assert pykokoro.request_api_contract is request_api_contract


def test_request_api_contract_and_synthesizer_resolution_are_dependency_light() -> None:
    code = r"""
import importlib.abc
import sys

blocked = {
    "onnxvoice",
    "pykokoro.prepared_g2p",
    "pykokoro.reference_audio",
    "pykokoro.request_renderer",
}

class RuntimeImportBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError(f"request API probe imported runtime module {fullname}")
        return None

sys.meta_path.insert(0, RuntimeImportBlocker())
import pykokoro

assert pykokoro.REQUEST_API_VERSION == 1
contract = pykokoro.request_api_contract()
assert contract.request_type == "SynthesisRequest"
assert contract.result_type == "RenderedSegment"
assert contract.synthesizer_type == "KokoroSynthesizer"

synthesizer = pykokoro.KokoroSynthesizer
assert callable(synthesizer.prepare)
assert callable(synthesizer.synthesize)
assert not blocked.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr


REQUIRED_READIO_API = (
    "KokoroSynthesizer",
    "SynthesisConfig",
    "SynthesisRequest",
    "PronunciationOverride",
    "LinguisticToken",
    "VoiceLevelConfig",
    "SynthesisInputTooLongError",
    "ShortSentenceConfig",
)


def test_root_exports_resolve_for_readio_and_dependency_light_values_construct() -> None:
    for name in REQUIRED_READIO_API:
        assert getattr(pykokoro, name) is not None

    assert callable(pykokoro.KokoroSynthesizer.prepare)
    assert callable(pykokoro.KokoroSynthesizer.synthesize)

    request = pykokoro.SynthesisRequest(
        id="api-smoke",
        text="Hello world.",
        language="en-us",
    )
    config = pykokoro.SynthesisConfig(long_text_split="none", long_text_use_spacy=False)
    assert request.text == "Hello world."
    assert config.long_text_split == "none"
    assert config.long_text_use_spacy is False
