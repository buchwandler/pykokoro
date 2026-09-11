from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from pykokoro.audio_generator import AudioGenerator


class _Tokenizer:
    def tokenize(self, phonemes: str) -> list[int]:
        return list(range(1, len(phonemes) + 1))


class _Session:
    def get_inputs(self) -> list[Any]:
        return [
            SimpleNamespace(name="tokens", type="tensor(int64)"),
            SimpleNamespace(name="style", type="tensor(float)"),
            SimpleNamespace(name="speed", type="tensor(float)"),
        ]

    def get_outputs(self) -> list[Any]:
        return []

    def run(self, _outputs, _inputs) -> list[np.ndarray]:
        return [np.zeros((1, 4), dtype=np.float32)]


def test_inference_logging_reports_counts_cache_and_runtime_without_payload(caplog) -> None:
    generator = AudioGenerator(
        session=cast(Any, _Session()),
        tokenizer=cast(Any, _Tokenizer()),
    )
    source = "PRIVATE_PHONEME_PAYLOAD"
    caplog.set_level(logging.DEBUG, logger="pykokoro.audio_generator")

    generator._run_onnx(source, np.zeros((16, 256), dtype=np.float32), 1.0)
    generator._run_onnx(source, np.zeros((16, 256), dtype=np.float32), 1.0, attempt_kind="retry")

    messages = [record.message for record in caplog.records if "inference.finish" in record.message]
    assert len(messages) == 2
    assert "cache_hit=False" in messages[0]
    assert "cache_hit=True" in messages[1]
    assert "attempt=retry" in messages[1]
    for message in messages:
        assert "phonemes=23" in message
        assert "tokens=23" in message
        assert "samples=4" in message
        assert "runtime_ms=" in message
        assert source not in message
