from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from pykokoro.audio_generator import AudioGenerator
from pykokoro.short_sentence_handler import ShortSentenceConfig


class _Tokenizer:
    def tokenize(self, phonemes: str) -> list[int]:
        return list(range(1, len(phonemes) + 1))


class _Session:
    supports_timings = False
    sample_rate = 24_000

    def infer(self, token_ids, *, style, speed, seed=None):
        _ = token_ids, style, speed, seed
        return SimpleNamespace(audio=np.zeros(4, dtype=np.float32), timings=None)


class _TimestampSession(_Session):
    supports_timings = True

    def infer(self, token_ids, *, style, speed, seed=None):
        _ = token_ids, style, speed, seed
        return SimpleNamespace(
            audio=np.zeros(4, dtype=np.float32),
            timings=np.ones(4, dtype=np.float32),
        )


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


class _UndeclaredTimestampSession:
    sample_rate = 24_000
    cache_identity = "undeclared-timestamp"

    def infer(self, token_ids, *, style, speed, seed=None):
        _ = token_ids, style, speed, seed
        return SimpleNamespace(
            audio=np.zeros(4, dtype=np.float32),
            timings=np.ones(4, dtype=np.float32),
        )


def test_unknown_timestamp_logging_reports_observed_count_without_payload(caplog) -> None:
    generator = AudioGenerator(
        session=cast(Any, _UndeclaredTimestampSession()),
        tokenizer=cast(Any, _Tokenizer()),
        short_sentence_config=ShortSentenceConfig(resolve_mode="phrase"),
    )
    caplog.set_level(logging.DEBUG, logger="pykokoro.audio_generator")

    config = generator._resolve_short_sentence_config(None)
    assert config is not None
    assert config.resolve_mode == "phrase"
    generator._run_onnx("abc", np.zeros((16, 256), dtype=np.float32), 1.0)

    messages = [record.message for record in caplog.records if "inference.finish" in record.message]
    assert len(messages) == 1
    assert "timing_values=4" in messages[0]
    assert "timestamp_support=observed" in messages[0]
    assert "timings=[" not in messages[0]
    assert not any(
        "Loaded ONNX model has no timestamp output" in record.message for record in caplog.records
    )


def test_implicit_short_sentence_default_is_disabled(caplog, capsys):
    generator = AudioGenerator(
        session=cast(Any, _Session()),
        tokenizer=cast(Any, _Tokenizer()),
    )
    caplog.set_level(logging.WARNING, logger="pykokoro.audio_generator")

    config = generator._resolve_short_sentence_config(None)

    assert config is None
    assert not caplog.records
    assert capsys.readouterr().out == ""


def test_explicit_phrase_mode_without_timestamps_warns_once(caplog):
    generator = AudioGenerator(
        session=cast(Any, _Session()),
        tokenizer=cast(Any, _Tokenizer()),
        short_sentence_config=ShortSentenceConfig(resolve_mode="phrase"),
    )
    caplog.set_level(logging.WARNING, logger="pykokoro.audio_generator")

    first = generator._resolve_short_sentence_config(None)
    second = generator._resolve_short_sentence_config(None)

    assert first is not None
    assert second is not None
    assert first.resolve_mode == "wrap"
    assert second.resolve_mode == "wrap"

    messages = [
        record.message for record in caplog.records if "no timestamp output" in record.message
    ]
    assert len(messages) == 1


def test_implicit_short_sentence_default_stays_disabled_when_timestamps_exist():
    generator = AudioGenerator(
        session=cast(Any, _TimestampSession()),
        tokenizer=cast(Any, _Tokenizer()),
    )

    assert generator._resolve_short_sentence_config(None) is None


def test_explicit_enable_uses_short_sentence_configuration():
    generator = AudioGenerator(
        session=cast(Any, _TimestampSession()),
        tokenizer=cast(Any, _Tokenizer()),
    )

    config = generator._resolve_short_sentence_config(True)

    assert config is not None
    assert config.enabled is True
    assert config.resolve_mode == "randomized-phrase"
