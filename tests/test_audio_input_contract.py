from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from kokorog2p import get_kokoro_vocab

from pykokoro.audio_generator import AudioGenerator
from pykokoro.tokenizer import Tokenizer


class Runtime:
    sample_rate = 24_000
    supports_timings = True
    cache_identity = ("kokoro:v1.0", "fp32", "cpu")

    def __init__(self) -> None:
        self.calls: list[tuple[list[int], np.ndarray, float, int | None]] = []

    def infer(self, token_ids, *, style, speed, seed=None):
        self.calls.append((list(token_ids), np.asarray(style), speed, seed))
        return SimpleNamespace(
            audio=np.zeros(8, dtype=np.float32),
            timings=np.ones(len(token_ids), dtype=np.float32),
        )

    def close(self) -> None:
        pass


def test_runtime_receives_model_ready_tokens_without_graph_padding() -> None:
    phonemes = "ㄋㄧ2ㄏㄠ3"
    tokenizer = Tokenizer(vocab_version="1.1", vocab=get_kokoro_vocab(model="1.1"))
    prepared_tokens = tokenizer.tokenize(phonemes)
    runtime = Runtime()
    generator = AudioGenerator(runtime=runtime, tokenizer=tokenizer, model_source="github")

    generator.generate_from_phonemes(
        phonemes,
        np.zeros((510, 256), dtype=np.float32),
        1.125,
    )

    assert len(runtime.calls) == 1
    token_ids, style, speed, seed = runtime.calls[0]
    assert token_ids == prepared_tokens
    assert style.shape == (1, 256)
    assert speed == 1.125
    assert seed is None


def test_runtime_cache_key_includes_style_and_speed() -> None:
    runtime = Runtime()
    generator = AudioGenerator(
        runtime=runtime, tokenizer=SimpleNamespace(tokenize=lambda text: [1, 2])
    )
    style = np.zeros((4, 256), dtype=np.float32)

    generator.generate_from_phonemes("ab", style, 1.0)
    generator.generate_from_phonemes("ab", style, 1.0)
    generator.generate_from_phonemes("ab", style, 1.1)

    assert len(runtime.calls) == 2


def test_runtime_identity_changes_cache_key() -> None:
    style = np.zeros((4, 256), dtype=np.float32)
    first = Runtime()
    second = Runtime()
    second.cache_identity = ("kokoro:v1.1", "fp32", "cpu")
    generator = AudioGenerator(
        runtime=first, tokenizer=SimpleNamespace(tokenize=lambda text: [1, 2])
    )
    generator.generate_from_phonemes("ab", style, 1.0)

    generator._runtime = second
    generator.generate_from_phonemes("ab", style, 1.0)

    assert len(first.calls) == 1
    assert len(second.calls) == 1
