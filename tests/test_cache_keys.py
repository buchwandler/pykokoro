from dataclasses import asdict

from pykokoro.pipeline import KokoroPipeline
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.runtime.cache import make_g2p_key
from pykokoro.tokenizer import TokenizerConfig


def test_make_g2p_key_changes_with_is_phonemes():
    base = make_g2p_key(
        text="Hello",
        lang="en-us",
        is_phonemes=False,
        tokenizer_config=None,
        phoneme_override=None,
        kokorog2p_version="1.0",
    )
    alt = make_g2p_key(
        text="Hello",
        lang="en-us",
        is_phonemes=True,
        tokenizer_config=None,
        phoneme_override=None,
        kokorog2p_version="1.0",
    )

    assert base != alt


def test_kokoro_key_changes_with_model_quality():
    pipeline = KokoroPipeline(PipelineConfig())
    key_fp32 = pipeline._kokoro_key(PipelineConfig(model_quality="fp32"))
    key_fp16 = pipeline._kokoro_key(PipelineConfig(model_quality="fp16"))

    assert key_fp32 != key_fp16


def test_make_g2p_key_changes_with_runtime_frontend_contract():
    base = {
        "text": "Brücke",
        "lang": "de",
        "is_phonemes": False,
        "tokenizer_config": None,
        "phoneme_override": None,
        "kokorog2p_version": "1.0",
        "model_quality": "fp32",
        "model_source": "github",
        "model_variant": "de-crane",
        "frontend": "german-ipa-v1",
        "phoneme_postprocess": "german-short-u-to-y",
    }

    native = make_g2p_key(**base, g2p_backend="kokorog2p")
    espeak = make_g2p_key(**base, g2p_backend="espeak")

    assert native != espeak


def test_make_g2p_key_changes_with_named_lexicons():
    base = {
        "text": "Haus",
        "lang": "de",
        "is_phonemes": False,
        "phoneme_override": None,
        "kokorog2p_version": "test",
    }

    gold = make_g2p_key(
        **base,
        tokenizer_config=asdict(TokenizerConfig(lexicons=("gold",))),
    )
    crane = make_g2p_key(
        **base,
        tokenizer_config=asdict(TokenizerConfig(lexicons=("crane",))),
    )

    assert gold != crane
