from __future__ import annotations

import pykokoro
from pykokoro.generation_config import GenerationConfig
from pykokoro.synthesis_config import SynthesisConfig, resolve_synthesis_config
from pykokoro.synthesis_identity import SynthesisIdentity, build_synthesis_identity


def _identity(config: SynthesisConfig | None = None) -> SynthesisIdentity:
    resolved = resolve_synthesis_config(
        config or SynthesisConfig(), language="en-us", voice="af_heart"
    )
    return build_synthesis_identity(resolved, language="en-us", voice=resolved.voice)


def test_resolved_synthesis_identity_is_stable_and_output_sensitive() -> None:
    base = _identity()
    same = _identity()
    faster = _identity(SynthesisConfig(generation=GenerationConfig(speed=1.2)))

    assert base.package_version == pykokoro.__version__
    assert base.model_id == "github:v1.0:fp32"
    assert base.voice == "af_heart"
    assert base.language == "en-us"
    assert base.short_sentence == '{"enabled":false}'
    assert base.cache_key == same.cache_key
    assert base.cache_key != faster.cache_key
    assert len(base.cache_key) == 64


def test_synthesis_identity_includes_active_long_text_policy() -> None:
    base = _identity()
    sentence = _identity(SynthesisConfig(long_text_split="sentence"))
    automatic_spacy = _identity(
        SynthesisConfig(long_text_split="sentence", long_text_use_spacy=None)
    )
    inactive_spacy_setting = _identity(SynthesisConfig(long_text_use_spacy=True))

    assert base.cache_key == inactive_spacy_setting.cache_key
    assert base.cache_key != sentence.cache_key
    assert sentence.cache_key != automatic_spacy.cache_key


def test_synthesis_identity_does_not_include_transient_model_paths(tmp_path) -> None:
    first = _identity(SynthesisConfig(model_path=tmp_path / "one.onnx"))
    second = _identity(SynthesisConfig(model_path=tmp_path / "two.onnx"))

    assert first.cache_key == second.cache_key
