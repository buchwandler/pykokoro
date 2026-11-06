from __future__ import annotations

import builtins
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from kokorog2p import phonemes_to_ids

from pykokoro import KokoroSynthesizer
from pykokoro.exceptions import InvalidVoiceError
from pykokoro.frontend_contracts import FRONTEND_FIXTURES
from pykokoro.model_profiles import (
    IMPLEMENTED_FRONTENDS,
    IMPLEMENTED_LAYOUTS,
    get_model_profile,
    profile_for_language,
    profile_for_voice,
)
from pykokoro.prepared_g2p import PreparedG2PAdapter
from pykokoro.synthesis_config import SynthesisConfig, resolve_synthesis_config
from pykokoro.synthesis_types import SynthesisSegment
from pykokoro.tokenizer import N_TOKENS, TokenizerConfig

_ENGLISH_VOICES = (
    "AV_ariala",
    "AV_brody",
    "AV_david",
    "AV_elliott",
    "AV_ralph",
    "AV_cori",
    "AV_phil",
    "AV_helen",
)
_GERMAN_VOICES = (
    "AV_thorsten",
    "AV_bernd",
    "AV_eva_k",
    "AV_karlsson",
    "AV_kerstin",
    "AV_mucky",
    "AV_hokuspokus",
    "AV_julia",
)


def test_akinvox_profile_is_ready_and_selectable_by_all_named_voices() -> None:
    profile = get_model_profile("akinvox-multispeaker-v1", "github")

    assert profile.support_status == "ready"
    assert profile.runtime_available is True
    assert profile.layout == "language-router-onnx-v1"
    assert profile.frontend == "akinvox-multispeaker-v1"
    assert profile.frontend in IMPLEMENTED_FRONTENDS
    assert profile.layout in IMPLEMENTED_LAYOUTS
    assert profile.language_codes == ("en", "en-us", "en-gb", "de", "de-de", "de-at", "de-ch")
    assert profile.default_voice == "AV_ariala"
    assert profile.voice_names == _ENGLISH_VOICES + _GERMAN_VOICES
    assert profile.voice_languages == {
        **dict.fromkeys(_ENGLISH_VOICES, ("en",)),
        **dict.fromkeys(_GERMAN_VOICES, ("de",)),
    }
    assert profile.auto_select_for_language is False
    assert profile_for_language("en") is not profile
    assert profile_for_language("de") is not profile

    for voice in profile.voice_names:
        assert profile_for_voice(voice) is profile
        language = "en-us" if voice in _ENGLISH_VOICES else "de-de"
        resolved = resolve_synthesis_config(SynthesisConfig(voice=voice), language=language)
        assert resolved.model_variant == "akinvox-multispeaker-v1"
        assert resolved.model_source == "github"
        assert resolved.voice == voice


@pytest.mark.parametrize(
    "fixture_key",
    ("akinvox-multispeaker-v1", "akinvox-multispeaker-v1-de"),
)
def test_akinvox_frontend_matches_release_ready_phoneme_and_token_fixtures(
    fixture_key: str,
) -> None:
    fixture = FRONTEND_FIXTURES[fixture_key]
    config = SynthesisConfig(
        model_source="github",
        model_variant="akinvox-multispeaker-v1",
        tokenizer_config=TokenizerConfig(use_spacy=False),
    )
    request = SynthesisSegment(
        id="akinvox-frontend-fixture",
        text=fixture.text,
        language=fixture.language,
    )

    prepared = PreparedG2PAdapter().phonemize(request, config)
    expected_ids = tuple(phonemes_to_ids(fixture.diagnostic_phonemes, model="1.0"))

    assert fixture.release_ready is True
    assert fixture.diagnostic_backend == "kokorog2p"
    assert prepared.phonemes == fixture.diagnostic_phonemes
    assert prepared.token_ids == expected_ids
    assert prepared.token_ids
    assert all(0 < token_id < N_TOKENS for token_id in prepared.token_ids)


def test_akinvox_frontend_does_not_import_upstream_package(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "kokoro_multispeaker" or name.startswith("kokoro_multispeaker."):
            raise AssertionError("PyKokoro must not import kokoro_multispeaker")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    config = SynthesisConfig(
        model_source="github",
        model_variant="akinvox-multispeaker-v1",
        tokenizer_config=TokenizerConfig(use_spacy=False),
    )
    request = SynthesisSegment(
        id="akinvox-upstream-import-check",
        text="Welcome to our next story.",
        language="en-us",
    )

    assert PreparedG2PAdapter().phonemize(request, config).token_ids


@pytest.mark.parametrize(("voice", "language"), [("AV_cori", "de-DE"), ("AV_julia", "en-US")])
def test_akinvox_named_voice_rejects_mismatched_language(voice: str, language: str) -> None:
    with pytest.raises(InvalidVoiceError, match=rf"{voice}.*does not support language"):
        resolve_synthesis_config(
            SynthesisConfig(
                model_source="github",
                model_variant="akinvox-multispeaker-v1",
                voice=voice,
            ),
            language=language,
        )


def test_akinvox_router_default_voice_rejects_german_request() -> None:
    with pytest.raises(InvalidVoiceError, match="AV_ariala.*does not support language"):
        resolve_synthesis_config(
            SynthesisConfig(model_source="github", model_variant="akinvox-multispeaker-v1"),
            language="de-DE",
        )


def test_akinvox_voicepack_without_language_metadata_remains_user_owned() -> None:
    from pykokoro.voice_pack import KokoroVoicePack

    voice_pack = KokoroVoicePack.from_array(np.zeros((510, 1, 256), dtype=np.float32))
    resolved = resolve_synthesis_config(
        SynthesisConfig(
            model_source="github",
            model_variant="akinvox-multispeaker-v1",
            voice=voice_pack,
        ),
        language="de-DE",
    )

    assert resolved.voice is voice_pack


@pytest.mark.slow
def test_akinvox_local_published_release_audio_smoke() -> None:
    manifest_value = os.environ.get("PYKOKORO_AKINVOX_RELEASE_MANIFEST")
    if not manifest_value:
        pytest.skip("set PYKOKORO_AKINVOX_RELEASE_MANIFEST to a local release manifest")
    manifest_path = Path(manifest_value).expanduser()
    if not manifest_path.is_file():
        pytest.fail(f"local AKinVox release manifest does not exist: {manifest_path}")

    config = SynthesisConfig(
        model_source="github",
        model_variant="akinvox-multispeaker-v1",
        model_quality="fp32",
        release_manifest_path=manifest_path,
        provider="cpu",
        tokenizer_config=TokenizerConfig(
            lexicons=(),
            lexicon_data_policy="installed-only",
            use_spacy=False,
        ),
    )
    with KokoroSynthesizer(config) as synthesizer:
        english = synthesizer.synthesize_text(
            "Welcome to our next story.", language="en-us", voice="AV_ariala"
        )
        german = synthesizer.synthesize_text(
            "Willkommen zu unserer nächsten Geschichte.",
            language="de-de",
            voice="AV_thorsten",
        )

    assert english.audio.size > 0
    assert german.audio.size > 0
    assert np.isfinite(english.audio).all()
    assert np.isfinite(german.audio).all()
