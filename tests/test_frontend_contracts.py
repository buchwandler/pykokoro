import pytest

from pykokoro.frontend_contracts import (
    FRONTEND_FIXTURES,
    named_lexicons_for_frontend,
    require_frontend,
)
from pykokoro.model_profiles import get_model_profile


@pytest.mark.parametrize(
    ("variant", "frontend", "g2p_backend"),
    [
        ("vi-contextbox", "vig2p-v1", "espeak"),
        ("vi-anphunl", "vig2p-v1", "espeak"),
        ("de-crane", "german-ipa-v1", "kokorog2p"),
        ("he-hebrew-nc", "hebrew-g2p-v1", "espeak"),
    ],
)
def test_experimental_frontend_contracts_are_explicit(variant, frontend, g2p_backend):
    profile = get_model_profile(variant, "github")
    assert profile.frontend == frontend
    assert profile.g2p_backend == g2p_backend
    assert profile.frontend_experimental
    assert profile.sample_rate == 24000
    fixture = FRONTEND_FIXTURES[variant]
    assert fixture.language == profile.language_codes[0]
    assert fixture.diagnostic_phonemes
    with pytest.raises(ValueError, match="requires"):
        require_frontend(variant, allow_experimental=False)
    assert require_frontend(variant, allow_experimental=True) == frontend


def test_nabra_frontend_is_release_ready():
    from kokorog2p import get_g2p

    profile = get_model_profile("ar-nabra", "github")
    fixture = FRONTEND_FIXTURES["ar-nabra"]
    assert not profile.frontend_experimental
    assert require_frontend("ar-nabra", allow_experimental=False) == profile.frontend
    g2p = get_g2p(
        language=fixture.language,
        model_profile="nabra-82m-v0.1",
        load_gold=False,
        load_silver=False,
    )
    assert g2p.get_target_model() == "nabra-82m-v0.1"
    assert g2p.phonemize(fixture.text) == fixture.diagnostic_phonemes


def test_native_profiles_are_not_marked_experimental():
    assert not get_model_profile("v1.2-de-martin", "github").frontend_experimental


def test_named_lexicon_inventory_has_explicit_unknown_and_empty_states() -> None:
    assert named_lexicons_for_frontend("kokorog2p-de-thorsten-v1") == ("gold", "crane")
    assert named_lexicons_for_frontend("nabra-arabic-v1") == ()
    assert named_lexicons_for_frontend("unclassified-frontend") is None
