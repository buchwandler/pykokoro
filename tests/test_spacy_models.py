from pykokoro.spacy_models import (
    normalize_spacy_language,
    resolve_configured_spacy_model,
    resolve_spacy_model,
)


def test_resolve_spacy_model_with_sizes_and_langs():
    assert resolve_spacy_model("en-us", size="md") == "en_core_web_md"
    assert resolve_spacy_model("de", size="sm") == "de_core_news_sm"
    assert resolve_spacy_model("zh-cn", size="trf") == "zh_core_web_trf"


def test_resolve_spacy_model_normalizes_aliases():
    assert normalize_spacy_language("cmn") == "zh"
    assert resolve_spacy_model("cmn", size="lg") == "zh_core_web_lg"


def test_resolve_configured_spacy_model_explicit_and_auto():
    assert (
        resolve_configured_spacy_model(
            spacy_model="en_core_web_trf",
            lang="en-us",
            size="md",
        )
        == "en_core_web_trf"
    )
    assert (
        resolve_configured_spacy_model(
            spacy_model="auto",
            lang="fr-fr",
            size="md",
        )
        == "fr_core_news_md"
    )
