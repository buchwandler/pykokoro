import pytest

from pykokoro.language_routing import LanguageRoutingConfig, resolve_language_routing


def test_language_routing_normalizes_and_deduplicates_supported_codes():
    config = LanguageRoutingConfig(mode="auto", languages=("de", "EN_us", "de-de"))

    assert config.languages == ("de-de", "en-us")
    assert resolve_language_routing(config) == {"mode": "auto", "languages": config.languages}


def test_language_routing_requires_multiple_supported_languages_for_auto():
    with pytest.raises(ValueError, match="at least two"):
        LanguageRoutingConfig(mode="auto", languages=("en-us",))
    with pytest.raises(ValueError, match="unsupported"):
        LanguageRoutingConfig(mode="auto", languages=("en-us", "unknown"))
    with pytest.raises(TypeError, match="not a string"):
        LanguageRoutingConfig(mode="auto", languages="en-us")  # type: ignore[arg-type]


def test_language_routing_resolver_has_no_document_header_input():
    assert resolve_language_routing(None) is None
    assert resolve_language_routing(LanguageRoutingConfig()) is None
