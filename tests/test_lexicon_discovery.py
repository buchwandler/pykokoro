from __future__ import annotations

from types import SimpleNamespace

import pykokoro
import pytest

from pykokoro import lexicon_discovery


def test_root_package_exports_lexicon_discovery() -> None:
    assert pykokoro.discover_lexicons is lexicon_discovery.discover_lexicons
    assert pykokoro.LexiconCapabilities is lexicon_discovery.LexiconCapabilities


@pytest.mark.parametrize(
    ("language", "selector", "asset_id"),
    (
        ("de", "gold", "de-de:gold"),
        ("de", "crane", "de-de:crane"),
        ("en-us", "gold", "en-us:gold"),
        ("en-gb", "gold", "en-gb:gold"),
        ("fr-fr", "gold", "fr-fr:gold"),
        ("pt-pt", "lexhint", "pt-pt:lexhint"),
    ),
)
def test_named_selectors_map_to_language_qualified_assets(
    monkeypatch: pytest.MonkeyPatch,
    language: str,
    selector: str,
    asset_id: str,
) -> None:
    monkeypatch.setattr(lexicon_discovery, "_installed_state", lambda _asset: False)

    result = lexicon_discovery.discover_lexicons(language=language)
    matching = [item for item in result.lexicons if item.selector == selector]

    assert matching
    assert matching[0].asset_id == asset_id
    assert matching[0].selector == selector


def test_base_language_keeps_regional_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lexicon_discovery, "_installed_state", lambda _asset: False)

    result = lexicon_discovery.discover_lexicons(language="EN_us")
    assert [(item.locale, item.selector) for item in result.lexicons] == [("en-US", "gold")]

    result = lexicon_discovery.discover_lexicons(language="en")
    assert [(item.locale, item.selector) for item in result.lexicons] == [
        ("en-GB", "gold"),
        ("en-US", "gold"),
    ]


def test_known_model_filter_and_unknown_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lexicon_discovery, "_installed_state", lambda _asset: False)
    monkeypatch.setattr(
        lexicon_discovery,
        "discover_models",
        lambda **_kwargs: SimpleNamespace(
            models=(
                SimpleNamespace(model_id="de-thorsten", lexicons=("gold", "crane")),
            )
        ),
    )

    known = lexicon_discovery.discover_lexicons(language="de", model_variant="de-thorsten")
    assert [item.selector for item in known.lexicons] == ["crane", "gold"]
    assert all(item.model_support == "known" for item in known.lexicons)
    assert all(item.models == ("de-thorsten",) for item in known.lexicons)

    monkeypatch.setattr(
        lexicon_discovery,
        "discover_models",
        lambda **_kwargs: SimpleNamespace(
            models=(SimpleNamespace(model_id="de-thorsten", lexicons=None),)
        ),
    )
    unknown = lexicon_discovery.discover_lexicons(language="de", model_variant="de-thorsten")
    assert {item.selector for item in unknown.lexicons} >= {"gold", "crane"}
    assert all(item.model_support == "unknown" for item in unknown.lexicons)


def test_discovery_never_installs_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class Store:
        def verify(self, asset_id: str) -> bool:
            calls.append(asset_id)
            return False

    monkeypatch.setattr(lexicon_discovery, "DataStore", Store, raising=False)
    monkeypatch.setattr(lexicon_discovery, "_installed_state", lambda asset: Store().verify(asset))

    result = lexicon_discovery.discover_lexicons(language="de", offline=True)
    assert result.offline is True
    assert calls
