from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import g2lex
import lexphon
import pytest

from pykokoro.lexicon_data import (
    create_g2p_with_lexphon_retry,
    install_missing_lexphon_data,
    required_lexphon_ids,
)
from pykokoro.tokenizer import TokenizerConfig


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _local_catalog(tmp_path: Path) -> Any:
    source = tmp_path / "de.jsonl"
    source.write_text('{"word":"Haus","kind":"scalar","value":"haʊ̯s"}\n', encoding="utf-8")
    asset = tmp_path / "de.g2lex"
    g2lex.pack_file(
        source,
        asset,
        input_format="jsonl",
        source_id="test-de",
        metadata={"pronunciation_alphabet": "ipa"},
    )
    manifest = tmp_path / "de.manifest.json"
    _write_json(
        manifest,
        {
            "id": "de-de:gold",
            "contract_version": 1,
            "manifest_version": 1,
            "data_version": "test-1",
            "kind": "pronunciation",
            "language": "de-DE",
            "name": "gold",
            "phoneme_encoding": "ipa",
            "asset_sha256": _sha(asset),
        },
    )
    catalog_path = tmp_path / "catalog.json"
    _write_json(
        catalog_path,
        {
            "catalog_version": 1,
            "runtime_contract": "g2lex-data.catalog.v1",
            "artifacts": [
                {
                    "id": "de-de:gold",
                    "language": "de-DE",
                    "name": "gold",
                    "display_name": "gold",
                    "kind": "pronunciation",
                    "phoneme_encoding": "ipa",
                    "data_version": "test-1",
                    "release_tag": "test-1",
                    "manifest": {
                        "name": manifest.name,
                        "url": manifest.as_uri(),
                        "sha256": _sha(manifest),
                    },
                    "asset": {
                        "name": asset.name,
                        "url": asset.as_uri(),
                        "sha256": _sha(asset),
                        "size": asset.stat().st_size,
                        "format": "g2lex",
                    },
                    "source": {"provider": "test"},
                }
            ],
        },
    )
    return lexphon.catalog.load_catalog(str(catalog_path))


def test_required_lexphon_ids_resolves_selected_metadata() -> None:
    config = TokenizerConfig(lexicons=("gold", "crane"))

    assert required_lexphon_ids("de", config) == ("de-de:gold", "de-de:crane")


def test_install_checks_missing_assets_and_skips_catalog_when_warm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lexphon import LexiconNotInstalledError

    class FakeStore:
        installed = {"de-de:gold"}
        installed_calls: list[str] = []

        def path(self, identifier: str) -> Path:
            self.installed_calls.append(identifier)
            if identifier not in self.installed:
                raise LexiconNotInstalledError(identifier)
            return Path(identifier)

        def install(self, artifact: object) -> None:
            raise AssertionError("warm provisioning must not install")

    store = FakeStore()
    monkeypatch.setattr(lexphon, "DataStore", lambda: store)
    monkeypatch.setattr(
        "lexphon.catalog.load_catalog",
        lambda: pytest.fail("warm provisioning must not load the catalog"),
    )

    assert install_missing_lexphon_data("de", TokenizerConfig(lexicons=("gold",))) == ()
    assert store.installed_calls == ["de-de:gold"]

def test_install_rechecks_under_lock_after_another_installer_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lexphon import LexiconNotInstalledError

    class RaceStore:
        calls = 0

        def path(self, identifier: str) -> Path:
            self.calls += 1
            if self.calls == 1:
                raise LexiconNotInstalledError(identifier)
            return Path(identifier)

        def install(self, artifact: object) -> None:
            raise AssertionError("the in-lock recheck should observe the installed asset")

    store = RaceStore()
    monkeypatch.setattr(lexphon, "DataStore", lambda: store)
    monkeypatch.setattr(
        "lexphon.catalog.load_catalog",
        lambda: pytest.fail("the in-lock recheck should avoid catalog access"),
    )

    assert install_missing_lexphon_data("de", TokenizerConfig(lexicons=("gold",))) == ()
    assert store.calls == 2





def test_install_uses_local_catalog_and_g2lex_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lexphon import DataStore

    catalog = _local_catalog(tmp_path)
    store = DataStore(tmp_path / "store")
    monkeypatch.setattr(lexphon, "DataStore", lambda: store)
    monkeypatch.setattr("lexphon.catalog.load_catalog", lambda: catalog)

    installed = install_missing_lexphon_data("de", TokenizerConfig(lexicons=("gold",)))

    assert installed == ("de-de:gold",)
    assert store.path("de-de:gold").is_file()
    assert store.verify("de-de:gold")


def test_retry_provisions_once_and_retries_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from lexphon import LexiconNotInstalledError

    calls = 0
    installations: list[tuple[str, str]] = []

    def get_g2p(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise LexiconNotInstalledError("de-de:gold")
        return SimpleNamespace(kwargs=kwargs)

    monkeypatch.setattr(
        "pykokoro.lexicon_data.install_missing_lexphon_data",
        lambda language, config: installations.append((language, config.lexicon_data_policy)) or ("de-de:gold",),
    )

    result = create_g2p_with_lexphon_retry(
        SimpleNamespace(get_g2p=get_g2p),
        language="de",
        config=TokenizerConfig(lexicons=("gold",)),
        kwargs={"language": "de"},
    )

    assert result.kwargs == {"language": "de"}
    assert calls == 2
    assert installations == [("de", "auto")]


def test_installed_only_propagates_missing_error_without_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lexphon import LexiconNotInstalledError

    error = LexiconNotInstalledError("de-de:gold")
    install = monkeypatch.setattr
    install(
        "pykokoro.lexicon_data.install_missing_lexphon_data",
        lambda *args: pytest.fail("installed-only must not install"),
    )

    def get_g2p(**kwargs: object) -> object:
        raise error

    with pytest.raises(LexiconNotInstalledError) as caught:
        create_g2p_with_lexphon_retry(
            SimpleNamespace(get_g2p=get_g2p),
            language="de",
            config=TokenizerConfig(lexicons=("gold",), lexicon_data_policy="installed-only"),
            kwargs={},
        )
    assert caught.value is error


def test_retry_propagates_non_install_errors_and_does_not_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RuntimeError("catalog failure")
    monkeypatch.setattr(
        "pykokoro.lexicon_data.install_missing_lexphon_data",
        lambda *args: pytest.fail("non-install errors must not provision"),
    )
    calls = 0

    def get_g2p(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(RuntimeError) as caught:
        create_g2p_with_lexphon_retry(
            SimpleNamespace(get_g2p=get_g2p),
            language="de",
            config=TokenizerConfig(lexicons=("gold",)),
            kwargs={},
        )
    assert caught.value is error
    assert calls == 1


def test_retry_does_not_attempt_a_third_g2p_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lexphon import LexiconNotInstalledError

    error = LexiconNotInstalledError("de-de:gold")
    calls = 0
    monkeypatch.setattr(
        "pykokoro.lexicon_data.install_missing_lexphon_data",
        lambda *args: ("de-de:gold",),
    )

    def get_g2p(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(LexiconNotInstalledError) as caught:
        create_g2p_with_lexphon_retry(
            SimpleNamespace(get_g2p=get_g2p),
            language="de",
            config=TokenizerConfig(lexicons=("gold",)),
            kwargs={},
        )
    assert caught.value is error
    assert calls == 2
