"""Application-level provisioning for Lexphon-backed KokoroG2P lexicons."""

from __future__ import annotations

import logging
from threading import Lock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .tokenizer import TokenizerConfig

logger = logging.getLogger(__name__)

_LEXPHON_INSTALL_LOCK = Lock()


def required_lexphon_ids(
    language: str,
    config: TokenizerConfig,
) -> tuple[str, ...]:
    """Return selected Lexphon logical IDs for a KokoroG2P configuration."""
    from kokorog2p.lexicons import get_lexicon_spec, normalize_lexicon_selection

    names = normalize_lexicon_selection(
        language,
        config.lexicons,
        load_gold=config.load_gold,
        load_silver=config.load_silver,
    )
    return tuple(
        spec.id
        for name in names
        if (spec := get_lexicon_spec(language, name)).backend == "lexphon"
    )


def _missing_lexphon_ids(store: Any, required: tuple[str, ...]) -> list[str]:
    from lexphon import LexiconNotInstalledError

    missing: list[str] = []
    for identifier in required:
        try:
            store.path(identifier)
        except LexiconNotInstalledError:
            missing.append(identifier)
    return missing


def install_missing_lexphon_data(
    language: str,
    config: TokenizerConfig,
) -> tuple[str, ...]:
    """Install selected Lexphon assets that are absent from the local store."""
    from lexphon import DataStore
    from lexphon.catalog import load_catalog

    required = required_lexphon_ids(language, config)
    if not required:
        return ()

    store = DataStore()
    if not _missing_lexphon_ids(store, required):
        return ()

    with _LEXPHON_INSTALL_LOCK:
        store = DataStore()
        missing = _missing_lexphon_ids(store, required)
        if not missing:
            return ()

        catalog = load_catalog()
        logger.info("Installing Lexphon data: %s", ", ".join(missing))
        for identifier in missing:
            store.install(catalog.artifact(identifier))
        logger.info("Installed Lexphon data: %s", ", ".join(missing))
        return tuple(missing)


def create_g2p_with_lexphon_retry(
    g2p_module: Any,
    *,
    language: str,
    config: TokenizerConfig,
    kwargs: dict[str, Any],
) -> Any:
    """Construct G2P, provisioning selected Lexphon data once when needed."""
    from lexphon import LexiconNotInstalledError

    try:
        return g2p_module.get_g2p(**kwargs)
    except LexiconNotInstalledError:
        if config.lexicon_data_policy != "auto":
            raise
        if not required_lexphon_ids(language, config):
            raise
        install_missing_lexphon_data(language, config)

    return g2p_module.get_g2p(**kwargs)
