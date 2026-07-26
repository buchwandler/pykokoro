from pykokoro import mixed_language_handler
from pykokoro.mixed_language_handler import MixedLanguageHandler
from pykokoro.tokenizer import TokenizerConfig


def _make_handler() -> MixedLanguageHandler:
    config = TokenizerConfig(
        use_mixed_language=True,
        mixed_language_primary="en-us",
        mixed_language_allowed=["en-us", "de"],
    )
    return MixedLanguageHandler(config)


def test_mixed_language_preserves_existing_annotations(monkeypatch):
    handler = _make_handler()
    calls: list[str] = []

    def fake_preprocess_multilang(**kwargs):
        calls.append(kwargs["text"])
        return f'[{kwargs["text"]}]{{lang="de"}}'

    monkeypatch.setattr(mixed_language_handler, "preprocess_multilang", fake_preprocess_multilang)

    text = 'Hello[Bonjour]{lang="fr"}World'
    result = handler.preprocess_text(text, default_language="en-us")

    assert calls == ["Hello", "World"]
    assert result == '[Hello]{lang="de"}[Bonjour]{lang="fr"}[World]{lang="de"}'


def test_mixed_language_all_annotated_returns_original(monkeypatch):
    handler = _make_handler()
    calls: list[str] = []

    def fake_preprocess_multilang(**kwargs):
        calls.append(kwargs["text"])
        return f'[{kwargs["text"]}]{{lang="de"}}'

    monkeypatch.setattr(mixed_language_handler, "preprocess_multilang", fake_preprocess_multilang)

    text = '[Bonjour]{lang="fr"}'
    result = handler.preprocess_text(text, default_language="en-us")

    assert calls == []
    assert result == text
