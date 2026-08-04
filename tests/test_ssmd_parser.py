from __future__ import annotations

from types import SimpleNamespace

from pykokoro.ssmd_parser import parse_ssmd_document


def test_parse_ssmd_document_forwards_spacy_settings(monkeypatch):
    captured: dict[str, object] = {}

    def fake_body_parser(*args: object, **kwargs: object):
        captured.update(kwargs)
        return 0.0, []

    monkeypatch.setattr("pykokoro.ssmd_parser._parse_ssmd_body_to_segments", fake_body_parser)

    parsed = parse_ssmd_document(
        "Hello.",
        lang="en-us",
        spacy_model="en_core_web_sm",
        model_size="sm",
        use_spacy=True,
    )

    assert parsed.segments == ()
    assert captured["spacy_model"] == "en_core_web_sm"
    assert captured["model_size"] == "sm"
    assert captured["use_spacy"] is True


def test_parse_ssmd_document_keeps_sentence_diagnostics(monkeypatch):
    diagnostic = SimpleNamespace(selected_model="en_core_web_lg", selected_model_size="lg")

    def fake_body_parser(*args: object, **kwargs: object):
        sink = kwargs["sentence_diagnostics"]
        sink.append(diagnostic)
        return 0.0, []

    monkeypatch.setattr("pykokoro.ssmd_parser._parse_ssmd_body_to_segments", fake_body_parser)

    parsed = parse_ssmd_document("Hello.")

    assert parsed.sentence_diagnostics is diagnostic
