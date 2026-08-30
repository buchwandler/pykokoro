import pathlib

import pytest

from pykokoro.exceptions import SSMDDocumentError
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.ssmd_config import SSMDPauseOverrides, resolve_pause_defaults
from pykokoro.stages.doc_parsers.ssmd import SsmdDocumentParser
from pykokoro.types import Trace


def test_review_fixture_uses_longest_paragraph_voice_change_default():
    text = pathlib.Path("tests/fixtures/ssmd_080_review_podcast.ssmd").read_text()
    doc = SsmdDocumentParser().parse(text, PipelineConfig(), Trace())
    assert doc.clean_text == ("Welcome to the review.\n\nThe proposed change improves authoring.")
    assert {span.attrs.get("voice_name") for span in doc.annotation_spans} >= {
        "af_sarah",
        "af_bella",
    }
    assert not doc.boundary_events


def test_explicit_break_beats_longer_header_default():
    text = "---\npause_defaults:\n  enabled: true\n  sentence: 2s\n---\nHello. ...100ms"
    doc = SsmdDocumentParser().parse(text, PipelineConfig(), Trace())
    assert any(event.duration_s == pytest.approx(0.1) for event in doc.boundary_events)


def test_pause_defaults_validate_and_merge():
    assert resolve_pause_defaults({"enabled": True, "sentence": "250ms"}).sentence == 0.25
    assert (
        resolve_pause_defaults(
            {"enabled": True, "sentence": "250ms"},
            SSMDPauseOverrides(sentence="1.5s"),
        ).sentence
        == 1.5
    )
    assert (
        resolve_pause_defaults(
            {"enabled": True, "sentence": "250ms"},
            SSMDPauseOverrides(enabled=False),
        ).enabled
        is False
    )


@pytest.mark.parametrize("value", [{"enabled": "yes"}, {"enabled": True}])
def test_invalid_pause_defaults_fail(value):
    with pytest.raises(SSMDDocumentError):
        resolve_pause_defaults(value)


def test_negative_pause_duration_fails():
    with pytest.raises(SSMDDocumentError):
        resolve_pause_defaults({"enabled": True, "sentence": "-1ms"})


def test_pause_parsing_is_deterministic():
    parser = SsmdDocumentParser()
    cfg = PipelineConfig()
    text = "---\npause_defaults:\n  enabled: true\n  sentence: 250ms\n---\nFirst. Second."
    expected = None
    for _ in range(200):
        current = [
            (event.pos, event.duration_s)
            for event in parser.parse(text, cfg, Trace()).boundary_events
        ]
        expected = current if expected is None else expected
        assert current == expected
