import pytest

from pykokoro.exceptions import SSMDDocumentError
from pykokoro.ssmd_config import SSMDRenderConfig, resolve_document_voice
from pykokoro.ssmd_parser import parse_ssmd_document


def test_api_binding_wins_and_emits_conflict_diagnostic():
    parsed = parse_ssmd_document(
        '---\nvoice_bindings:\n  kokoro:\n    moderator: af_sarah\n---\n<div voice="moderator">Hello.</div>',
        render_config=SSMDRenderConfig(voice_bindings={"kokoro": {"moderator": "af_bella"}}),
    )
    metadata = parsed.segments[0].metadata
    assert (metadata.voice_reference, metadata.voice_name, metadata.voice_source) == (
        "moderator",
        "af_bella",
        "api",
    )
    assert parsed.diagnostics[0].code == "ssmd.voice_binding_override"


@pytest.mark.parametrize(
    ("reference", "target"),
    [("moderator", "af_sarah"), ("Moderator", "Moderator")],
)
def test_voice_resolution_is_case_sensitive(reference, target):
    result = resolve_document_voice(
        reference,
        provider="kokoro",
        api_bindings={},
        header_bindings={"kokoro": {"moderator": "af_sarah"}},
    )
    assert result.target == target


def test_unrelated_provider_does_not_bind():
    parsed = parse_ssmd_document(
        '---\nvoice_bindings:\n  other:\n    moderator: af_sarah\n---\n<div voice="moderator">Hello.</div>'
    )
    assert parsed.segments[0].metadata.voice_name == "moderator"


def test_invalid_binding_shape_is_rejected():
    with pytest.raises(SSMDDocumentError):
        parse_ssmd_document("---\nvoice_bindings: bad\n---\nHello")
