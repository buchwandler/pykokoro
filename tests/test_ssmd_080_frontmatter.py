import pytest

from pykokoro.exceptions import SSMDDocumentError
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.ssmd_config import SSMDRenderConfig
from pykokoro.ssmd_parser import parse_ssmd_document
from pykokoro.stages.doc_parsers.ssmd import SsmdDocumentParser
from pykokoro.types import Trace


def test_frontmatter_title_is_metadata_and_not_spoken():
    parsed = parse_ssmd_document("---\ntitle: Review\n---\nHello.")
    assert parsed.header["title"] == "Review"
    assert parsed.body == "Hello."
    assert "Review" not in " ".join(segment.text for segment in parsed.segments)


@pytest.mark.parametrize("closing", ["---", "..."])
def test_frontmatter_closers_are_supported(closing):
    assert parse_ssmd_document(f"---\ntitle: X\n{closing}\nHello").body == "Hello"


def test_four_dash_line_is_body_text():
    parsed = parse_ssmd_document("----\nHello")
    assert parsed.body == "----\nHello"


def test_malformed_and_non_mapping_headers_are_coded():
    with pytest.raises(SSMDDocumentError, match="closing") as missing:
        parse_ssmd_document("---\ntitle: X\nHello")
    assert missing.value.code == "header.yaml_invalid"
    with pytest.raises(SSMDDocumentError) as non_mapping:
        parse_ssmd_document("---\n- item\n---\nHello")
    assert non_mapping.value.code == "header.root_not_mapping"


def test_literal_header_escape_hatch_preserves_source():
    parser = SsmdDocumentParser()
    cfg = PipelineConfig(ssmd=SSMDRenderConfig(parse_header=False))
    doc = parser.parse("---\ntitle: Literal\n---\nHello", cfg, Trace())
    assert "title" in doc.clean_text


def test_unknown_header_policy_warns_or_errors():
    parsed = parse_ssmd_document("---\nfuture_key: yes\n---\nHello")
    assert parsed.diagnostics[0].code == "header.unknown_key"
    with pytest.raises(SSMDDocumentError) as exc:
        parse_ssmd_document(
            "---\nfuture_key: yes\n---\nHello",
            render_config=SSMDRenderConfig(unknown_header="error"),
        )
    assert exc.value.code == "header.unknown_key"


@pytest.mark.parametrize("value", ["yes", "", 3])
def test_invalid_title_is_rejected(value):
    with pytest.raises(SSMDDocumentError) as exc:
        parse_ssmd_document(f"---\ntitle: {value}\n---\nHello")
    assert exc.value.code == "header.title_invalid"
