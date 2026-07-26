import pytest

from pykokoro.pipeline_config import PipelineConfig
from pykokoro.stages.doc_parsers.ssmd import SsmdDocumentParser
from pykokoro.types import Trace


def test_ssmd_explicit_breaks_are_deterministic():
    parser = SsmdDocumentParser()
    cfg = PipelineConfig()
    text = "Hello ...500ms world"

    for _ in range(200):
        doc = parser.parse(text, cfg, Trace())
        assert any(boundary.duration_s == pytest.approx(0.5) for boundary in doc.boundary_events)
