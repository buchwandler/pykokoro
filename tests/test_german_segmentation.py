from __future__ import annotations

from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.stages.doc_parsers.plain import PlainTextDocumentParser
from pykokoro.stages.segmentation.phrasplit import PhrasplitSentenceSegmenter
from pykokoro.stages.text_preparation.spokenform import SpokenformTextPreparer
from pykokoro.types import Trace


def test_german_abbreviations_and_ordinals_do_not_create_false_boundaries():
    text = (
        "Prof. Klein sagt: Bitte stelle die Form auf die 2. Schiene, backe alles für "
        "45 Min. und lass es danach 1 Min. ruhen. Die Kosten liegen bei ca. "
        "12,80 EUR zzgl. Pfand."
    )
    cfg = PipelineConfig(generation=GenerationConfig(lang="de"))
    trace = Trace()
    doc = PlainTextDocumentParser().parse(text, cfg, trace)
    SpokenformTextPreparer().prepare(doc, cfg, trace)
    doc.segments = PhrasplitSentenceSegmenter().split(doc, cfg, trace)
    assert len(doc.segments) == 2
    assert doc.segments[0].text.endswith("eine Minute ruhen.")
    assert (
        doc.segments[1].text
        == "Die Kosten liegen bei zirka zwölf Euro achtzig Cent zuzüglich Pfand."
    )
    for segment in doc.segments:
        assert segment.text == doc.clean_text[segment.char_start : segment.char_end]
