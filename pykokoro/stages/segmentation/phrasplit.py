"""Phrasplit sentence segmentation after spoken-text preparation."""

from __future__ import annotations

import importlib
import re
from dataclasses import replace
from typing import Any

from ...pipeline_config import PipelineConfig
from ...types import Segment, Trace, TraceEvent
from ..doc_parsers.plain import PhrasplitSentenceSplitter
from ..protocols import DocumentResult, SentenceSegmenter

_HARD_METADATA_KEYS = frozenset(
    {
        "lang",
        "language",
        "voice",
        "voice_name",
        "ph",
        "phonemes",
        "audio_src",
        "audio_alt_text",
        "audio",
        "event",
        "paragraph",
    }
)


class PhrasplitSentenceSegmenter(PhrasplitSentenceSplitter, SentenceSegmenter):
    """Split prepared text per language run and refine structural boundaries."""

    def split(self, doc: DocumentResult, cfg: PipelineConfig, trace: Trace) -> list[Segment]:
        state = doc.linguistic_state
        if state is None or not getattr(state, "prepared_analysis", None):
            segments = super().split(doc, cfg, trace)
        else:
            segments = self._split_prepared_runs(doc, cfg, trace, state.prepared_analysis)
        segments = self._repair_closing_quote_boundaries(doc, segments)
        refined: list[Segment] = []
        for segment in segments:
            boundaries = self._structural_cuts(segment, doc)
            if not boundaries:
                refined.append(segment)
                continue
            cuts = [segment.char_start, *boundaries, segment.char_end]
            for clause_idx, (start, end) in enumerate(zip(cuts, cuts[1:], strict=False)):
                if end <= start:
                    continue
                refined.append(
                    replace(
                        segment,
                        id=f"{segment.id}_c{clause_idx}",
                        text=doc.clean_text[start:end],
                        char_start=start,
                        char_end=end,
                        clause_idx=clause_idx,
                    )
                )
        self._record_diagnostics(doc, trace, refined, cfg)
        return refined

    def _split_prepared_runs(
        self, doc: DocumentResult, cfg: PipelineConfig, trace: Trace, analyses: list[Any]
    ) -> list[Segment]:
        try:
            phrasplit = importlib.import_module("phrasplit")
        except ImportError:
            phrasplit = None

        segments: list[Segment] = []
        segment_index = 0
        sentence_index = 0
        for analysis in analyses:
            run = analysis.run
            text = analysis.text
            if not text:
                continue
            items: list[Any] = []
            if phrasplit is not None:
                diagnostics: list[Any] = []
                items = self._split_with_offsets(
                    phrasplit,
                    text,
                    None,
                    use_spacy=analysis.doc is not None,
                    language=run.language,
                    diagnostics_sink=diagnostics,
                    doc=analysis.doc,
                )
                if diagnostics:
                    trace.events.append(
                        TraceEvent(
                            stage="segmentation_run",
                            name="precomputed",
                            ms=0.0,
                            details={"language": run.language, "diagnostics": len(diagnostics)},
                        )
                    )
                if items and self._has_non_whitespace_gap(text, items):
                    items = self._split_with_offsets(
                        phrasplit,
                        text,
                        None,
                        use_spacy=False,
                        language=run.language,
                    )
            if not items and phrasplit is not None:
                items = self._split_with_offsets(
                    phrasplit,
                    text,
                    None,
                    use_spacy=False,
                    language=run.language,
                )
            if not items:
                items = [(text, 0, len(text), 0, None, None)]

            cursor = 0
            for item in items:
                item_text, start, end, paragraph, sentence, clause = item
                if not isinstance(item_text, str):
                    continue
                if not isinstance(start, int) or not isinstance(end, int):
                    start, end = cursor, cursor + len(item_text)
                if start < cursor or end < start or end > len(text) or text[start:end] != item_text:
                    start, end = cursor, min(len(text), cursor + len(item_text))
                if end <= start:
                    continue
                abs_start, abs_end = run.char_start + start, run.char_start + end
                resolved_sentence = (
                    sentence_index if sentence is None else sentence_index + sentence
                )
                segments.append(
                    Segment(
                        id=f"p0_s{resolved_sentence}_c{clause or 0}_seg{segment_index}",
                        text=doc.clean_text[abs_start:abs_end],
                        char_start=abs_start,
                        char_end=abs_end,
                        paragraph_idx=paragraph,
                        sentence_idx=resolved_sentence,
                        clause_idx=clause,
                    )
                )
                segment_index += 1
                cursor = end
                if sentence is None:
                    sentence_index += 1
            if cursor < len(text):
                abs_start, abs_end = run.char_start + cursor, run.char_end
                segments.append(
                    Segment(
                        id=f"p0_s{sentence_index}_c0_seg{segment_index}",
                        text=doc.clean_text[abs_start:abs_end],
                        char_start=abs_start,
                        char_end=abs_end,
                        paragraph_idx=0,
                        sentence_idx=sentence_index,
                        clause_idx=0,
                    )
                )
                segment_index += 1
                sentence_index += 1
        return segments

    @staticmethod
    def _has_non_whitespace_gap(text: str, items: list[Any]) -> bool:
        cursor = 0
        for item in items:
            start, end = item[1], item[2]
            if not isinstance(start, int) or not isinstance(end, int):
                return True
            if text[cursor:start].strip():
                return True
            cursor = max(cursor, end)
        return bool(text[cursor:].strip())

    @staticmethod
    def _repair_closing_quote_boundaries(
        doc: DocumentResult, segments: list[Segment]
    ) -> list[Segment]:
        """Split a terminator followed by a closing quote and new sentence."""
        repaired: list[Segment] = []
        pattern = re.compile(r"[.!?][\"'»”’]+\s+(?=[A-ZÄÖÜÀ-Þ])")
        next_sentence: dict[int, int] = {}
        for segment in segments:
            paragraph_idx = segment.paragraph_idx if segment.paragraph_idx is not None else 0
            sentence_idx = next_sentence.get(paragraph_idx, segment.sentence_idx or 0)
            start = segment.char_start
            for match in pattern.finditer(segment.text):
                end = segment.char_start + match.start() + len(match.group(0).rstrip())
                if end <= start or end >= segment.char_end:
                    continue
                repaired.append(
                    replace(
                        segment,
                        text=doc.clean_text[start:end],
                        char_start=start,
                        char_end=end,
                        sentence_idx=sentence_idx,
                    )
                )
                sentence_idx += 1
                start = segment.char_start + match.end()
            if start < segment.char_end:
                repaired.append(
                    replace(
                        segment,
                        text=doc.clean_text[start : segment.char_end],
                        char_start=start,
                        char_end=segment.char_end,
                        sentence_idx=sentence_idx,
                    )
                )
            next_sentence[paragraph_idx] = sentence_idx + 1
        return repaired

    @staticmethod
    def _structural_cuts(segment: Segment, doc: DocumentResult) -> list[int]:
        cuts: set[int] = set()
        for span in doc.annotation_spans:
            if _HARD_METADATA_KEYS.intersection(span.attrs) and span.char_start < span.char_end:
                if segment.char_start < span.char_start < segment.char_end:
                    cuts.add(span.char_start)
                if segment.char_start < span.char_end < segment.char_end:
                    cuts.add(span.char_end)
        for boundary in doc.boundary_events:
            if segment.char_start < boundary.pos < segment.char_end:
                position = boundary.pos
                if (
                    boundary.attrs.get("anchor", "after") == "after"
                    and position < len(doc.clean_text)
                    and doc.clean_text[position] in ".!?。！？"
                ):
                    position += 1
                if segment.char_start < position < segment.char_end:
                    cuts.add(position)
        return sorted(cuts)

    @staticmethod
    def _record_diagnostics(
        doc: DocumentResult, trace: Trace, segments: list[Segment], cfg: PipelineConfig
    ) -> None:
        metadata = doc.metadata.setdefault("segmentation", {})
        if not isinstance(metadata, dict):
            metadata = {}
            doc.metadata["segmentation"] = metadata
        metadata["backend"] = "phrasplit"
        metadata["segment_count"] = len(segments)
        trace.events.append(
            TraceEvent(
                stage="segmentation_run",
                name="split",
                ms=0.0,
                details={"backend": "phrasplit", "segment_count": len(segments)},
            )
        )


__all__ = ["PhrasplitSentenceSegmenter"]
