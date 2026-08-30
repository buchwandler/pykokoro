"""Phrasplit sentence segmentation after spoken-text preparation."""

from __future__ import annotations

import re
from dataclasses import replace

from ...pipeline_config import PipelineConfig
from ...types import Segment, Trace
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
    """Split prepared text and refine each sentence at hard metadata boundaries."""

    def split(self, doc: DocumentResult, cfg: PipelineConfig, trace: Trace) -> list[Segment]:
        segments = self._repair_closing_quote_boundaries(doc, super().split(doc, cfg, trace))
        refined: list[Segment] = []
        for segment in segments:
            cuts = self._metadata_cuts(segment, doc)
            if not cuts:
                refined.append(segment)
                continue
            boundaries = [segment.char_start, *cuts, segment.char_end]
            for clause_idx, (start, end) in enumerate(
                zip(boundaries, boundaries[1:], strict=False)
            ):
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
            local_matches = list(pattern.finditer(segment.text))
            for match in local_matches:
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
                        sentence_idx=sentence_idx,
                    )
                )
            next_sentence[paragraph_idx] = sentence_idx + 1
        return repaired

    @staticmethod
    def _metadata_cuts(segment: Segment, doc: DocumentResult) -> list[int]:
        cuts: set[int] = set()
        for span in doc.annotation_spans:
            if not (
                _HARD_METADATA_KEYS.intersection(span.attrs) and span.char_start < span.char_end
            ):
                continue
            if segment.char_start < span.char_start < segment.char_end:
                cuts.add(span.char_start)
            if segment.char_start < span.char_end < segment.char_end:
                cuts.add(span.char_end)
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
        tokenizer_config = cfg.tokenizer_config
        metadata["use_spacy"] = bool(
            tokenizer_config.use_spacy if tokenizer_config is not None else True
        )
        metadata["segment_count"] = len(segments)
        from ...types import TraceEvent

        trace.events.append(
            TraceEvent(
                stage="segmentation",
                name="split",
                ms=0.0,
                details={
                    "backend": "phrasplit",
                    "use_spacy": metadata["use_spacy"],
                    "segment_count": len(segments),
                },
            )
        )


__all__ = ["PhrasplitSentenceSegmenter"]
