"""Phrasplit sentence segmentation after spoken-text preparation."""

from __future__ import annotations

import importlib
import re
from dataclasses import replace
from typing import Any

from ...pipeline_config import PipelineConfig
from ...runtime.language_plan import canonicalize_language
from ...types import BoundaryEvent, Segment, Trace, TraceEvent
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
            self._add_auto_clause_boundaries(doc, cfg, trace, state.prepared_analysis)
            self._add_auto_parenthetical_boundaries(doc, cfg, trace, state.prepared_analysis)
        segments = self._repair_closing_quote_boundaries(doc, segments)
        segments = [self._with_semantic_language(doc, segment, cfg) for segment in segments]
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

    @staticmethod
    def _add_auto_clause_boundaries(
        doc: DocumentResult, cfg: PipelineConfig, trace: Trace, analyses: list[Any]
    ) -> None:
        """Add high-confidence Phrasplit comma pauses from prepared analyses."""
        counters = {
            "detected": 0,
            "added": 0,
            "deduplicated": 0,
            "runs_with_analysis": 0,
            "runs_without_analysis": 0,
        }
        if cfg.generation.pause_mode != "auto" or cfg.generation.pause_clause <= 0.0:
            return

        try:
            phrasplit = importlib.import_module("phrasplit")
        except ImportError:
            phrasplit = None
        detector = (
            getattr(phrasplit, "detect_clause_boundaries", None) if phrasplit is not None else None
        )
        if phrasplit is not None and not callable(detector):
            trace.warnings.append(
                "phrasplit.detect_clause_boundaries is unavailable; skipping clausal comma detection."
            )
            return

        existing_pause_positions = {
            event.pos for event in doc.boundary_events if event.kind == "pause"
        }
        for analysis in analyses:
            text = analysis.text
            if not text:
                continue
            if analysis.doc is None:
                counters["runs_without_analysis"] += 1
                continue
            counters["runs_with_analysis"] += 1
            if not callable(detector):
                continue
            try:
                boundaries = detector(
                    text,
                    language=analysis.run.language,
                    doc=analysis.doc,
                )
            except (TypeError, ValueError) as exc:
                trace.warnings.append(
                    f"Skipped clausal comma detection for {analysis.run.language!r}: {exc}."
                )
                continue
            try:
                boundary_iter = iter(boundaries or ())
            except TypeError:
                trace.warnings.append(
                    "Skipped non-iterable Phrasplit clausal comma boundary results."
                )
                continue
            for boundary in boundary_iter:
                counters["detected"] += 1
                kind = getattr(boundary, "kind", None)
                char_start = getattr(boundary, "char_start", None)
                char_end = getattr(boundary, "char_end", None)
                boundary_text = getattr(boundary, "text", None)
                if (
                    kind != "clausal_comma"
                    or not isinstance(char_start, int)
                    or isinstance(char_start, bool)
                    or not isinstance(char_end, int)
                    or isinstance(char_end, bool)
                    or not isinstance(boundary_text, str)
                ):
                    trace.warnings.append(
                        "Skipped malformed Phrasplit clausal comma boundary result."
                    )
                    continue
                if char_start < 0 or char_end <= char_start or char_end > len(text):
                    trace.warnings.append(
                        f"Skipped malformed Phrasplit boundary offsets {char_start}:{char_end}."
                    )
                    continue
                absolute_start = analysis.run.char_start + char_start
                absolute_end = analysis.run.char_start + char_end
                if (
                    absolute_start < 0
                    or absolute_start < analysis.run.char_start
                    or absolute_end > analysis.run.char_end
                    or absolute_end > len(doc.clean_text)
                    or doc.clean_text[absolute_start:absolute_end] != boundary_text
                ):
                    trace.warnings.append(
                        "Skipped Phrasplit boundary with a mismatched clean_text slice "
                        f"at {absolute_start}:{absolute_end}."
                    )
                    continue
                if absolute_start in existing_pause_positions:
                    counters["deduplicated"] += 1
                    continue
                doc.boundary_events.append(
                    BoundaryEvent(
                        pos=absolute_start,
                        kind="pause",
                        duration_s=float(cfg.generation.pause_clause),
                        attrs={
                            "strength": "c",
                            "anchor": "after",
                            "source": "pipeline_default",
                            "detector": "phrasplit",
                            "boundary_kind": "clausal_comma",
                        },
                    )
                )
                existing_pause_positions.add(absolute_start)
                counters["added"] += 1

        metadata = doc.metadata.setdefault("segmentation", {})
        if not isinstance(metadata, dict):
            metadata = {}
            doc.metadata["segmentation"] = metadata
        metadata["clausal_comma_boundaries"] = counters["added"]
        trace.events.append(
            TraceEvent(
                stage="segmentation_run",
                name="clausal_comma_boundaries",
                ms=0.0,
                details=counters,
            )
        )

    @staticmethod
    def _add_auto_parenthetical_boundaries(
        doc: DocumentResult, cfg: PipelineConfig, trace: Trace, analyses: list[Any]
    ) -> None:
        """Add Phrasplit parenthetical pauses from prepared text analyses."""
        counters = {
            "detected": 0,
            "open_detected": 0,
            "close_detected": 0,
            "added": 0,
            "deduplicated": 0,
            "malformed_skipped": 0,
        }
        generation = cfg.generation
        if generation.pause_mode != "auto" or generation.pause_parenthetical <= 0.0:
            return

        try:
            phrasplit = importlib.import_module("phrasplit")
        except ImportError:
            phrasplit = None
        detector = (
            getattr(phrasplit, "detect_parenthetical_boundaries", None)
            if phrasplit is not None
            else None
        )
        if not callable(detector):
            trace.warnings.append(
                "phrasplit.detect_parenthetical_boundaries is unavailable; "
                "skipping parenthetical detection."
            )
            return

        existing_pause_positions = {
            event.pos for event in doc.boundary_events if event.kind == "pause"
        }
        for analysis in analyses:
            text = analysis.text
            if not text:
                continue
            try:
                boundaries = detector(text, language=analysis.run.language)
                boundary_iter = iter(boundaries or ())
            except (TypeError, ValueError) as exc:
                trace.warnings.append(
                    f"Skipped parenthetical detection for {analysis.run.language!r}: {exc}."
                )
                continue
            for boundary in boundary_iter:
                counters["detected"] += 1
                kind = getattr(boundary, "kind", None)
                char_start = getattr(boundary, "char_start", None)
                char_end = getattr(boundary, "char_end", None)
                boundary_text = getattr(boundary, "text", None)
                if kind == "parenthetical_open":
                    counters["open_detected"] += 1
                elif kind == "parenthetical_close":
                    counters["close_detected"] += 1
                if (
                    kind not in {"parenthetical_open", "parenthetical_close"}
                    or not isinstance(char_start, int)
                    or isinstance(char_start, bool)
                    or not isinstance(char_end, int)
                    or isinstance(char_end, bool)
                    or not isinstance(boundary_text, str)
                    or char_start < 0
                    or char_start >= char_end
                    or char_end > len(text)
                    or text[char_start:char_end] != boundary_text
                    or boundary_text != ("(" if kind == "parenthetical_open" else ")")
                ):
                    counters["malformed_skipped"] += 1
                    trace.warnings.append(
                        "Skipped malformed Phrasplit parenthetical boundary result."
                    )
                    continue
                event_pos = analysis.run.char_start + (
                    char_start if kind == "parenthetical_open" else char_end
                )
                if (
                    event_pos < analysis.run.char_start
                    or event_pos > analysis.run.char_end
                    or event_pos < 0
                    or event_pos > len(doc.clean_text)
                ):
                    counters["malformed_skipped"] += 1
                    trace.warnings.append(
                        f"Skipped Phrasplit parenthetical boundary at invalid position {event_pos}."
                    )
                    continue
                if event_pos in existing_pause_positions:
                    counters["deduplicated"] += 1
                    continue
                attrs = {
                    "strength": "w",
                    "anchor": "before",
                    "source": "pipeline_default",
                    "detector": "phrasplit",
                    "pause_kind": "parenthetical",
                    "boundary_kind": kind,
                }
                meta = getattr(boundary, "meta", None)
                if isinstance(meta, dict):
                    for key in ("confidence", "reason"):
                        if key in meta:
                            attrs[key] = str(meta[key])
                doc.boundary_events.append(
                    BoundaryEvent(
                        pos=event_pos,
                        kind="pause",
                        duration_s=float(generation.pause_parenthetical),
                        attrs=attrs,
                    )
                )
                existing_pause_positions.add(event_pos)
                counters["added"] += 1

        metadata = doc.metadata.setdefault("segmentation", {})
        if not isinstance(metadata, dict):
            metadata = {}
            doc.metadata["segmentation"] = metadata
        metadata["parenthetical_boundaries"] = counters["added"]
        trace.events.append(
            TraceEvent(
                stage="segmentation_run",
                name="parenthetical_boundaries",
                ms=0.0,
                details=counters,
            )
        )

    @staticmethod
    def _with_semantic_language(
        doc: DocumentResult, segment: Segment, cfg: PipelineConfig
    ) -> Segment:
        state = doc.linguistic_state
        language = None
        plans = getattr(state, "prepared_plan", ()) if state is not None else ()
        covering = [
            run
            for run in plans
            if run.char_start <= segment.char_start and segment.char_end <= run.char_end
        ]
        if covering:
            language = min(covering, key=lambda run: run.char_end - run.char_start).language
        if language is None:
            language = canonicalize_language(cfg.generation.lang or "en-us")
        return replace(segment, meta={**segment.meta, "language": language})

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
    def _hard_metadata_keys_for_span(span: Any) -> set[str]:
        attrs = dict(span.attrs)
        if attrs.get("scope", "semantic") == "pronunciation":
            attrs.pop("lang", None)
            attrs.pop("language", None)
        return set(_HARD_METADATA_KEYS.intersection(attrs))

    @staticmethod
    def _structural_cuts(segment: Segment, doc: DocumentResult) -> list[int]:
        cuts: set[int] = set()
        for span in doc.annotation_spans:
            if (
                PhrasplitSentenceSegmenter._hard_metadata_keys_for_span(span)
                and span.char_start < span.char_end
            ):
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
                    and doc.clean_text[position] in ".!?。！？,;:"
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
