from __future__ import annotations

import importlib
import logging
import os
import re
from bisect import bisect_left
from typing import TYPE_CHECKING, Any

from ...types import AnnotationSpan, BoundaryEvent, Segment, Trace
from ..protocols import DocumentResult

if TYPE_CHECKING:
    from ...pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)

SplitItem = tuple[
    str | None,
    int | None,
    int | None,
    int | None,
    int | None,
    int | None,
]


class PlainTextDocumentParser:
    def parse(self, text: str, cfg: PipelineConfig, trace: Trace) -> DocumentResult:
        boundaries = self._paragraph_boundaries(text)
        doc = DocumentResult(clean_text=text, boundary_events=boundaries)
        splitter = PhrasplitSentenceSplitter()
        doc.segments = splitter.split(doc, cfg, trace)
        if cfg.generation.pause_mode == "auto":
            doc.boundary_events.extend(
                self._sentence_boundaries(doc.segments, doc.boundary_events)
            )
        return doc

    @staticmethod
    def _paragraph_boundaries(text: str) -> list[BoundaryEvent]:
        boundaries: list[BoundaryEvent] = []
        for match in re.finditer(r"\n\s*\n", text):
            if match.start() == 0:
                continue
            boundary_pos = match.start() - 1
            if boundary_pos < 0:
                continue
            boundaries.append(
                BoundaryEvent(
                    pos=boundary_pos,
                    kind="pause",
                    duration_s=None,
                    attrs={"strength": "p"},
                )
            )
        return boundaries

    @staticmethod
    def _sentence_boundaries(
        segments: list[Segment], boundaries: list[BoundaryEvent]
    ) -> list[BoundaryEvent]:
        if not segments:
            return []
        pause_positions = {
            boundary.pos for boundary in boundaries if boundary.kind == "pause"
        }
        out: list[BoundaryEvent] = []
        last_sentence = None
        last_paragraph = None
        last_end = None
        for segment in segments:
            sentence = segment.sentence_idx
            paragraph = segment.paragraph_idx
            if sentence is None:
                continue
            if last_sentence is None:
                last_sentence = sentence
                last_paragraph = paragraph
                last_end = segment.char_end
                continue
            if sentence != last_sentence or paragraph != last_paragraph:
                if (
                    last_end is not None
                    and last_paragraph == paragraph
                    and last_end > 0
                ):
                    boundary_pos = max(0, last_end - 1)
                    if boundary_pos not in pause_positions:
                        out.append(
                            BoundaryEvent(
                                pos=boundary_pos,
                                kind="pause",
                                duration_s=None,
                                attrs={"strength": "s"},
                            )
                        )
                        pause_positions.add(boundary_pos)
                last_sentence = sentence
                last_paragraph = paragraph
                last_end = segment.char_end
            else:
                if last_end is None or segment.char_end > last_end:
                    last_end = segment.char_end
        return out


class PhrasplitSentenceSplitter:
    def split(  # noqa: C901
        self, doc: DocumentResult, cfg: PipelineConfig, trace: Trace
    ) -> list[Segment]:
        text = doc.clean_text
        language_model = self._language_model_from_lang(cfg.generation.lang)
        try:
            phrasplit = importlib.import_module("phrasplit")
        except Exception:
            if not text:
                return []
            return [
                Segment(
                    id="p0_s0_c0_seg0",
                    text=text,
                    char_start=0,
                    char_end=len(text),
                    paragraph_idx=0,
                    sentence_idx=0,
                    clause_idx=0,
                )
            ]

        override_ranges = self._override_ranges(doc.annotation_spans)
        ranges = self._hard_ranges(text, doc.boundary_events, override_ranges)
        paragraph_breaks = sorted(
            boundary.pos
            for boundary in doc.boundary_events
            if boundary.kind == "pause"
            and boundary.duration_s is None
            and boundary.attrs.get("strength") == "p"
        )
        segments: list[Segment] = []
        seg_idx = 0
        sentence_idx = 0

        for start, end in ranges:
            if end <= start:
                continue
            chunk = text[start:end]
            split_items: list[SplitItem]
            if (start, end) in override_ranges:
                split_items = [(chunk, 0, len(chunk), None, None, None)]
            else:
                split_items = self._split_with_offsets(phrasplit, chunk, language_model)
                if not split_items:
                    split_items = [(chunk, 0, len(chunk), None, None, None)]

            cursor = 0
            chunk_len = len(chunk)

            for item in split_items:
                seg_text, seg_start, seg_end, para, sent, clause = item
                if seg_text is None:
                    continue

                offsets_valid = True
                reason = ""
                if seg_start is None or seg_end is None:
                    offsets_valid = False
                    reason = "missing offsets"
                elif seg_start < 0 or seg_end < seg_start or seg_end > chunk_len:
                    offsets_valid = False
                    reason = "invalid offsets"
                else:
                    slice_text = chunk[seg_start:seg_end]
                    if (
                        slice_text != seg_text
                        and slice_text.strip() != seg_text.strip()
                    ):
                        offsets_valid = False
                        reason = "offset slice mismatch"
                    elif seg_start < cursor:
                        offsets_valid = False
                        reason = "overlapping offsets"

                if not offsets_valid:
                    found = chunk.find(seg_text, cursor) if seg_text else -1
                    if found >= 0:
                        seg_start = found
                        seg_end = found + len(seg_text)
                    else:
                        seg_start = cursor
                        seg_end = cursor + len(seg_text)
                    trace.warnings.append(
                        "Adjusted splitter offsets for segment "
                        f"{seg_idx} ({seg_text!r}): {reason}."
                    )

                assert seg_start is not None and seg_end is not None
                seg_start = max(0, min(seg_start, chunk_len))
                seg_end = max(seg_start, min(seg_end, chunk_len))

                if seg_start < cursor:
                    adjusted_start = cursor
                    adjusted_end = min(chunk_len, adjusted_start + len(seg_text))
                    trace.warnings.append(
                        "Clamped splitter offsets to avoid overlap for segment "
                        f"{seg_idx} ({seg_text!r})."
                    )
                    seg_start = adjusted_start
                    seg_end = max(seg_start, adjusted_end)
                elif seg_start > cursor:
                    gap = chunk[cursor:seg_start]
                    if gap.strip():
                        offset = next(
                            idx for idx, ch in enumerate(gap) if not ch.isspace()
                        )
                        adjusted_start = cursor + offset
                        trace.warnings.append(
                            "Adjusted splitter offsets to avoid dropping "
                            f"non-whitespace characters for segment {seg_idx} "
                            f"({seg_text!r})."
                        )
                        seg_start = adjusted_start
                        seg_end = max(seg_start, seg_end)

                abs_start = start + seg_start
                abs_end = start + seg_end
                range_paragraph = bisect_left(paragraph_breaks, start)
                resolved_sentence = sent if sent is not None else sentence_idx
                if sent is None:
                    sentence_idx += 1
                else:
                    sentence_idx = max(sentence_idx, sent + 1)
                if para is None:
                    resolved_paragraph = range_paragraph
                elif paragraph_breaks and para == 0 and range_paragraph != 0:
                    resolved_paragraph = range_paragraph
                else:
                    resolved_paragraph = para
                resolved_clause = clause if clause is not None else 0
                segment_id = (
                    f"p{resolved_paragraph}"
                    f"_s{resolved_sentence}"
                    f"_c{resolved_clause}"
                    f"_seg{seg_idx}"
                )
                segments.append(
                    Segment(
                        id=segment_id,
                        text=text[abs_start:abs_end],
                        char_start=abs_start,
                        char_end=abs_end,
                        paragraph_idx=resolved_paragraph,
                        sentence_idx=resolved_sentence,
                        clause_idx=resolved_clause,
                    )
                )
                seg_idx += 1
                cursor = max(cursor, seg_end)

        if not segments and text:
            segments.append(
                Segment(
                    id="p0_s0_c0_seg0",
                    text=text,
                    char_start=0,
                    char_end=len(text),
                    paragraph_idx=0,
                    sentence_idx=0,
                    clause_idx=0,
                )
            )
        if os.getenv("PYKOKORO_DEBUG_SEGMENTS"):
            logger.debug("Splitter clean_text: %r", text)
            for segment in segments:
                logger.debug(
                    "Segment %s: %d:%d %r",
                    segment.id,
                    segment.char_start,
                    segment.char_end,
                    segment.text,
                )
            recon = "".join(
                text[segment.char_start : segment.char_end] for segment in segments
            )
            if recon != text:
                mismatch = _first_mismatch(recon, text)
                logger.debug(
                    "Segment reconstruction mismatch at %d (recon=%r text=%r)",
                    mismatch,
                    recon[mismatch : mismatch + 40],
                    text[mismatch : mismatch + 40],
                )
            else:
                logger.debug("Segment reconstruction matches clean_text")
        return segments

    def _hard_ranges(
        self,
        text: str,
        boundaries: list[BoundaryEvent],
        override_ranges: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        positions_set: set[int] = set()
        for boundary in boundaries:
            pos = boundary.pos
            if (
                boundary.kind == "pause"
                and boundary.duration_s is None
                and pos > 0
                and pos < len(text)
            ):
                pos = min(len(text), pos + 1)
            positions_set.add(pos)
        for start, end in override_ranges:
            positions_set.add(start)
            positions_set.add(end)
        sorted_positions = sorted(positions_set)
        ranges: list[tuple[int, int]] = []
        start = 0
        for pos in sorted_positions:
            pos = max(0, min(len(text), pos))
            if pos > start:
                ranges.append((start, pos))
            start = pos
        if start < len(text):
            ranges.append((start, len(text)))
        if not ranges:
            ranges.append((0, len(text)))
        return ranges

    def _override_ranges(self, spans: list[AnnotationSpan]) -> set[tuple[int, int]]:
        ranges: set[tuple[int, int]] = set()
        for span in spans:
            if "ph" in span.attrs or "phonemes" in span.attrs:
                ranges.add((span.char_start, span.char_end))
        return ranges

    def _split_with_offsets(
        self, phrasplit_module: Any, text: str, language_model: str
    ) -> list[SplitItem]:
        kwargs: dict[str, object] = {
            "mode": "sentence",
            "language_model": language_model,
        }
        for key in ("apply_corrections", "split_on_colon"):
            kwargs[key] = True

        if hasattr(phrasplit_module, "split_with_offsets"):
            try:
                segments = phrasplit_module.split_with_offsets(text, **kwargs)
            except (OSError, TypeError):
                try:
                    segments = phrasplit_module.split_with_offsets(
                        text,
                        mode="sentence",
                        language_model=language_model,
                    )
                except OSError:
                    return []
        elif hasattr(phrasplit_module, "iter_split_with_offsets"):
            try:
                segments = list(
                    phrasplit_module.iter_split_with_offsets(text, **kwargs)
                )
            except (OSError, TypeError):
                try:
                    segments = list(
                        phrasplit_module.iter_split_with_offsets(
                            text,
                            mode="sentence",
                            language_model=language_model,
                        )
                    )
                except OSError:
                    return []
        else:
            return []

        out: list[SplitItem] = []
        for seg in segments:
            seg_text = getattr(seg, "text", None)
            start = getattr(seg, "start", None)
            end = getattr(seg, "end", None)
            if start is None:
                start = getattr(seg, "char_start", None)
            if end is None:
                end = getattr(seg, "char_end", None)
            para = getattr(seg, "paragraph", None)
            sent = getattr(seg, "sentence", None)
            clause = getattr(seg, "clause", None)
            if para is None:
                para = getattr(seg, "paragraph_idx", None)
            if sent is None:
                sent = getattr(seg, "sentence_idx", None)
            if clause is None:
                clause = getattr(seg, "clause_idx", None)
            if seg_text is None and start is not None and end is not None:
                seg_text = text[start:end]
            if seg_text is None:
                continue
            out.append((seg_text, start, end, para, sent, clause))
        return out

    def _language_model_from_lang(self, lang: str | None) -> str:
        lang_code = (lang or "en").lower()
        for sep in ("-", "_"):
            if sep in lang_code:
                lang_code = lang_code.split(sep, 1)[0]
                break
        if not lang_code:
            lang_code = "en"
        web_langs = {"en", "zh"}
        size = "sm"
        if lang_code in web_langs:
            return f"{lang_code}_core_web_{size}"
        return f"{lang_code}_core_news_{size}"


def _first_mismatch(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    for idx in range(limit):
        if left[idx] != right[idx]:
            return idx
    return limit
