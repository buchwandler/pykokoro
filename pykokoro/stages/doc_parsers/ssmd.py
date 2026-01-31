from __future__ import annotations

from collections.abc import Sequence

from ...pipeline_config import PipelineConfig
from ...ssmd_parser import (
    DEFAULT_PAUSE_NONE,
    DEFAULT_PAUSE_WEAK,
    SSMDMetadata,
    SSMDSegment,
    parse_ssmd_to_segments,
)
from ...types import AnnotationSpan, BoundaryEvent, Segment, Trace
from ..protocols import DocumentResult


class SsmdDocumentParser:
    def parse(self, text: str, cfg: PipelineConfig, trace: Trace) -> DocumentResult:
        generation = cfg.generation
        initial_pause, segments = parse_ssmd_to_segments(
            text,
            lang=generation.lang,
            pause_none=DEFAULT_PAUSE_NONE,
            pause_weak=DEFAULT_PAUSE_WEAK,
            pause_clause=generation.pause_clause,
            pause_sentence=generation.pause_sentence,
            pause_paragraph=generation.pause_paragraph,
        )
        clean_text, spans, boundaries, doc_segments = self._build_document(
            segments, initial_pause, trace
        )
        if generation.pause_mode == "auto":
            boundaries.extend(self._sentence_boundaries(doc_segments, boundaries))
        return DocumentResult(
            clean_text=clean_text,
            annotation_spans=spans,
            boundary_events=boundaries,
            segments=doc_segments,
        )

    def _build_document(
        self,
        segments: Sequence[SSMDSegment],
        initial_pause: float,
        trace: Trace,
    ) -> tuple[str, list[AnnotationSpan], list[BoundaryEvent], list[Segment]]:
        clean_parts: list[str] = []
        spans: list[AnnotationSpan] = []
        boundaries: list[BoundaryEvent] = []
        doc_segments: list[Segment] = []
        cursor = 0
        current_paragraph = None
        previous_start = None
        previous_end = None
        seg_idx = 0

        if initial_pause > 0:
            boundaries.append(
                BoundaryEvent(pos=0, kind="pause", duration_s=initial_pause, attrs={})
            )

        for segment in segments:
            if (
                current_paragraph is not None
                and segment.paragraph != current_paragraph
                and previous_start is not None
                and previous_end is not None
            ):
                boundary_pos = self._paragraph_boundary_pos(
                    previous_start, previous_end
                )
                if boundary_pos is not None:
                    boundaries.append(
                        BoundaryEvent(
                            pos=boundary_pos,
                            kind="pause",
                            duration_s=None,
                            attrs={"strength": "p"},
                        )
                    )
                clean_parts.append("\n\n")
                cursor += 2
            if segment.metadata.audio_src:
                if not segment.text.strip():
                    self._warn_once(
                        trace,
                        "SSMD audio segment has no alt_text; skipping audio segment.",
                    )
                    continue
                self._warn_once(
                    trace,
                    "SSMD audio segments are not mixed; speaking alt_text instead.",
                )
            start, end, cursor = self._append_text(clean_parts, segment.text, cursor)
            attrs = self._metadata_to_attrs(segment.metadata)
            if attrs and end > start:
                spans.append(
                    AnnotationSpan(char_start=start, char_end=end, attrs=attrs)
                )
            if segment.pause_before > 0:
                boundaries.append(
                    BoundaryEvent(
                        pos=start,
                        kind="pause",
                        duration_s=segment.pause_before,
                        attrs={},
                    )
                )
            if segment.pause_after > 0:
                boundary_pos = self._pause_boundary_pos(start, end)
                if boundary_pos is None:
                    boundary_pos = end
                boundaries.append(
                    BoundaryEvent(
                        pos=boundary_pos,
                        kind="pause",
                        duration_s=segment.pause_after,
                        attrs={},
                    )
                )
            current_paragraph = segment.paragraph
            previous_start = start
            previous_end = end
            if end > start:
                segment_id = f"p{segment.paragraph}_s{segment.sentence}_c0_seg{seg_idx}"
                doc_segments.append(
                    Segment(
                        id=segment_id,
                        text=segment.text,
                        char_start=start,
                        char_end=end,
                        paragraph_idx=segment.paragraph,
                        sentence_idx=segment.sentence,
                        clause_idx=0,
                    )
                )
                seg_idx += 1

        clean_text = "".join(clean_parts)
        return clean_text, spans, boundaries, doc_segments

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

    @staticmethod
    def _warn_once(trace: Trace, message: str) -> None:
        if message not in trace.warnings:
            trace.warnings.append(message)

    def _append_text(
        self, clean_parts: list[str], text: str, cursor: int
    ) -> tuple[int, int, int]:
        if not text:
            return cursor, cursor, cursor
        if clean_parts:
            previous = clean_parts[-1]
            if previous and not previous[-1].isspace() and not text[0].isspace():
                clean_parts.append(" ")
                cursor += 1
        start = cursor
        clean_parts.append(text)
        cursor += len(text)
        return start, cursor, cursor

    @staticmethod
    def _paragraph_boundary_pos(start: int, end: int) -> int | None:
        if end <= start:
            return None
        return max(start, end - 1)

    @staticmethod
    def _pause_boundary_pos(start: int, end: int) -> int | None:
        if end <= start:
            return None
        return max(start, end - 1)

    def _metadata_to_attrs(self, metadata: SSMDMetadata) -> dict[str, str]:
        attrs: dict[str, str] = {}
        if metadata.language:
            attrs["lang"] = metadata.language
        if metadata.phonemes:
            attrs["ph"] = metadata.phonemes
        if metadata.voice_name:
            attrs["voice_name"] = metadata.voice_name
        if metadata.voice_language:
            attrs["voice_language"] = metadata.voice_language
        if metadata.voice_gender:
            attrs["voice_gender"] = metadata.voice_gender
        if metadata.voice_variant:
            attrs["voice_variant"] = metadata.voice_variant
        if metadata.prosody_rate:
            attrs["prosody_rate"] = metadata.prosody_rate
        if metadata.prosody_pitch:
            attrs["prosody_pitch"] = metadata.prosody_pitch
        if metadata.prosody_volume:
            attrs["prosody_volume"] = metadata.prosody_volume
        if metadata.emphasis:
            attrs["emphasis"] = metadata.emphasis
        if metadata.say_as_interpret:
            attrs["say_as_interpret"] = metadata.say_as_interpret
        if metadata.say_as_format:
            attrs["say_as_format"] = metadata.say_as_format
        if metadata.say_as_detail:
            attrs["say_as_detail"] = metadata.say_as_detail
        if metadata.substitution:
            attrs["substitution"] = metadata.substitution
        if metadata.markers:
            attrs["markers"] = ",".join(metadata.markers)
        if metadata.audio_src:
            attrs["audio_src"] = metadata.audio_src
        if metadata.audio_alt_text:
            attrs["audio_alt_text"] = metadata.audio_alt_text
        return attrs
