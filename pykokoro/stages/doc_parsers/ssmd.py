from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

from ...pipeline_config import PipelineConfig
from ...spacy_models import make_spacy_model_request, spacy_selection_metadata
from ...ssmd_config import PauseCandidate, ResolvedPauseDefaults
from ...ssmd_parser import (
    DEFAULT_PAUSE_NONE,
    DEFAULT_PAUSE_WEAK,
    SSMDMetadata,
    SSMDSegment,
    parse_ssmd_document,
)
from ...tokenizer import TokenizerConfig
from ...types import AnnotationSpan, BoundaryEvent, Segment, Trace
from ..protocols import DocumentResult


class SsmdDocumentParser:
    def parse(self, text: str, cfg: PipelineConfig, trace: Trace) -> DocumentResult:
        generation = cfg.generation
        tokenizer_config = cfg.tokenizer_config or TokenizerConfig()
        parsed = parse_ssmd_document(
            text,
            render_config=cfg.ssmd,
            lang=generation.lang,
            pause_none=DEFAULT_PAUSE_NONE,
            pause_weak=DEFAULT_PAUSE_WEAK,
            pause_clause=generation.pause_clause,
            pause_sentence=generation.pause_sentence,
            pause_paragraph=generation.pause_paragraph,
            spacy_model=tokenizer_config.spacy_model,
            model_size=tokenizer_config.spacy_model_size,
            use_spacy=tokenizer_config.use_spacy,
        )
        for diagnostic in parsed.diagnostics:
            self._warn_once(trace, f"{diagnostic.code}: {diagnostic.message}")
        request = make_spacy_model_request(
            model=tokenizer_config.spacy_model,
            size=tokenizer_config.spacy_model_size,
        )
        sentence_diagnostics = parsed.sentence_diagnostics
        selected_model = getattr(sentence_diagnostics, "selected_model", None)
        selected_size = getattr(sentence_diagnostics, "selected_model_size", None)
        doc_metadata = {
            "spacy_models": {
                "sentence": spacy_selection_metadata(
                    language=generation.lang,
                    request=request,
                    selected_model=selected_model,
                    selected_size=selected_size,
                )
            }
        }
        initial_pause = parsed.initial_pause
        segments = list(parsed.segments)
        clean_text, spans, raw_boundaries, doc_segments = self._build_document(
            segments, initial_pause, trace, parsed.pause_defaults, cfg
        )
        candidates = [item for item in raw_boundaries if isinstance(item, PauseCandidate)]
        markers = [item for item in raw_boundaries if isinstance(item, BoundaryEvent)]
        boundaries = self._reduce_pause_candidates(candidates) + markers
        return DocumentResult(
            clean_text=clean_text,
            annotation_spans=spans,
            boundary_events=boundaries,
            segments=doc_segments,
            header=parsed.header,
            body=parsed.body,
            diagnostics=list(parsed.diagnostics),
            metadata=doc_metadata,
        )

    def _build_document(
        self,
        segments: Sequence[SSMDSegment],
        initial_pause: float,
        trace: Trace,
        pause_defaults: ResolvedPauseDefaults | None = None,
        cfg: PipelineConfig | None = None,
    ) -> tuple[str, list[AnnotationSpan], list[Any], list[Segment]]:
        cfg = cfg or PipelineConfig()
        clean_parts: list[str] = []
        spans: list[AnnotationSpan] = []
        boundaries: list[Any] = []
        doc_segments: list[Segment] = []
        cursor = 0
        current_paragraph = None
        previous_start = None
        previous_end = None
        seg_idx = 0

        if initial_pause > 0:
            boundaries.append(PauseCandidate(0, initial_pause, "explicit", "break", 400))

        for segment in segments:
            if (
                current_paragraph is not None
                and segment.paragraph != current_paragraph
                and previous_start is not None
                and previous_end is not None
            ):
                boundary_pos = self._paragraph_boundary_pos(previous_start, previous_end)
                if boundary_pos is not None:
                    header_duration = pause_defaults.paragraph if pause_defaults else None
                    duration = (
                        header_duration
                        if header_duration is not None
                        else cfg.generation.pause_paragraph
                    )
                    if duration is not None:
                        boundaries.append(
                            self._default_candidate(
                                boundary_pos,
                                duration,
                                "paragraph",
                                "header_default"
                                if header_duration is not None
                                else "pipeline_default",
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
                spans.append(AnnotationSpan(char_start=start, char_end=end, attrs=attrs))
            for marker in segment.metadata.markers_before:
                boundaries.append(
                    BoundaryEvent(
                        pos=start,
                        kind="marker",
                        attrs={"marker": marker},
                    )
                )
            for marker in segment.metadata.markers_after:
                boundaries.append(
                    BoundaryEvent(
                        pos=max(start, end - 1),
                        kind="marker",
                        attrs={"marker": marker},
                    )
                )
            if segment.pause_before > 0:
                boundaries.append(
                    PauseCandidate(start, segment.pause_before, "explicit", "break", 400)
                )
            if segment.pause_after > 0:
                boundary_pos = self._pause_boundary_pos(start, end)
                if boundary_pos is None:
                    boundary_pos = end
                boundaries.append(
                    PauseCandidate(boundary_pos, segment.pause_after, "explicit", "break", 400)
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
                        meta=attrs,
                        paragraph_idx=segment.paragraph,
                        sentence_idx=segment.sentence,
                        clause_idx=0,
                    )
                )
                seg_idx += 1

            if (
                previous_start is not None
                and previous_end is not None
                and doc_segments
                and len(doc_segments) >= 2
            ):
                previous = doc_segments[-2]
                current = doc_segments[-1]
                previous_voice = self._effective_voice(previous, cfg)
                current_voice = self._effective_voice(current, cfg)
                if previous_voice != current_voice:
                    boundary_pos = self._pause_boundary_pos(previous.char_start, previous.char_end)
                    if boundary_pos is not None:
                        voice_change = pause_defaults.voice_change if pause_defaults else None
                        if voice_change is not None:
                            boundaries.append(
                                self._default_candidate(
                                    boundary_pos,
                                    voice_change,
                                    "voice_change",
                                    "header_default",
                                )
                            )

        clean_text = "".join(clean_parts)
        boundaries.extend(
            self._sentence_candidates(
                doc_segments,
                pause_defaults,
                cfg.generation.pause_mode,
                cfg.generation.pause_sentence,
            )
        )
        return clean_text, spans, boundaries, doc_segments

    @staticmethod
    def _default_candidate(
        position: int,
        duration_s: float,
        kind: str,
        source: str,
    ) -> PauseCandidate:
        return PauseCandidate(
            position=position,
            duration_s=duration_s,
            source=cast(
                Literal["explicit", "api_default", "header_default", "pipeline_default"],
                source,
            ),
            kind=cast(Literal["break", "sentence", "paragraph", "voice_change"], kind),
            priority=200 if source == "header_default" else 100,
        )

    @staticmethod
    def _effective_voice(segment: Segment, cfg: PipelineConfig) -> object:
        voice = segment.meta.get("voice_name")
        if voice is not None:
            return voice
        return cfg.voice

    def _sentence_candidates(
        self,
        segments: list[Segment],
        pause_defaults: ResolvedPauseDefaults | None,
        pause_mode: str,
        pause_sentence: float,
    ) -> list[PauseCandidate]:
        if not segments:
            return []
        out: list[PauseCandidate] = []
        last_sentence = None
        last_paragraph = None
        last_end = None
        for segment in segments:
            sentence = segment.sentence_idx
            paragraph = segment.paragraph_idx
            if sentence is None:
                continue
            if (
                last_sentence is not None
                and (sentence != last_sentence or paragraph != last_paragraph)
                and last_end is not None
                and last_paragraph == paragraph
                and last_end > 0
            ):
                header_duration = pause_defaults.sentence if pause_defaults else None
                if header_duration is not None or pause_mode == "auto":
                    duration = header_duration if header_duration is not None else pause_sentence
                    out.append(
                        self._default_candidate(
                            max(0, last_end - 1),
                            duration,
                            "sentence",
                            "header_default" if header_duration is not None else "pipeline_default",
                        )
                    )
            last_sentence = sentence
            last_paragraph = paragraph
            last_end = max(last_end or 0, segment.char_end)
        return out

    @staticmethod
    def _reduce_pause_candidates(candidates: list[PauseCandidate]) -> list[BoundaryEvent]:
        grouped: dict[int, list[PauseCandidate]] = {}
        for candidate in candidates:
            if candidate.duration_s <= 0:
                continue
            grouped.setdefault(candidate.position, []).append(candidate)
        reduced: list[BoundaryEvent] = []
        for position in sorted(grouped):
            values = grouped[position]
            explicit = [candidate for candidate in values if candidate.source == "explicit"]
            selected = explicit or values
            duration = (
                sum(item.duration_s for item in selected)
                if explicit
                else max(item.duration_s for item in selected)
            )
            winner = max(selected, key=lambda item: (item.priority, item.duration_s))
            reduced.append(
                BoundaryEvent(
                    pos=position,
                    kind="pause",
                    duration_s=duration,
                    attrs={
                        "source": winner.source,
                        "kind": winner.kind,
                        "deterministic_pause_boundary": "true",
                        **(
                            {"strength": {"sentence": "s", "paragraph": "p"}[winner.kind]}
                            if winner.kind in {"sentence", "paragraph"}
                            else {}
                        ),
                    },
                )
            )
        return reduced

    @staticmethod
    def _sentence_boundaries(
        segments: list[Segment], boundaries: list[BoundaryEvent]
    ) -> list[BoundaryEvent]:
        if not segments:
            return []
        pause_positions = {boundary.pos for boundary in boundaries if boundary.kind == "pause"}
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
                if last_end is not None and last_paragraph == paragraph and last_end > 0:
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

    def _append_text(self, clean_parts: list[str], text: str, cursor: int) -> tuple[int, int, int]:
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
        if metadata.voice_reference:
            attrs["voice_reference"] = metadata.voice_reference
        if metadata.voice_source:
            attrs["voice_source"] = metadata.voice_source
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
        if metadata.emphasis:
            attrs["emphasis"] = metadata.emphasis
        if metadata.audio_src:
            attrs["audio_src"] = metadata.audio_src
        if metadata.audio_alt_text:
            attrs["audio_alt_text"] = metadata.audio_alt_text
        for key, value in (
            ("audio_clip_begin", metadata.audio_clip_begin),
            ("audio_clip_end", metadata.audio_clip_end),
            ("audio_speed", metadata.audio_speed),
            ("audio_repeat_dur", metadata.audio_repeat_dur),
            ("audio_sound_level", metadata.audio_sound_level),
        ):
            if value is not None:
                attrs[key] = str(value)
        if metadata.audio_repeat_count is not None:
            attrs["audio_repeat_count"] = str(metadata.audio_repeat_count)
        return attrs
