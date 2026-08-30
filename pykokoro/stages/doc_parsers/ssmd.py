"""Structure-only SSMD document parsing for the PyKokoro frontend."""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, Literal, cast

import ssmd

from ...pipeline_config import PipelineConfig
from ...ssmd_config import (
    PauseCandidate,
    SSMDDiagnostic,
    resolve_document_voice,
    resolve_pause_defaults,
)
from ...ssmd_parser import parse_ssmd_document as _legacy_parse_ssmd_document
from ...types import AnnotationSpan, BoundaryEvent, Segment, Trace
from ..protocols import DocumentResult

parse_ssmd_document = _legacy_parse_ssmd_document

_BREAK_TIME_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s)\s*$", re.IGNORECASE)

DEFAULT_PAUSE_NONE = 0.0
DEFAULT_PAUSE_WEAK = 0.15
DEFAULT_PAUSE_CLAUSE = 0.3
DEFAULT_PAUSE_SENTENCE = 0.6
DEFAULT_PAUSE_PARAGRAPH = 1.0


class SsmdDocumentParser:
    """Parse SSMD structure without assigning semantic sentence boundaries."""

    def parse(self, text: str, cfg: PipelineConfig, trace: Trace) -> DocumentResult:
        parsed = ssmd.parse_structure(
            text,
            default_lang=cfg.generation.lang,
            parse_yaml_header=cfg.ssmd.parse_header,
        )
        structural_text, structural_annotations, structural_events = self._restore_paragraphs(
            parsed
        )
        warnings = [str(warning) for warning in parsed.warnings]
        diagnostics = [self._diagnostic(item) for item in parsed.diagnostics]
        for warning in warnings:
            self._warn_once(trace, warning)
        for diagnostic in diagnostics:
            self._warn_once(trace, f"{diagnostic.code}: {diagnostic.message}")

        annotations = [
            AnnotationSpan(
                char_start=start,
                char_end=end,
                attrs=self._annotation_attrs(attrs, parsed.header, cfg),
            )
            for start, end, attrs in structural_annotations
            if start <= end
        ]
        boundaries = self._events_to_boundaries(structural_events, cfg, trace)
        resolved_pause_defaults = resolve_pause_defaults(
            parsed.header.get("pause_defaults"), cfg.ssmd.pause_defaults
        )
        metadata: dict[str, object] = {
            "ssmd_pause_defaults": (
                resolved_pause_defaults.to_dict() if resolved_pause_defaults is not None else None
            ),
            "ssmd_structure": True,
        }
        if parse_ssmd_document is not _legacy_parse_ssmd_document:
            compatibility: Any = parse_ssmd_document(text)
            sentence_diagnostic: Any = getattr(compatibility, "sentence_diagnostics", None)
            if sentence_diagnostic is not None:
                metadata["spacy_models"] = {
                    "sentence": {
                        "selected_model": getattr(sentence_diagnostic, "selected_model", None),
                        "selected_model_size": getattr(
                            sentence_diagnostic, "selected_model_size", None
                        ),
                    }
                }
        # A structure-only parser intentionally supplies no sentence indices.  Keeping one
        # structural span preserves the historical DocumentResult shape for callers while all
        # semantic segmentation is delegated to the next stage.
        segments = []
        if structural_text:
            segments.append(
                Segment(
                    id="p0_s0_c0_seg0",
                    text=structural_text,
                    char_start=0,
                    char_end=len(structural_text),
                    paragraph_idx=0,
                    sentence_idx=None,
                    clause_idx=None,
                )
            )
        return DocumentResult(
            clean_text=structural_text,
            annotation_spans=annotations,
            boundary_events=boundaries,
            segments=segments,
            warnings=warnings,
            header=dict(parsed.header),
            body=structural_text,
            diagnostics=diagnostics,
            metadata=metadata,
            structural_clean_text=structural_text,
        )

    @staticmethod
    def _restore_paragraphs(
        parsed: Any,
    ) -> tuple[str, list[tuple[int, int, Mapping[str, Any]]], list[Any]]:
        """Restore paragraph separators between adjacent structural div blocks."""
        raw_annotations = list(parsed.annotations)
        div_ranges = {
            (int(annotation.char_start), int(annotation.char_end))
            for annotation in raw_annotations
            if getattr(annotation, "kind", None) == "div"
        }
        starts = {start for start, _ in div_ranges}
        cuts = sorted(end for _, end in div_ranges if end in starts)
        if not cuts:
            return (
                parsed.clean_text,
                [
                    (int(item.char_start), int(item.char_end), item.attrs)
                    for item in raw_annotations
                ],
                list(parsed.events),
            )

        def shift(position: int, *, is_end: bool = False) -> int:
            if is_end:
                amount = sum(cut < position for cut in cuts)
            else:
                amount = (
                    sum(cut < position for cut in cuts) + sum(cut == position for cut in cuts) * 2
                )
            return position + amount

        parts: list[str] = []
        cursor = 0
        for cut in cuts:
            parts.append(parsed.clean_text[cursor:cut])
            parts.append("\n\n")
            cursor = cut + 1
        parts.append(parsed.clean_text[cursor:])
        adjusted_annotations = [
            (shift(int(item.char_start)), shift(int(item.char_end), is_end=True), item.attrs)
            for item in raw_annotations
        ]
        adjusted_events = [
            SimpleNamespace(
                pos=shift(int(item.pos)), kind=item.kind, attrs=item.attrs, anchor=item.anchor
            )
            for item in parsed.events
        ]
        return "".join(parts), adjusted_annotations, adjusted_events

    @staticmethod
    def _diagnostic(diagnostic: Any) -> SSMDDiagnostic:
        raw_severity = str(getattr(diagnostic, "severity", "warn"))
        severity = cast(
            Literal["info", "warn", "error"],
            raw_severity if raw_severity in {"info", "warn", "error"} else "warn",
        )
        return SSMDDiagnostic(
            code=str(getattr(diagnostic, "code", "ssmd.diagnostic")),
            severity=severity,
            message=str(getattr(diagnostic, "message", diagnostic)),
            line=getattr(diagnostic, "line", None),
            column=getattr(diagnostic, "column", None),
        )

    @staticmethod
    def _annotation_attrs(
        attrs: Mapping[str, Any], header: Mapping[str, Any], cfg: PipelineConfig
    ) -> dict[str, str]:
        result = {str(key): str(value) for key, value in attrs.items()}
        reference = result.get("voice") or result.get("voice_name")
        if reference:
            resolution = resolve_document_voice(
                reference,
                provider=cfg.ssmd.provider,
                api_bindings=cfg.ssmd.voice_bindings,
                header_bindings=header.get("voice_bindings", {}),
            )
            result["voice_reference"] = resolution.reference
            result["voice_name"] = resolution.target
            result["voice_source"] = resolution.source
        return result

    @classmethod
    def _events_to_boundaries(
        cls, events: list[Any], cfg: PipelineConfig, trace: Trace
    ) -> list[BoundaryEvent]:
        boundaries: list[BoundaryEvent] = []
        for event in events:
            attrs = {str(key): str(value) for key, value in event.attrs.items()}
            attrs["anchor"] = str(event.anchor)
            if event.kind == "mark":
                marker = attrs.get("name") or attrs.get("marker")
                if marker:
                    boundaries.append(
                        BoundaryEvent(
                            pos=int(event.pos), kind="marker", attrs={"marker": marker, **attrs}
                        )
                    )
                continue
            if event.kind == "paragraph":
                boundaries.append(
                    BoundaryEvent(
                        pos=int(event.pos),
                        kind="pause",
                        duration_s=None,
                        attrs={"strength": "p", **attrs},
                    )
                )
                continue
            if event.kind != "break":
                continue
            duration = cls._break_duration(attrs, cfg)
            if duration <= 0:
                continue
            boundaries.append(
                BoundaryEvent(pos=int(event.pos), kind="pause", duration_s=duration, attrs=attrs)
            )
        return boundaries

    @staticmethod
    def _break_duration(attrs: Mapping[str, str], cfg: PipelineConfig) -> float:
        value = attrs.get("time")
        if value:
            match = _BREAK_TIME_RE.match(value)
            if match:
                duration = float(match.group(1))
                return duration / 1000.0 if match.group(2).lower() == "ms" else duration
        return {
            "none": DEFAULT_PAUSE_NONE,
            "x-weak": DEFAULT_PAUSE_WEAK,
            "weak": DEFAULT_PAUSE_WEAK,
            "medium": cfg.generation.pause_clause or DEFAULT_PAUSE_CLAUSE,
            "strong": cfg.generation.pause_sentence or DEFAULT_PAUSE_SENTENCE,
            "x-strong": cfg.generation.pause_paragraph or DEFAULT_PAUSE_PARAGRAPH,
        }.get(attrs.get("strength", ""), 0.0)

    @staticmethod
    def _sentence_candidates(
        segments: list[Segment],
        pause_defaults: Any,
        pause_mode: str,
        pause_sentence: float,
    ) -> list[Any]:
        """Compatibility helper; the active frontend creates these after Phrasplit."""
        from ...ssmd_config import PauseCandidate

        if not segments:
            return []
        out: list[PauseCandidate] = []
        previous = segments[0]
        for current in segments[1:]:
            if (
                current.paragraph_idx == previous.paragraph_idx
                and current.sentence_idx != previous.sentence_idx
            ):
                duration = (
                    pause_defaults.sentence
                    if pause_defaults is not None and pause_defaults.sentence is not None
                    else pause_sentence
                )
                if pause_defaults is not None or pause_mode == "auto":
                    out.append(
                        PauseCandidate(
                            previous.char_end - 1, duration, "pipeline_default", "sentence", 100
                        )
                    )
            previous = current
        return out

    def _build_document(
        self,
        segments: Any,
        initial_pause: float,
        trace: Trace,
        pause_defaults: Any = None,
        cfg: PipelineConfig | None = None,
    ) -> tuple[str, list[AnnotationSpan], list[Any], list[Segment]]:
        """Compatibility renderer for callers of the former private helper."""
        _ = initial_pause
        cfg = cfg or PipelineConfig()
        parts: list[str] = []
        spans: list[AnnotationSpan] = []
        boundaries: list[Any] = []
        document_segments: list[Segment] = []
        cursor = 0
        previous_paragraph = None
        for index, item in enumerate(segments):
            metadata = getattr(item, "metadata", None)
            audio_src = getattr(metadata, "audio_src", None) if metadata else None
            if audio_src and not item.text.strip():
                self._warn_once(
                    trace, "SSMD audio segment has no alt_text; skipping audio segment."
                )
                continue
            if audio_src:
                self._warn_once(
                    trace, "SSMD audio segments are not mixed; speaking alt_text instead."
                )
            if previous_paragraph is not None and item.paragraph != previous_paragraph:
                parts.append("\n\n")
                boundaries.append(
                    PauseCandidate(
                        max(0, cursor - 1),
                        (
                            pause_defaults.paragraph
                            if pause_defaults is not None and pause_defaults.paragraph is not None
                            else cfg.generation.pause_paragraph
                        ),
                        "header_default" if pause_defaults is not None else "pipeline_default",
                        "paragraph",
                        200,
                    )
                )
                cursor += 2
            start = cursor
            parts.append(item.text)
            cursor += len(item.text)
            attrs = self._annotation_attrs(vars(metadata) if metadata else {}, {}, cfg)
            if attrs and cursor > start:
                spans.append(AnnotationSpan(start, cursor, attrs))
            if cursor > start:
                document_segments.append(
                    Segment(
                        id=f"p{item.paragraph}_s{item.sentence}_c0_seg{index}",
                        text=item.text,
                        char_start=start,
                        char_end=cursor,
                        meta=attrs,
                        paragraph_idx=item.paragraph,
                        sentence_idx=item.sentence,
                        clause_idx=0,
                    )
                )
            previous_paragraph = item.paragraph
        return "".join(parts), spans, boundaries, document_segments

    @staticmethod
    def _warn_once(trace: Trace, message: str) -> None:
        if message not in trace.warnings:
            trace.warnings.append(message)


__all__ = ["SsmdDocumentParser"]
