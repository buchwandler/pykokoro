"""Spokenform text preparation before sentence segmentation."""

from __future__ import annotations

from typing import Any

from spokenform import OffsetMap, PreparationConfig, ProtectedSpan, prepare_for_kokorog2p

from ...pipeline_config import PipelineConfig
from ...runtime.language_plan import build_language_plan
from ...types import AnnotationSpan, BoundaryEvent, TextPreparationInfo, Trace, TraceEvent
from ..protocols import DocumentResult, TextPreparer


class SpokenformTextPreparer(TextPreparer):
    """Prepare structural SSMD text once per maximal language run."""

    backend = "spokenform"

    def prepare(self, doc: DocumentResult, cfg: PipelineConfig, trace: Trace) -> DocumentResult:
        source = doc.clean_text
        if not source:
            doc.structural_clean_text = doc.structural_clean_text or source
            doc.preparation = TextPreparationInfo(source, source, languages=())
            return doc

        state = doc.linguistic_state
        if state is not None and hasattr(state, "source_plan") and state.source_plan:
            runs = state.source_plan
            source_analysis = {item.run: item for item in state.source_analysis}
        else:
            if cfg.generation.lang is None:
                raise ValueError("A document language is required before text preparation")
            runs = build_language_plan(
                source, doc.annotation_spans, default_language=cfg.generation.lang
            )
            source_analysis = {}
        output_parts: list[str] = []
        run_maps: list[tuple[int, int, OffsetMap, str]] = []
        replacements: list[dict[str, Any]] = []
        warnings: list[str] = []
        output_cursor = 0
        for run in runs:
            start, end, language = run.char_start, run.char_end, run.language
            run_source = source[start:end]
            analysis = source_analysis.get(run)
            explicit_map = OffsetMap.identity(len(run_source))
            protected = self._protected_spans(
                start, end, doc.annotation_spans, explicit_map, len(run_source)
            )
            prepared = prepare_for_kokorog2p(
                run_source,
                language=language,
                config=PreparationConfig.for_kokorog2p(language),
                annotations=self._spokenform_annotations(analysis.annotations) if analysis else None,
                nlp=analysis.doc if analysis else None,
                protected_spans=protected,
            )
            combined_map = explicit_map.compose(
                prepared.offset_map or OffsetMap.identity(len(run_source))
            )
            spoken = prepared.spoken_text
            output_parts.append(spoken)
            run_maps.append((start, end, combined_map, language))
            replacements.extend(
                self._global_replacements(prepared, start, output_cursor, explicit_map)
            )
            warnings.extend(str(item) for item in prepared.warnings)
            output_cursor += len(spoken)

        spoken_text = "".join(output_parts)
        document_map = self._document_map(len(source), spoken_text, run_maps)
        remapped_annotations = [
            AnnotationSpan(
                char_start=document_map.map_source_span(span.char_start, span.char_end)[0],
                char_end=document_map.map_source_span(span.char_start, span.char_end)[1],
                attrs=dict(span.attrs),
            )
            for span in doc.annotation_spans
            if span.char_start <= span.char_end
        ]
        remapped_events: list[BoundaryEvent] = []
        for event in doc.boundary_events:
            remapped = self._remap_event(event, document_map)
            if remapped is not None:
                remapped_events.append(remapped)
        for previous, current in zip(doc.segments, doc.segments[1:], strict=False):
            if previous.paragraph_idx != current.paragraph_idx:
                position = document_map.source_to_output(previous.char_end, bias="left")
                if not any(
                    event.kind == "pause"
                    and event.attrs.get("strength") == "p"
                    and event.pos in {position, max(0, position - 1)}
                    for event in remapped_events
                ):
                    remapped_events.append(
                        BoundaryEvent(
                            pos=position,
                            kind="pause",
                            duration_s=None,
                            attrs={"strength": "p", "anchor": "after"},
                        )
                    )
        languages = tuple(dict.fromkeys(language for _, _, _, language in run_maps))
        version = getattr(__import__("spokenform"), "__version__", None)
        info = TextPreparationInfo(
            source_text=source,
            spoken_text=spoken_text,
            replacements=tuple(replacements),
            offset_map=document_map.to_dict(),
            languages=languages,
            backend=self.backend,
            version=str(version) if version is not None else None,
            warnings=tuple(warnings),
        )
        doc.clean_text = spoken_text
        doc.annotation_spans = remapped_annotations
        doc.boundary_events = remapped_events
        doc.segments = []
        doc.structural_clean_text = source
        doc.preparation = info
        doc.metadata["text_preparation"] = {
            "backend": info.backend,
            "version": info.version,
            "changed": source != spoken_text,
            "replacements": len(info.replacements),
            "languages": list(info.languages),
            "warnings": list(info.warnings),
        }
        trace.warnings.extend(item for item in warnings if item not in trace.warnings)
        trace.events.append(
            TraceEvent(
                stage="text_preparation",
                name="prepare",
                ms=0.0,
                details={
                    "backend": self.backend,
                    "changed": source != spoken_text,
                    "replacements": len(replacements),
                    "languages": list(languages),
                    "warnings": len(warnings),
                },
            )
        )
        return doc



    @staticmethod
    def _spokenform_annotations(annotations: tuple[Any, ...]) -> tuple[Any, ...]:
        if not annotations:
            return ()
        from spokenform import TokenAnnotation as SpokenformTokenAnnotation

        return tuple(
            SpokenformTokenAnnotation(
                start=item.start,
                end=item.end,
                text=item.text,
                pos=item.pos,
                tag=item.tag,
                lemma=item.lemma,
                language=item.language,
            )
            for item in annotations
        )


    @staticmethod
    def _protected_spans(
        run_start: int,
        run_end: int,
        spans: list[AnnotationSpan],
        explicit_map: OffsetMap,
        intermediate_length: int,
    ) -> tuple[ProtectedSpan, ...]:
        protected: list[ProtectedSpan] = []
        for span in spans:
            if "ph" not in span.attrs and "phonemes" not in span.attrs:
                continue
            if span.char_start < run_start or span.char_end > run_end:
                continue
            start, end = explicit_map.map_source_span(
                span.char_start - run_start, span.char_end - run_start
            )
            if 0 <= start <= end <= intermediate_length and start < end:
                protected.append(ProtectedSpan(start, end, kind="g2p-override", source="ssmd"))
        return tuple(protected)

    @staticmethod
    def _global_replacements(
        prepared: Any, run_start: int, output_start: int, explicit_map: OffsetMap
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for item in prepared.source_replacements:
            source_start, source_end = explicit_map.map_output_span(
                item.source_start, item.source_end
            )
            result.append(
                {
                    "source_start": run_start + source_start,
                    "source_end": run_start + source_end,
                    "output_start": output_start + item.output_start,
                    "output_end": output_start + item.output_end,
                    "source": item.source,
                    "replacement": item.replacement,
                    "kind": item.kind,
                    "language": item.language,
                    "rule": item.rule,
                }
            )
        return result

    @staticmethod
    def _document_map(
        source_length: int,
        spoken_text: str,
        runs: list[tuple[int, int, OffsetMap, str]],
    ) -> OffsetMap:
        source_left = [0] * (source_length + 1)
        source_right = [0] * (source_length + 1)
        output_left: list[int] = []
        output_right: list[int] = []
        output_cursor = 0
        for start, end, mapping, _language in runs:
            for local in range(end - start + 1):
                source_left[start + local] = output_cursor + mapping.source_to_output(
                    local, bias="left"
                )
                source_right[start + local] = output_cursor + mapping.source_to_output(
                    local, bias="right"
                )
            output_boundaries = range(mapping.output_length + 1)
            if output_left:
                output_boundaries = range(1, mapping.output_length + 1)
            for local in output_boundaries:
                output_left.append(start + mapping.output_to_source(local, bias="left"))
                output_right.append(start + mapping.output_to_source(local, bias="right"))
            output_cursor += mapping.output_length
        if not runs:
            return OffsetMap.identity(source_length)
        source_left[0] = 0
        source_right[0] = 0
        source_left[-1] = len(spoken_text)
        source_right[-1] = len(spoken_text)
        return OffsetMap.from_boundaries(
            source_length,
            len(spoken_text),
            tuple(source_left),
            tuple(source_right),
            tuple(output_left[: len(spoken_text) + 1]),
            tuple(output_right[: len(spoken_text) + 1]),
        )

    @staticmethod
    def _remap_event(event: BoundaryEvent, mapping: OffsetMap) -> BoundaryEvent | None:
        anchor = event.attrs.get("anchor", "after")
        bias = "left" if anchor == "before" else "right"
        try:
            pos = mapping.source_to_output(event.pos, bias=bias)
        except IndexError:
            return None
        return BoundaryEvent(
            pos=pos, kind=event.kind, duration_s=event.duration_s, attrs=dict(event.attrs)
        )


__all__ = ["SpokenformTextPreparer"]
