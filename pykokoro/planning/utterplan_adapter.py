from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from utterplan import CURRENT_SCHEMA_VERSION, UtterancePlan

from ..exceptions import PlanConfigurationConflict, PlanConsumptionError
from ..stages.protocols import DocumentResult
from ..types import AnnotationSpan, BoundaryEvent, Segment, TextPreparationInfo
from .config_adapter import PLANNING_OVERRIDE_KEYS


@dataclass(frozen=True, slots=True)
class RenderSegment:
    """A renderer-owned view of a plan segment.

    ``char_start`` and ``char_end`` are offsets into the plan's spoken text,
    never offsets into its structural source text.
    """

    id: str
    text: str
    char_start: int
    char_end: int
    paragraph_idx: int
    sentence_idx: int
    clause_idx: int
    language: str
    token_indices: tuple[int, ...]
    annotation_ids: tuple[str, ...]
    pause_before: float
    pause_after: float
    directives: Mapping[str, Any]

    def to_segment(self) -> Segment:
        return Segment(
            id=self.id,
            text=self.text,
            char_start=self.char_start,
            char_end=self.char_end,
            paragraph_idx=self.paragraph_idx,
            sentence_idx=self.sentence_idx,
            clause_idx=self.clause_idx,
            meta={
                "language": self.language,
                "plan_segment_id": self.id,
                "plan_token_indices": self.token_indices,
                "plan_annotation_ids": self.annotation_ids,
                "plan_directives": dict(self.directives),
            },
        )


@dataclass(frozen=True, slots=True)
class RenderPlan:
    """Immutable renderer adapter for an UtterancePlan."""

    plan: UtterancePlan
    spoken_text: str
    source_text: str | None
    segments: tuple[RenderSegment, ...]
    document: DocumentResult
    document_metadata: Mapping[str, Any]


def _directive_attrs(segment: Any) -> dict[str, str]:
    directives = segment.directives
    attrs: dict[str, str] = {}
    voice = directives.voice
    if voice is not None:
        attrs["voice_reference"] = voice.reference
        attrs["voice"] = voice.reference
    pronunciation = directives.pronunciation
    if pronunciation is not None:
        attrs["ph"] = pronunciation.phonemes
        attrs["phoneme_alphabet"] = pronunciation.alphabet
    prosody = directives.prosody
    if prosody is not None:
        for source, target in (
            ("rate", "prosody_rate"),
            ("pitch", "prosody_pitch"),
            ("volume", "prosody_volume"),
        ):
            value = getattr(prosody, source)
            if value is not None:
                attrs[target] = value
    emphasis = directives.emphasis
    if emphasis is not None:
        attrs["emphasis"] = emphasis.level
    audio = directives.audio
    if audio is not None:
        for source, target in (
            ("src", "audio_src"),
            ("alt_text", "audio_alt_text"),
            ("clip_begin", "audio_clip_begin"),
            ("clip_end", "audio_clip_end"),
            ("speed", "audio_speed"),
            ("repeat_duration", "audio_repeat_dur"),
            ("repeat_count", "audio_repeat_count"),
            ("sound_level", "audio_sound_level"),
        ):
            value = getattr(audio, source)
            if value is not None:
                attrs[target] = str(value)
    return attrs


def _metadata(plan: UtterancePlan) -> dict[str, Any]:
    metadata = deepcopy(dict(plan.document_metadata))
    header = metadata.get("header")
    if isinstance(header, Mapping):
        for key in ("voice_bindings", "language_detection", "title", "pause_defaults"):
            if key in header and key not in metadata:
                metadata[key] = deepcopy(header[key])
    metadata["utterplan"] = {
        "plan_id": plan.plan_id,
        "producer": deepcopy(dict(plan.producer)),
        "schema_version": plan.schema_version,
    }
    metadata["utterplan_tokens"] = tuple(plan.tokens)
    metadata["utterplan_linguistic_runs"] = tuple(plan.linguistic_runs)
    metadata["utterplan_units"] = tuple(plan.units)
    metadata["utterplan_markers"] = tuple(plan.markers)
    metadata["utterplan_annotations"] = tuple(plan.annotations)
    metadata["utterplan_pauses"] = {
        item.id: (item.pause_before.seconds, item.pause_after.seconds) for item in plan.segments
    }
    metadata["language_detection"] = deepcopy(metadata.get("language_detection"))
    if "voice_bindings" not in metadata:
        metadata["voice_bindings"] = {}
    return metadata


def adapt_plan(plan: UtterancePlan) -> RenderPlan:
    """Validate and adapt a plan without changing any plan-owned value."""
    if not isinstance(plan, UtterancePlan):
        raise PlanConsumptionError(
            f"run_plan() requires utterplan.UtterancePlan, got {type(plan).__name__}"
        )
    if plan.schema_version != CURRENT_SCHEMA_VERSION:
        raise PlanConsumptionError(
            f"Unsupported UtterPlan schema version {plan.schema_version}; "
            f"expected {CURRENT_SCHEMA_VERSION}"
        )
    try:
        plan.validate()
    except Exception as exc:
        raise PlanConsumptionError(f"Invalid UtterPlan: {exc}") from exc

    segments = tuple(
        RenderSegment(
            id=item.id,
            text=item.text,
            char_start=item.spoken_start,
            char_end=item.spoken_end,
            paragraph_idx=item.paragraph,
            sentence_idx=item.sentence,
            clause_idx=item.clause,
            language=item.language,
            token_indices=tuple(item.token_indices),
            annotation_ids=tuple(item.annotation_ids),
            pause_before=item.pause_before.seconds,
            pause_after=item.pause_after.seconds,
            directives=MappingProxyType(_directive_attrs(item)),
        )
        for item in plan.segments
    )
    metadata = _metadata(plan)
    annotations = [
        AnnotationSpan(
            char_start=item.spoken_start,
            char_end=item.spoken_end,
            attrs={str(key): str(value) for key, value in item.attrs.items()},
        )
        for item in plan.annotations
        if item.spoken_start is not None and item.spoken_end is not None
    ]
    boundaries = [
        BoundaryEvent(
            pos=item.position,
            kind="pause",
            duration_s=item.seconds,
            attrs={"origin": item.origin, "strength": item.strength or ""},
        )
        for item in plan.boundaries
        if item.kind == "pause"
    ]
    boundaries.extend(
        BoundaryEvent(pos=item.spoken_position, kind="marker", attrs={"marker": item.name})
        for item in plan.markers
    )
    document = DocumentResult(
        clean_text=plan.texts.spoken,
        structural_clean_text=plan.texts.structural,
        annotation_spans=annotations,
        boundary_events=boundaries,
        segments=[item.to_segment() for item in segments],
        warnings=list(plan.warnings),
        header=metadata,
        body=plan.texts.spoken,
        metadata=metadata,
        preparation=TextPreparationInfo(
            source_text=plan.texts.structural,
            spoken_text=plan.texts.spoken,
            backend=plan.preparation.backend,
            version=plan.preparation.version,
            languages=tuple(plan.preparation.languages),
            replacements=tuple(dict(item) for item in plan.preparation.replacements),
            warnings=tuple(plan.preparation.warnings),
        ),
    )
    return RenderPlan(
        plan=plan,
        spoken_text=plan.texts.spoken,
        source_text=plan.texts.structural,
        segments=segments,
        document=document,
        document_metadata=MappingProxyType(metadata),
    )


def assert_renderer_overrides(overrides: Mapping[str, Any]) -> None:
    for key in overrides:
        if key in PLANNING_OVERRIDE_KEYS:
            raise PlanConfigurationConflict(
                f"run_plan() received planning override {key!r}; create a new UtterancePlan instead."
            )
