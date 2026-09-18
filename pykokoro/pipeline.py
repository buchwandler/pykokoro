from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import nullcontext as _nullcontext
from copy import copy, deepcopy
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from audiocompose import AudioJob
from typing_extensions import Self
from utterplan import UtterancePlan, UtterancePlanner

from .audio_generator import _collect_unit_word_timings
from .audio_job import rendered_segments_to_audio_job, waveform_to_audio_job
from .composition import compose_audio_job
from .constants import SAMPLE_RATE
from .emphasis import apply_emphasis_policy
from .exceptions import ConfigurationError, PlanConsumptionError
from .generation_config import GenerationConfig
from .language_detection import LanguageDetectionConfig
from .loudness_config import LoudnessConfig
from .pipeline_config import PipelineConfig, require_document_language, resolve_model_defaults
from .planning import adapt_plan, assert_renderer_overrides, planner_config_from_pipeline
from .runtime.language_plan import build_language_plan
from .runtime.linguistics import (
    LinguisticRequestState,
    LinguisticResourcePool,
    PreparedRunAnalysis,
)
from .runtime.tracing import trace_timing
from .spacy_models import SpacyModelSize
from .ssmd_config import SSMDRenderConfig
from .stages.doc_parsers.ssmd import SsmdDocumentParser
from .stages.g2p.kokorog2p import KokoroG2PAdapter
from .stages.protocols import (
    AudioGeneratorStage,
    AudioPostprocessor,
    DocumentParser,
    G2PAdapter,
    PhonemeProcessor,
    SentenceSegmenter,
    TextPreparer,
)
from .stages.segmentation.phrasplit import PhrasplitSentenceSegmenter
from .stages.text_preparation.spokenform import SpokenformTextPreparer
from .types import (
    AudioResult,
    AudioUnitDescriptor,
    AudioUnitKind,
    AudioUnitResult,
    BoundaryEvent,
    PhonemeSegment,
    Segment,
    Trace,
    TraceEvent,
)

if TYPE_CHECKING:
    from .onnx_backend import Kokoro
    from .tokenizer import TokenizerConfig

logger = logging.getLogger(__name__)


def _acoustic_tokenizer_key(config: TokenizerConfig | None) -> object:
    """Return tokenizer settings that affect the acoustic backend."""
    if config is None:
        return None
    # Named lexicons affect frontend G2P only, not the acoustic runtime.
    return _freeze_config_value(replace(config, lexicons=None, lexicon_data_policy="auto"))


def _load_default_onnx_adapters() -> tuple[type[Any], type[Any], type[Any]]:
    """Load ONNX stage classes only when a default stage is actually needed."""
    try:
        from .stages.audio_generation.onnx import OnnxAudioGenerationAdapter
        from .stages.audio_postprocessing.onnx import OnnxAudioPostprocessingAdapter
        from .stages.phoneme_processing.onnx import OnnxPhonemeProcessorAdapter
    except ModuleNotFoundError as exc:
        if exc.name == "onnxruntime":
            raise RuntimeError(
                "ONNX-backed pipeline stages require ONNX Runtime; install "
                "pykokoro[cpu] or a platform provider extra."
            ) from exc
        raise
    return (
        OnnxPhonemeProcessorAdapter,
        OnnxAudioGenerationAdapter,
        OnnxAudioPostprocessingAdapter,
    )


@dataclass(frozen=True, slots=True)
class _PreparedUnitGroup:
    descriptor: AudioUnitDescriptor
    phoneme_start: int
    phoneme_end: int
    marker_events: tuple[Any, ...]


@dataclass(slots=True)
class _PreparedDocument:
    cfg: PipelineConfig
    unit_kind: AudioUnitKind
    trace: Trace
    doc: Any
    segments: list[Segment]
    phoneme_segments: list[PhonemeSegment]
    groups: tuple[_PreparedUnitGroup, ...]
    phoneme_processor: PhonemeProcessor
    audio_generator: AudioGeneratorStage
    audio_postprocessor: AudioPostprocessor


@dataclass(slots=True)
class _PreparedAudioJobContext:
    job: AudioJob
    cfg: PipelineConfig
    segments: list[Segment]
    phoneme_segments: list[PhonemeSegment]
    trace: Trace
    document_metadata: dict[str, Any]
    clean_text: str
    source_text: str | None
    marker_positions: dict[str, int]


@dataclass(slots=True)
class PreparedFrontend:
    """Reusable lexicon-independent frontend preparation for one request."""

    _pipeline: KokoroPipeline
    _cfg: PipelineConfig
    _trace: Trace
    _doc: Any
    _segments: list[Segment]
    _state: LinguisticRequestState
    _unit_kind: AudioUnitKind
    _plan: UtterancePlan | None = None
    _closed: bool = False

    @property
    def text(self) -> str:
        self._ensure_open()
        return self._doc.clean_text

    @property
    def segments(self) -> tuple[Segment, ...]:
        self._ensure_open()
        return tuple(self._segments)

    @property
    def trace(self) -> Trace:
        self._ensure_open()
        return self._trace

    @property
    def document_metadata(self) -> Mapping[str, Any]:
        self._ensure_open()
        return MappingProxyType(_copy_metadata_value(self._doc.metadata))

    def close(self) -> None:
        if self._closed:
            return
        self._state.release_docs()
        self._doc.linguistic_state = None
        self._segments.clear()
        self._doc = None
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("PreparedFrontend is closed")

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class PreparedAudioUnits:
    """A globally prepared document that can render selected units sequentially."""

    def __init__(self, pipeline: KokoroPipeline, prepared: _PreparedDocument) -> None:
        self._pipeline = pipeline
        self._prepared: _PreparedDocument | None = prepared
        self._units = tuple(group.descriptor for group in prepared.groups)
        self._unit_kind = prepared.unit_kind
        self._clean_text = prepared.doc.clean_text
        self._source_text = prepared.doc.structural_clean_text
        self._document_metadata = {
            "title": _copy_metadata_value(prepared.doc.header.get("title")),
            "voice_bindings": _copy_metadata_value(prepared.doc.header.get("voice_bindings", {})),
            "pause_defaults": _copy_metadata_value(prepared.doc.header.get("pause_defaults", {})),
        }
        self._document_metadata.update(_copy_metadata_value(prepared.doc.metadata))
        self._diagnostics = tuple(prepared.doc.diagnostics)
        self._closed = False
        self._render_started = False
        self._render_active = False
        self._active_result: AudioUnitResult | None = None

    @property
    def units(self) -> tuple[AudioUnitDescriptor, ...]:
        return self._units

    @property
    def unit_kind(self) -> AudioUnitKind:
        return self._unit_kind

    @property
    def clean_text(self) -> str:
        """Prepared text used as the coordinate space for segments."""
        return self._clean_text

    @property
    def source_text(self) -> str | None:
        """Structural text before spokenform preparation."""
        return self._source_text

    @property
    def document_metadata(self) -> Mapping[str, Any]:
        return MappingProxyType(_copy_metadata_value(self._document_metadata))

    @property
    def diagnostics(self) -> Sequence[Any]:
        return self._diagnostics

    def render(
        self,
        *,
        indices: Iterable[int] | None = None,
        skip_indices: Collection[int] = (),
    ) -> Iterator[AudioUnitResult]:
        """Render selected units in order, allowing only one render pass."""
        if self._closed:
            raise RuntimeError("PreparedAudioUnits is closed")
        if self._render_started:
            raise RuntimeError("PreparedAudioUnits supports one render pass only")

        selected = self._normalize_indices(indices, "indices")
        skipped = self._normalize_indices(skip_indices, "skip_indices")
        selected_set = set(range(len(self._units))) if selected is None else set(selected)
        selected_set.difference_update(skipped or ())
        ordered = tuple(index for index in range(len(self._units)) if index in selected_set)
        self._render_started = True
        self._render_active = True
        return self._iterate(ordered)

    def _normalize_indices(self, values: Iterable[int] | None, name: str) -> tuple[int, ...] | None:
        if values is None:
            return None
        normalized = tuple(values)
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{name} contains duplicate unit indices")
        invalid: list[object] = []
        for value in normalized:
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                or value >= len(self._units)
            ):
                invalid.append(value)
        if invalid:
            raise IndexError(f"{name} contains out-of-range unit index {invalid[0]!r}")
        return tuple(sorted(normalized))

    def _iterate(self, ordered: tuple[int, ...]) -> Iterator[AudioUnitResult]:
        previous: AudioUnitResult | None = None
        try:
            for index in ordered:
                if previous is not None:
                    previous.release_audio()
                    self._active_result = None
                prepared = self._prepared
                if prepared is None or self._closed:
                    raise RuntimeError("PreparedAudioUnits is closed")
                result = self._pipeline._render_prepared_unit(prepared, index)
                self._document_metadata.update(_copy_metadata_value(result.document_metadata))
                previous = result
                self._active_result = result
                yield result
        finally:
            if previous is not None:
                previous.release_audio()
            self._active_result = None
            self._render_active = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._active_result is not None:
            self._active_result.release_audio()
            self._active_result = None
        prepared = self._prepared
        if prepared is not None:
            for segment in prepared.phoneme_segments:
                segment.raw_audio = None
                segment.processed_audio = None
            prepared.segments.clear()
            prepared.phoneme_segments.clear()
            prepared.groups = ()
        self._prepared = None
        self._render_active = False
        self._pipeline._unregister_prepared(self)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _coerce_generation(base: GenerationConfig, value: Any) -> GenerationConfig:
    if value is None:
        return base
    if isinstance(value, GenerationConfig):
        return value
    if isinstance(value, Mapping):
        return replace(base, **dict(value))
    raise TypeError(f"generation must be GenerationConfig | Mapping | None, got {type(value)!r}")


def _coerce_loudness(base: LoudnessConfig, value: Any) -> LoudnessConfig:
    if value is None:
        return base
    if isinstance(value, LoudnessConfig):
        return value
    if isinstance(value, Mapping):
        return replace(base, **dict(value))
    raise TypeError(f"loudness must be LoudnessConfig | Mapping | None, got {type(value)!r}")


def _coerce_ssmd(base: SSMDRenderConfig, value: Any) -> SSMDRenderConfig:
    if value is None:
        return base
    if isinstance(value, SSMDRenderConfig):
        return value
    if isinstance(value, Mapping):
        data = dict(value)
        pause_defaults = data.get("pause_defaults")
        if isinstance(pause_defaults, Mapping):
            from .ssmd_config import SSMDPauseOverrides

            data["pause_defaults"] = SSMDPauseOverrides(**dict(pause_defaults))
        return replace(base, **data)
    raise TypeError(f"ssmd must be SSMDRenderConfig | Mapping | None, got {type(value)!r}")


def _coerce_tokenizer(base: TokenizerConfig | None, value: Any) -> TokenizerConfig:
    from .tokenizer import TokenizerConfig

    current = base or TokenizerConfig()
    if value is None:
        return current
    if isinstance(value, TokenizerConfig):
        return value
    if isinstance(value, Mapping):
        return replace(current, **dict(value))
    raise TypeError(
        f"tokenizer_config must be TokenizerConfig | Mapping | None, got {type(value)!r}"
    )


def _coerce_language_detection(
    value: Any,
) -> LanguageDetectionConfig | None:
    if value is None:
        return None
    if isinstance(value, LanguageDetectionConfig):
        return value
    if isinstance(value, Mapping):
        return LanguageDetectionConfig(
            mode=value.get("mode", "off"),
            languages=tuple(value.get("languages", ())),
        )
    raise TypeError(
        f"language_detection must be LanguageDetectionConfig | Mapping | None, got {type(value)!r}"
    )


def _coerce_paths_inplace(data: dict[str, Any]) -> None:
    # Convenience: accept str paths in config dict.
    for key in ("model_path", "voices_path", "model_config_path", "release_manifest_path"):
        v = data.get(key)
        if isinstance(v, str):
            data[key] = Path(v)


def _coerce_pipeline_config(
    value: PipelineConfig | Mapping[str, Any] | None,
) -> PipelineConfig:
    if value is None:
        return PipelineConfig()

    if isinstance(value, PipelineConfig):
        return value

    if isinstance(value, Mapping):
        data = dict(value)
        gen_value = data.pop("generation", None)
        loudness_value = data.pop("loudness", None)
        ssmd_value = data.pop("ssmd", None)
        tokenizer_value = data.pop("tokenizer_config", None)
        language_detection_value = data.pop("language_detection", None)

        _coerce_paths_inplace(data)
        cfg = PipelineConfig(**data)

        if gen_value is not None:
            cfg = replace(cfg, generation=_coerce_generation(cfg.generation, gen_value))
        if loudness_value is not None:
            cfg = replace(cfg, loudness=_coerce_loudness(cfg.loudness, loudness_value))
        if ssmd_value is not None:
            cfg = replace(cfg, ssmd=_coerce_ssmd(cfg.ssmd, ssmd_value))
        if tokenizer_value is not None:
            cfg = replace(
                cfg,
                tokenizer_config=_coerce_tokenizer(cfg.tokenizer_config, tokenizer_value),
            )

        if language_detection_value is not None:
            cfg = replace(
                cfg,
                language_detection=_coerce_language_detection(language_detection_value),
            )

        return cfg

    raise TypeError(f"config must be PipelineConfig | Mapping | None, got {type(value)!r}")


def _merge_config(
    base: PipelineConfig,
    overrides: Mapping[str, Any] | None,
) -> PipelineConfig:
    if not overrides:
        return base

    data = dict(overrides)
    gen_value = data.pop("generation", None)
    loudness_value = data.pop("loudness", None)
    ssmd_value = data.pop("ssmd", None)
    tokenizer_value = data.pop("tokenizer_config", None)
    language_detection_value = data.pop("language_detection", None)

    _coerce_paths_inplace(data)
    cfg = replace(base, **data)

    if gen_value is not None:
        cfg = replace(cfg, generation=_coerce_generation(cfg.generation, gen_value))
    if loudness_value is not None:
        cfg = replace(cfg, loudness=_coerce_loudness(cfg.loudness, loudness_value))
    if ssmd_value is not None:
        cfg = replace(cfg, ssmd=_coerce_ssmd(cfg.ssmd, ssmd_value))
    if tokenizer_value is not None:
        cfg = replace(
            cfg,
            tokenizer_config=_coerce_tokenizer(cfg.tokenizer_config, tokenizer_value),
        )

    if language_detection_value is not None:
        cfg = replace(
            cfg,
            language_detection=_coerce_language_detection(language_detection_value),
        )

    return cfg


def _copy_config_for_preparation(cfg: PipelineConfig) -> PipelineConfig:
    """Copy preparation settings without copying runtime progress callbacks."""
    progress = cfg.asset_progress
    copied = deepcopy(replace(cfg, asset_progress=None))
    return replace(copied, asset_progress=progress)


PipelineConfigTransform = Callable[[PipelineConfig], PipelineConfig]


def _apply_spacy_model_settings(
    cfg: PipelineConfig,
    *,
    model: str | None,
    size: SpacyModelSize | None,
    use_spacy: bool | None,
) -> PipelineConfig:
    from .tokenizer import TokenizerConfig

    tokenizer_config: TokenizerConfig = cfg.tokenizer_config or TokenizerConfig()
    tokenizer_config = replace(
        tokenizer_config,
        use_spacy=use_spacy,
        spacy_model=model,
        spacy_model_size=size,
    )
    return replace(cfg, tokenizer_config=tokenizer_config)


def with_spacy_model(
    model: str | PipelineConfig | Mapping[str, Any] | None = None,
    *,
    size: SpacyModelSize | None = None,
    use_spacy: bool | None = None,
) -> PipelineConfigTransform | PipelineConfig:
    """Create a pipeline transform for one spaCy model selection request."""

    if isinstance(model, (PipelineConfig, Mapping)):
        return _apply_spacy_model_settings(
            _coerce_pipeline_config(model),
            model=None,
            size=size,
            use_spacy=use_spacy,
        )

    def transform(config: PipelineConfig) -> PipelineConfig:
        return _apply_spacy_model_settings(
            _coerce_pipeline_config(config),
            model=model,
            size=size,
            use_spacy=use_spacy,
        )

    return transform


def with_spacy_model_size(
    config: PipelineConfig | Mapping[str, Any] | None = None,
    *,
    size: SpacyModelSize | None = None,
    model: str | None = None,
) -> PipelineConfig:
    """Return a config with an intentional exact tier or explicit model.

    Omitting both settings leaves the request unset and never introduces a
    medium default.
    """

    return _apply_spacy_model_settings(
        _coerce_pipeline_config(config),
        model=model,
        size=size,
        use_spacy=True,
    )


def build_pipeline(
    *,
    config: PipelineConfig | Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
    backend: Kokoro | None = None,
    eager: bool = False,
    # stage overrides for advanced usage/testing
    doc_parser: DocumentParser | None = None,
    text_preparer: TextPreparer | None = None,
    sentence_segmenter: SentenceSegmenter | None = None,
    g2p: G2PAdapter | None = None,
    phoneme_processing: PhonemeProcessor | None = None,
    audio_generation: AudioGeneratorStage | None = None,
    audio_postprocessing: AudioPostprocessor | None = None,
) -> KokoroPipeline:
    """
    Construct a :class:`KokoroPipeline` from a single,
    user-friendly configuration surface.

    This helper is the recommended way to create pipelines.
    It supports:

    - **Single-shot configuration** via ``config=`` (a :class:`PipelineConfig`
        or a dict-like object).
    - **Predictable overrides** via ``overrides=``, which always take precedence
        over ``config``.
    - **Nested generation config**: both ``config`` and ``overrides`` may include
        a ``"generation"`` key
      as a :class:`GenerationConfig` or a mapping. Mappings are merged onto
      the existing
      :class:`GenerationConfig` (i.e. you can override only ``lang``
      or only ``speed``).
    - **Path convenience**: string values for ``model_path`` and
      ``voices_path`` are automatically
      converted to :class:`~pathlib.Path`.

    Precedence
    ----------
    The effective configuration is computed in this order (later wins):

    1. ``PipelineConfig()`` defaults
    2. ``config=`` (if provided)
    3. ``overrides=`` (if provided)

    Backend and stage wiring
    ------------------------
    By default the returned pipeline is *lazy*: it will create and own a
    :class:`~pykokoro.onnx_backend.Kokoro`
    instance on first use (via :meth:`KokoroPipeline.run`) based on the
    resolved :class:`PipelineConfig`.

    If ``backend`` is provided, the default ONNX stages are bound to that
    backend (unless you provide
    explicit stage instances). In this mode the pipeline does **not**
    manage the backend lifecycle.

    Eager initialization
    --------------------
    If ``eager=True`` and ``backend is None``, the pipeline will immediately:

    - create and own the :class:`~pykokoro.onnx_backend.Kokoro` backend,
    - create default ONNX stages (phoneme processing, audio generation,
        audio postprocessing),
    - fail fast if model/provider/session configuration is invalid or required
        files cannot be loaded.

    Stage overrides (advanced)
    --------------------------
    You may pass custom stage instances (``doc_parser``, ``g2p``,
    ``phoneme_processing``,
    ``audio_generation``, ``audio_postprocessing``) for testing
    or experimentation.
    Unspecified stages fall back to the library defaults.

    Examples
    --------
    Configure everything in one dict (including nested generation settings)::

        pipe = build_pipeline(
            config={
                "voice": "af_nova",
                "model_source": "huggingface",
                "model_variant": "v1.0",
                "provider": "cpu",
                "generation": {"lang": "en-us", "speed": 1.05},
            },
            eager=True,
        )

    Override only one generation field without replacing the others::

        pipe = build_pipeline(
            config={"voice": "af_nova", "generation": {"lang": "en-us", "speed": 1.0}},
            overrides={"generation": {"speed": 0.9}},
        )

    Args:
        config: Base pipeline configuration.
            May be a :class:`PipelineConfig` or a mapping.
        overrides: Additional configuration applied on top of ``config``.
            Always wins.
        backend: Optional pre-constructed :class:`~pykokoro.onnx_backend.Kokoro`
            to bind ONNX stages to.
        eager: If true (and no ``backend`` is supplied), eagerly create/own
            backend and default stages.
        doc_parser: Optional document parser stage.
        g2p: Optional grapheme-to-phoneme stage.
        phoneme_processing: Optional phoneme processing stage.
        audio_generation: Optional audio generation stage.
        audio_postprocessing: Optional audio postprocessing stage.

    Returns:
        A configured :class:`KokoroPipeline` instance.
    """
    cfg = _coerce_pipeline_config(config)
    cfg = _merge_config(cfg, overrides)

    pipeline = KokoroPipeline(
        cfg,
        doc_parser=doc_parser or SsmdDocumentParser(),
        text_preparer=text_preparer or SpokenformTextPreparer(),
        sentence_segmenter=sentence_segmenter or PhrasplitSentenceSegmenter(),
        g2p=g2p or KokoroG2PAdapter(),
        phoneme_processing=phoneme_processing,
        audio_generation=audio_generation,
        audio_postprocessing=audio_postprocessing,
    )

    # If backend injected: bind default stages to it
    # (unless user already provided stages)
    if backend is not None:
        (
            onnx_phoneme_processor,
            onnx_audio_generation,
            onnx_audio_postprocessing,
        ) = _load_default_onnx_adapters()
        if pipeline.phoneme_processing is None:
            pipeline.phoneme_processing = onnx_phoneme_processor(
                backend,
                context_phonemizer=getattr(pipeline.g2p, "phonemize_context", None),
            )
        if pipeline.audio_generation is None:
            pipeline.audio_generation = onnx_audio_generation(backend)
        if pipeline.audio_postprocessing is None:
            pipeline.audio_postprocessing = onnx_audio_postprocessing(backend)
        return pipeline

    # Eager warmup: create backend now + bind stages + own/close them
    if eager:
        kokoro, _ = pipeline._ensure_kokoro(cfg)
        (
            onnx_phoneme_processor,
            onnx_audio_generation,
            onnx_audio_postprocessing,
        ) = _load_default_onnx_adapters()

        if pipeline.phoneme_processing is None:
            pipeline.phoneme_processing = onnx_phoneme_processor(
                kokoro,
                context_phonemizer=getattr(pipeline.g2p, "phonemize_context", None),
            )
            pipeline._owns_phoneme_processing = True

        if pipeline.audio_generation is None:
            pipeline.audio_generation = onnx_audio_generation(kokoro)
            pipeline._owns_audio_generation = True

        if pipeline.audio_postprocessing is None:
            pipeline.audio_postprocessing = onnx_audio_postprocessing(kokoro)
            pipeline._owns_audio_postprocessing = True

    return pipeline


class KokoroPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        doc_parser: DocumentParser | None = None,
        text_preparer: TextPreparer | None = None,
        sentence_segmenter: SentenceSegmenter | None = None,
        g2p: G2PAdapter | None = None,
        phoneme_processing: PhonemeProcessor | None = None,
        audio_generation: AudioGeneratorStage | None = None,
        audio_postprocessing: AudioPostprocessor | None = None,
    ) -> None:
        self._legacy_frontend_compat = any(
            value is not None for value in (doc_parser, text_preparer, sentence_segmenter)
        )
        self.config = config
        self.doc_parser = doc_parser or SsmdDocumentParser()
        self.text_preparer = text_preparer or SpokenformTextPreparer()
        self.sentence_segmenter = sentence_segmenter or PhrasplitSentenceSegmenter()
        self.g2p = g2p or KokoroG2PAdapter()
        self.phoneme_processing = phoneme_processing
        self.audio_generation = audio_generation
        self.audio_postprocessing = audio_postprocessing
        self._kokoro: Kokoro | None = None
        self._kokoro_config_key: tuple[object, ...] | None = None
        self._owns_kokoro = False
        self._owns_phoneme_processing = False
        self._owns_audio_generation = False
        self._owns_audio_postprocessing = False
        self._prepared_objects: list[PreparedAudioUnits] = []
        self.linguistic_resources = LinguisticResourcePool()

        self._backend_lock = threading.RLock()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def warmup(self) -> None:
        """Synchronously initialize the configured backend and reuse it thereafter."""
        kokoro, _ = self._ensure_kokoro(self.config)
        kokoro.warmup()

    def close(self) -> None:
        for prepared in tuple(self._prepared_objects):
            prepared.close()
        if self._owns_phoneme_processing:
            self._close_stage(self.phoneme_processing)
            self.phoneme_processing = None
            self._owns_phoneme_processing = False
        if self._owns_audio_generation:
            self._close_stage(self.audio_generation)
            self.audio_generation = None
            self._owns_audio_generation = False
        if self._owns_audio_postprocessing:
            self._close_stage(self.audio_postprocessing)
            self.audio_postprocessing = None
            self._owns_audio_postprocessing = False
        if self._kokoro is not None and self._owns_kokoro:
            self._kokoro.close()
        self._kokoro = None
        self._kokoro_config_key = None

        self.linguistic_resources.clear()

    def _unregister_prepared(self, prepared: PreparedAudioUnits) -> None:
        if prepared in self._prepared_objects:
            self._prepared_objects.remove(prepared)

    def _kokoro_key(self, cfg: PipelineConfig) -> tuple[object, ...]:
        cfg = resolve_model_defaults(cfg)
        model_path = str(cfg.model_path) if cfg.model_path else None
        voices_path = str(cfg.voices_path) if cfg.voices_path else None
        model_config_path = str(cfg.model_config_path) if cfg.model_config_path else None
        release_manifest_path = (
            str(cfg.release_manifest_path) if cfg.release_manifest_path else None
        )
        return (
            model_path,
            voices_path,
            cfg.model_quality,
            cfg.model_source,
            cfg.model_variant,
            model_config_path,
            release_manifest_path,
            cfg.provider,
            _freeze_config_value(cfg.provider_options),
            _freeze_config_value(cfg.session_options),
            _acoustic_tokenizer_key(cfg.tokenizer_config),
            _freeze_config_value(cfg.espeak_config),
            _freeze_config_value(cfg.short_sentence_config),
            _freeze_config_value(cfg.waveform_validation),
            cfg.inference_audio_diagnostics,
            cfg.inference_cache_enabled,
            cfg.inference_cache_max_bytes,
            cfg.allow_experimental_frontend,
        )

    @staticmethod
    def _close_stage(stage: object | None) -> None:
        if stage is None:
            return
        close = getattr(stage, "close", None)
        if callable(close):
            close()

    def _ensure_kokoro(self, cfg: PipelineConfig) -> tuple[Kokoro, bool]:
        """Return a backend, creating it at most once concurrently."""
        with self._backend_lock:
            return self._ensure_kokoro_locked(cfg)

    def _ensure_kokoro_locked(self, cfg: PipelineConfig) -> tuple[Kokoro, bool]:
        cfg = resolve_model_defaults(cfg)
        kokoro_key = self._kokoro_key(cfg)
        if self._kokoro is not None and self._kokoro_config_key == kokoro_key:
            logger.debug(
                "backend.reuse model=%s source=%s quality=%s provider=%s",
                cfg.model_variant,
                cfg.model_source,
                cfg.model_quality,
                cfg.provider,
            )
            self._kokoro._asset_progress = cfg.asset_progress
            return self._kokoro, False
        try:
            from .onnx_backend import Kokoro
        except ModuleNotFoundError as exc:
            if exc.name == "onnxruntime":
                raise RuntimeError(
                    "The default pipeline backend requires ONNX Runtime; install "
                    "pykokoro[cpu] or a platform provider extra."
                ) from exc
            raise

        previous_kokoro = self._kokoro
        previous_owned = self._owns_kokoro
        assert cfg.model_source is not None
        assert cfg.model_variant is not None
        assert cfg.voice is not None
        logger.info(
            "backend.create model=%s source=%s quality=%s provider=%s",
            cfg.model_variant,
            cfg.model_source,
            cfg.model_quality,
            cfg.provider,
        )
        new_kokoro = Kokoro(
            model_path=Path(cfg.model_path) if cfg.model_path else None,
            voices_path=Path(cfg.voices_path) if cfg.voices_path else None,
            model_config_path=Path(cfg.model_config_path) if cfg.model_config_path else None,
            model_quality=cfg.model_quality,
            model_source=cfg.model_source,
            model_variant=cfg.model_variant,
            provider=cfg.provider,
            provider_options=cfg.provider_options,
            session_options=cfg.session_options,
            tokenizer_config=cfg.tokenizer_config,
            espeak_config=cfg.espeak_config,
            short_sentence_config=cfg.short_sentence_config,
            waveform_validation=cfg.waveform_validation,
            inference_audio_diagnostics=cfg.inference_audio_diagnostics,
            inference_cache_enabled=cfg.inference_cache_enabled,
            inference_cache_max_bytes=cfg.inference_cache_max_bytes,
            asset_progress=cfg.asset_progress,
        )
        self._kokoro = new_kokoro
        self._kokoro_config_key = kokoro_key
        self._owns_kokoro = True
        if previous_kokoro is not None:
            logger.debug(
                "backend.replace model=%s source=%s quality=%s provider=%s",
                cfg.model_variant,
                cfg.model_source,
                cfg.model_quality,
                cfg.provider,
            )
        if previous_kokoro is not None and previous_owned:
            try:
                previous_kokoro.close()
            except Exception:
                logger.warning("Failed to close replaced Kokoro backend", exc_info=True)
        logger.info("backend.ready model=%s voice=%s", cfg.model_variant, cfg.voice)
        return new_kokoro, True

    def _resolve_run_config(self, overrides: dict[str, Any]) -> PipelineConfig:
        if not overrides:
            return resolve_model_defaults(self.config)
        overrides = dict(overrides)
        lang = overrides.pop("lang", None)
        has_generation_override = "generation" in overrides
        ssmd_value = overrides.pop("ssmd", None)
        loudness_value = overrides.pop("loudness", None)
        generation = _coerce_generation(
            self.config.generation,
            overrides.pop("generation", None),
        )
        if lang is not None:
            generation = replace(generation, lang=lang)
        if has_generation_override or lang is not None:
            overrides["generation"] = generation
        if ssmd_value is not None:
            overrides["ssmd"] = _coerce_ssmd(self.config.ssmd, ssmd_value)
        if loudness_value is not None:
            overrides["loudness"] = _coerce_loudness(self.config.loudness, loudness_value)
        return resolve_model_defaults(replace(self.config, **overrides))

    def prepare_units(
        self,
        text: str,
        *,
        unit: AudioUnitKind = "paragraph",
        **overrides: Any,
    ) -> PreparedAudioUnits:
        """Prepare a document globally for sequential unit rendering."""
        if unit not in ("paragraph", "sentence"):
            raise ValueError(f"Unsupported audio unit kind: {unit!r}")
        cfg = _copy_config_for_preparation(self._resolve_run_config(overrides))
        if cfg.loudness.target_lufs is not None and unit != "paragraph":
            raise ConfigurationError(
                "Complete-output loudness normalization requires a complete paragraph result; "
                "it cannot be applied to isolated streaming units."
            )
        prepared = self._prepare_document(text, cfg, unit)
        result = PreparedAudioUnits(self, prepared)
        self._prepared_objects.append(result)
        return result

    def prepare_plan_units(
        self,
        plan: UtterancePlan,
        **overrides: Any,
    ) -> PreparedAudioUnits:
        """Prepare renderer units from an existing immutable UtterPlan."""
        assert_renderer_overrides(overrides)
        cfg = _copy_config_for_preparation(self._resolve_run_config(overrides))
        if cfg.generation.is_phonemes:
            raise ConfigurationError(
                "run_plan() requires a text UtterancePlan; phoneme input plans are unsupported"
            )
        unit = self._plan_unit_kind(plan)
        prepared = self._prepare_document_from_plan(plan, cfg, unit)
        result = PreparedAudioUnits(self, prepared)
        self._prepared_objects.append(result)
        return result

    def run_plan(self, plan: UtterancePlan, **overrides: Any) -> AudioResult:
        """Render an existing UtterPlan without reparsing or replanning it."""
        with self.prepare_plan_units(plan, **overrides) as prepared:
            return (
                self._audio_result_from_prepared(prepared._prepared)
                if prepared._prepared
                else AudioResult(
                    audio=np.array([], dtype=np.float32),
                    sample_rate=SAMPLE_RATE,
                    segments=[],
                    phoneme_segments=[],
                    trace=None,
                    document_metadata={},
                    markers=[],
                    word_timings=[],
                    clean_text="",
                    source_text="",
                )
            )

    @staticmethod
    def _plan_unit_kind(plan: UtterancePlan) -> AudioUnitKind:
        kinds = {unit.kind for unit in plan.units}
        if not kinds:
            return "paragraph"
        if len(kinds) != 1 or next(iter(kinds)) not in {"paragraph", "sentence"}:
            raise PlanConsumptionError(
                f"Unsupported or inconsistent plan unit kinds: {sorted(kinds)!r}"
            )
        return cast(AudioUnitKind, next(iter(kinds)))

    def prepare_frontend(
        self,
        text: str,
        *,
        unit: AudioUnitKind = "paragraph",
        **overrides: Any,
    ) -> PreparedFrontend:
        """Prepare lexicon-independent frontend state for repeated rendering."""
        if unit not in ("paragraph", "sentence"):
            raise ValueError(f"Unsupported audio unit kind: {unit!r}")
        cfg = _copy_config_for_preparation(self._resolve_run_config(overrides))
        if cfg.tokenizer_config is not None:
            cfg = replace(
                cfg,
                tokenizer_config=replace(cfg.tokenizer_config, lexicons=None),
            )
        return self._prepare_frontend(text, cfg, unit)

    def render_frontend(
        self,
        frontend: PreparedFrontend,
        *,
        tokenizer_config: TokenizerConfig | Mapping[str, Any] | None = None,
    ) -> AudioResult:
        """Render one lexicon-dependent result from prepared frontend state."""
        if frontend._pipeline is not self:
            raise ValueError("frontend belongs to a different KokoroPipeline")
        frontend._ensure_open()
        rendered_tokenizer_config = (
            frontend._cfg.tokenizer_config
            if tokenizer_config is None
            else _coerce_tokenizer(frontend._cfg.tokenizer_config, tokenizer_config)
        )
        config = replace(frontend._cfg, tokenizer_config=rendered_tokenizer_config)
        prepared = self._render_frontend(frontend, config)
        return self._audio_result_from_prepared(prepared)

    def iter_units(
        self,
        text: str,
        *,
        unit: AudioUnitKind = "paragraph",
        skip_indices: Collection[int] = (),
        **overrides: Any,
    ) -> Iterator[AudioUnitResult]:
        """Yield unit results while owning the prepared document lifecycle."""
        with self.prepare_units(text, unit=unit, **overrides) as prepared:
            yield from prepared.render(skip_indices=skip_indices)

    @staticmethod
    def _linguistic_policy(
        cfg: PipelineConfig,
    ) -> tuple[bool | None, str | None, SpacyModelSize | None]:
        tokenizer_config = cfg.tokenizer_config
        if tokenizer_config is None:
            return None, None, None
        return (
            tokenizer_config.use_spacy,
            tokenizer_config.spacy_model,
            tokenizer_config.spacy_model_size,
        )

    def _analyze_runs(
        self,
        text: str,
        runs: tuple[Any, ...],
        cfg: PipelineConfig,
        trace: Trace,
        pass_name: str,
    ) -> list[PreparedRunAnalysis]:
        use_spacy, model, model_size = self._linguistic_policy(cfg)
        analyses: list[PreparedRunAnalysis] = []
        for run in runs:
            run_text = text[run.char_start : run.char_end]
            analysis = None
            if use_spacy is not False:
                analysis = self.linguistic_resources.analyze(
                    run_text,
                    language=run.language,
                    model=model,
                    model_size=model_size,
                    require=use_spacy is True,
                )
            analyses.append(
                PreparedRunAnalysis(
                    run=run,
                    text=run_text,
                    doc=analysis.doc if analysis else None,
                    annotations=analysis.annotations if analysis else (),
                    model_name=analysis.model_name if analysis else None,
                )
            )
            trace.events.append(
                TraceEvent(
                    stage="linguistics_run",
                    name=pass_name,
                    ms=0.0,
                    details={
                        "language": run.language,
                        "char_start": run.char_start,
                        "char_end": run.char_end,
                        "character_count": len(run_text),
                        "model_name": analysis.model_name if analysis else None,
                        "annotation_count": len(analysis.annotations) if analysis else 0,
                        "fallback": analysis is None,
                    },
                )
            )
        return analyses

    def _prepare_frontend_legacy(
        self, text: str, cfg: PipelineConfig, unit: AudioUnitKind
    ) -> PreparedFrontend:
        language = require_document_language(cfg)
        trace = Trace()
        with trace_timing(trace, "doc", "parse"):
            doc = self.doc_parser.parse(text, cfg, trace)
        trace.warnings.extend(doc.warnings)
        state = LinguisticRequestState()
        doc.linguistic_state = state
        try:
            with trace_timing(trace, "language_plan", "source"):
                state.source_plan = build_language_plan(
                    doc.clean_text, doc.annotation_spans, default_language=language
                )
            with trace_timing(trace, "linguistics", "pass_a"):
                state.source_analysis = self._analyze_runs(
                    doc.clean_text, state.source_plan, cfg, trace, "pass_a"
                )
            with trace_timing(trace, "text_preparation", "prepare"):
                doc = self.text_preparer.prepare(doc, cfg, trace)
            doc.linguistic_state = state
            trace.warnings.extend(doc.preparation.warnings if doc.preparation else ())
            with trace_timing(trace, "language_plan", "prepared"):
                state.prepared_plan = build_language_plan(
                    doc.clean_text, doc.annotation_spans, default_language=language
                )
            with trace_timing(trace, "linguistics", "pass_b"):
                state.prepared_analysis = self._analyze_runs(
                    doc.clean_text, state.prepared_plan, cfg, trace, "pass_b"
                )
            state.release_source_docs()
            with trace_timing(trace, "segmentation", "split"):
                segments = self.sentence_segmenter.split(doc, cfg, trace)
        except Exception:
            state.release_docs()
            doc.linguistic_state = None
            raise
        if not segments and doc.clean_text:
            segments = [
                Segment(
                    id="p0_s0_c0_seg0",
                    text=doc.clean_text,
                    char_start=0,
                    char_end=len(doc.clean_text),
                    paragraph_idx=0,
                    sentence_idx=0,
                    clause_idx=0,
                )
            ]
        doc.segments = segments
        self._apply_post_segmentation_pauses(doc, segments, cfg)
        return PreparedFrontend(self, cfg, trace, doc, segments, state, unit)

    @staticmethod
    def _compile_plan(text: str, cfg: PipelineConfig, unit: AudioUnitKind) -> UtterancePlan:
        planner_config = planner_config_from_pipeline(cfg, unit=unit)
        return UtterancePlanner(planner_config).plan(text, config=planner_config, unit=unit)

    def _prepare_frontend(
        self, text: str, cfg: PipelineConfig, unit: AudioUnitKind
    ) -> PreparedFrontend:
        if self._legacy_frontend_compat:
            return self._prepare_frontend_legacy(text, cfg, unit)
        plan = self._compile_plan(text, cfg, unit)
        adapted = adapt_plan(plan)
        trace = Trace()
        trace.events.append(
            TraceEvent(
                stage="planning",
                name="compile",
                ms=0.0,
                details={
                    "plan_id": plan.plan_id,
                    "producer": dict(plan.producer),
                    "schema_version": plan.schema_version,
                },
            )
        )
        trace.warnings.extend(adapted.document.warnings)
        state = LinguisticRequestState()
        doc = adapted.document
        segments = [item.to_segment() for item in adapted.segments]
        doc.segments = segments
        return PreparedFrontend(self, cfg, trace, doc, segments, state, unit, plan)

    def _prepare_document_from_plan(
        self, plan: UtterancePlan, cfg: PipelineConfig, unit: AudioUnitKind
    ) -> _PreparedDocument:
        adapted = adapt_plan(plan)
        trace = Trace()
        trace.events.append(
            TraceEvent(
                stage="planning",
                name="consume",
                ms=0.0,
                details={
                    "plan_id": plan.plan_id,
                    "producer": dict(plan.producer),
                    "schema_version": plan.schema_version,
                },
            )
        )
        trace.warnings.extend(adapted.document.warnings)
        doc = adapted.document
        segments = [item.to_segment() for item in adapted.segments]
        doc.segments = segments
        with trace_timing(trace, "g2p", "phonemize") if segments else _nullcontext():
            logger.debug("Phonemizing %d plan segments", len(segments))
            phoneme_segments = self.g2p.phonemize(segments, doc, cfg, trace) if segments else []
        with trace_timing(trace, "runtime", "resolve_stages"):
            phoneme_processor, audio_generator, audio_postprocessor = self._resolve_stages(cfg)
        with (
            trace_timing(trace, "phoneme_processing", "preprocess")
            if phoneme_segments
            else _nullcontext()
        ):
            if phoneme_segments:
                phoneme_segments = phoneme_processor.process(phoneme_segments, cfg, trace)
        if phoneme_segments:
            apply_emphasis_policy(phoneme_segments, cfg, trace)
        groups = self._build_unit_groups_from_plan(adapted, doc, segments, phoneme_segments, cfg)
        return _PreparedDocument(
            cfg=cfg,
            unit_kind=unit,
            trace=trace,
            doc=doc,
            segments=segments,
            phoneme_segments=phoneme_segments,
            groups=groups,
            phoneme_processor=phoneme_processor,
            audio_generator=audio_generator,
            audio_postprocessor=audio_postprocessor,
        )

    def _prepare_document_legacy(
        self, text: str, cfg: PipelineConfig, unit: AudioUnitKind
    ) -> _PreparedDocument:
        frontend = self._prepare_frontend_legacy(text, cfg, unit)
        doc = frontend._doc
        state = frontend._state
        segments = list(frontend._segments)
        assert doc is not None
        try:
            with trace_timing(frontend._trace, "g2p", "phonemize"):
                phoneme_segments = self.g2p.phonemize(segments, doc, cfg, frontend._trace)
        finally:
            state.release_docs()
            doc.linguistic_state = None
        with trace_timing(frontend._trace, "runtime", "resolve_stages"):
            phoneme_processor, audio_generator, audio_postprocessor = self._resolve_stages(cfg)
        with trace_timing(frontend._trace, "phoneme_processing", "preprocess"):
            phoneme_segments = phoneme_processor.process(phoneme_segments, cfg, frontend._trace)
        apply_emphasis_policy(phoneme_segments, cfg, frontend._trace)
        groups = self._build_unit_groups(doc, segments, phoneme_segments, cfg, unit)
        frontend._closed = True
        return _PreparedDocument(
            cfg=cfg,
            unit_kind=unit,
            trace=frontend._trace,
            doc=doc,
            segments=segments,
            phoneme_segments=phoneme_segments,
            groups=groups,
            phoneme_processor=phoneme_processor,
            audio_generator=audio_generator,
            audio_postprocessor=audio_postprocessor,
        )

    def _prepare_document(
        self, text: str, cfg: PipelineConfig, unit: AudioUnitKind
    ) -> _PreparedDocument:
        if self._legacy_frontend_compat:
            return self._prepare_document_legacy(text, cfg, unit)
        plan = self._compile_plan(text, cfg, unit)
        return self._prepare_document_from_plan(plan, cfg, unit)

    @staticmethod
    def _copy_frontend_trace(trace: Trace) -> Trace:
        return Trace(
            warnings=list(trace.warnings),
            events=list(trace.events),
            prosody=deepcopy(trace.prosody),
            model=deepcopy(trace.model),
            counters=dict(trace.counters),
        )

    def _render_frontend(
        self, frontend: PreparedFrontend, cfg: PipelineConfig
    ) -> _PreparedDocument:
        frontend._ensure_open()
        if frontend._plan is not None:
            return self._prepare_document_from_plan(frontend._plan, cfg, frontend._unit_kind)
        doc = frontend._doc
        doc.linguistic_state = frontend._state
        trace = self._copy_frontend_trace(frontend._trace)
        try:
            with trace_timing(trace, "g2p", "phonemize"):
                logger.debug("Phonemizing %d segments", len(frontend._segments))
                phoneme_segments = self.g2p.phonemize(frontend._segments, doc, cfg, trace)
        finally:
            frontend._state.release_docs()
            doc.linguistic_state = None
        with trace_timing(trace, "runtime", "resolve_stages"):
            phoneme_processor, audio_generator, audio_postprocessor = self._resolve_stages(cfg)
        with trace_timing(trace, "phoneme_processing", "preprocess"):
            logger.debug("Preprocessing %d phoneme segments", len(phoneme_segments))
            phoneme_segments = phoneme_processor.process(phoneme_segments, cfg, trace)
        apply_emphasis_policy(phoneme_segments, cfg, trace)
        segments = list(frontend._segments)
        groups = self._build_unit_groups(doc, segments, phoneme_segments, cfg, frontend._unit_kind)
        return _PreparedDocument(
            cfg=cfg,
            unit_kind=frontend._unit_kind,
            trace=trace,
            doc=doc,
            segments=segments,
            phoneme_segments=phoneme_segments,
            groups=groups,
            phoneme_processor=phoneme_processor,
            audio_generator=audio_generator,
            audio_postprocessor=audio_postprocessor,
        )

    @staticmethod
    def _apply_post_segmentation_pauses(
        doc: Any, segments: list[Segment], cfg: PipelineConfig
    ) -> None:
        defaults = doc.metadata.get("ssmd_pause_defaults")
        if not isinstance(defaults, dict):
            defaults = {}
        existing = {boundary.pos for boundary in doc.boundary_events if boundary.kind == "pause"}
        sentence_duration = defaults.get("sentence")
        if sentence_duration is None and cfg.generation.pause_mode == "auto":
            sentence_duration = cfg.generation.pause_sentence
        paragraph_duration = defaults.get("paragraph")
        if paragraph_duration is None:
            paragraph_duration = cfg.generation.pause_paragraph
        for index, boundary in enumerate(doc.boundary_events):
            if boundary.kind != "pause" or boundary.attrs.get("strength") != "p":
                continue
            if boundary.duration_s is None and paragraph_duration is not None:
                doc.boundary_events[index] = replace(
                    boundary,
                    duration_s=paragraph_duration,
                    attrs={
                        "source": "header_default"
                        if defaults.get("paragraph") is not None
                        else "pipeline_default",
                        **boundary.attrs,
                    },
                )
        previous_paragraph_segment: Segment | None = None
        for segment in segments:
            if (
                previous_paragraph_segment is not None
                and previous_paragraph_segment.paragraph_idx != segment.paragraph_idx
            ):
                position = max(0, previous_paragraph_segment.char_end - 1)
                if position not in existing and paragraph_duration is not None:
                    doc.boundary_events.append(
                        BoundaryEvent(
                            pos=position,
                            kind="pause",
                            duration_s=float(paragraph_duration),
                            attrs={
                                "source": "header_default"
                                if defaults.get("paragraph") is not None
                                else "pipeline_default",
                                "strength": "p",
                            },
                        )
                    )
                    existing.add(position)
            previous_paragraph_segment = segment
        voice_duration = defaults.get("voice_change")
        if voice_duration is not None:
            previous_voice: str | None = None
            previous_segment: Segment | None = None
            for segment in segments:
                voice = KokoroPipeline._segment_voice(doc, segment)
                if previous_segment is not None and voice != previous_voice:
                    position = max(0, previous_segment.char_end - 1)
                    if position not in existing:
                        doc.boundary_events.append(
                            BoundaryEvent(
                                pos=position,
                                kind="pause",
                                duration_s=float(voice_duration),
                                attrs={"source": "header_default", "kind": "voice_change"},
                            )
                        )
                        existing.add(position)
                previous_voice = voice
                previous_segment = segment
        last: Segment | None = None
        for segment in segments:
            if (
                last is not None
                and last.paragraph_idx == segment.paragraph_idx
                and last.sentence_idx != segment.sentence_idx
                and sentence_duration is not None
            ):
                position = max(0, last.char_end - 1)
                if position not in existing:
                    doc.boundary_events.append(
                        BoundaryEvent(
                            pos=position,
                            kind="pause",
                            duration_s=sentence_duration,
                            attrs={
                                "source": "header_default"
                                if defaults.get("sentence") is not None
                                else "pipeline_default",
                                "strength": "s",
                            },
                        )
                    )
                    existing.add(position)
            last = segment

    @staticmethod
    def _segment_voice(doc: Any, segment: Segment) -> str | None:
        candidates = [
            span
            for span in doc.annotation_spans
            if span.char_start <= segment.char_start < span.char_end
        ]
        if not candidates:
            return None
        selected = min(candidates, key=lambda span: span.char_end - span.char_start)
        return selected.attrs.get("voice_name") or selected.attrs.get("voice")

    def _resolve_stages(
        self, cfg: PipelineConfig
    ) -> tuple[PhonemeProcessor, AudioGeneratorStage, AudioPostprocessor]:
        phoneme_processor = self.phoneme_processing
        audio_generator = self.audio_generation
        audio_postprocessor = self.audio_postprocessing

        needs_default_backend = (
            phoneme_processor is None
            or audio_generator is None
            or audio_postprocessor is None
            or self._owns_phoneme_processing
            or self._owns_audio_generation
            or self._owns_audio_postprocessing
        )

        if needs_default_backend:
            kokoro, kokoro_changed = self._ensure_kokoro(cfg)
            (
                onnx_phoneme_processor,
                onnx_audio_generation,
                onnx_audio_postprocessing,
            ) = _load_default_onnx_adapters()
            if phoneme_processor is None or (kokoro_changed and self._owns_phoneme_processing):
                phoneme_processor = onnx_phoneme_processor(
                    kokoro,
                    context_phonemizer=getattr(self.g2p, "phonemize_context", None),
                )
                if self.phoneme_processing is None or self._owns_phoneme_processing:
                    self.phoneme_processing = phoneme_processor
                    self._owns_phoneme_processing = True
            if audio_generator is None or (kokoro_changed and self._owns_audio_generation):
                audio_generator = onnx_audio_generation(kokoro)
                if self.audio_generation is None or self._owns_audio_generation:
                    self.audio_generation = audio_generator
                    self._owns_audio_generation = True
            if audio_postprocessor is None or (kokoro_changed and self._owns_audio_postprocessing):
                audio_postprocessor = onnx_audio_postprocessing(kokoro)
                if self.audio_postprocessing is None or self._owns_audio_postprocessing:
                    self.audio_postprocessing = audio_postprocessor
                    self._owns_audio_postprocessing = True

        assert phoneme_processor is not None
        assert audio_generator is not None
        assert audio_postprocessor is not None
        return phoneme_processor, audio_generator, audio_postprocessor

    def _build_unit_groups_from_plan(
        self,
        adapted: Any,
        doc: Any,
        segments: list[Segment],
        phoneme_segments: list[PhonemeSegment],
        cfg: PipelineConfig,
    ) -> tuple[_PreparedUnitGroup, ...]:
        segment_by_id = {segment.id: segment for segment in segments}
        phoneme_by_segment: dict[str, list[int]] = {}
        for index, phoneme in enumerate(phoneme_segments):
            phoneme_by_segment.setdefault(phoneme.segment_id, []).append(index)
        markers = {marker.id: marker for marker in adapted.plan.markers}
        groups: list[_PreparedUnitGroup] = []
        previous_end = -1
        for index, unit in enumerate(adapted.plan.units):
            if unit.index != index:
                raise PlanConsumptionError(f"Plan unit order is invalid at {unit.id!r}")
            unit_segments = []
            for segment_id in unit.segment_ids:
                if segment_id not in segment_by_id:
                    raise PlanConsumptionError(
                        f"Plan unit {unit.id!r} references unknown segment {segment_id!r}"
                    )
                unit_segments.append(segment_by_id[segment_id])
            positions = [
                position
                for segment_id in unit.segment_ids
                for position in phoneme_by_segment.get(segment_id, ())
            ]
            if positions and positions != list(range(min(positions), max(positions) + 1)):
                raise PlanConsumptionError(
                    f"Plan unit {unit.id!r} has non-contiguous phoneme segments"
                )
            if positions and min(positions) <= previous_end:
                raise PlanConsumptionError(f"Plan units overlap or are out of order at {unit.id!r}")
            if not positions and unit_segments:
                raise PlanConsumptionError(f"Plan unit {unit.id!r} has no phoneme output")
            marker_events = []
            for marker_id in unit.marker_ids:
                marker = markers.get(marker_id)
                if marker is None:
                    raise PlanConsumptionError(
                        f"Plan unit {unit.id!r} references unknown marker {marker_id!r}"
                    )
                marker_events.append(
                    BoundaryEvent(
                        pos=marker.spoken_position,
                        kind="marker",
                        attrs={"marker": marker.name},
                    )
                )
            if positions:
                start, end = min(positions), max(positions) + 1
            else:
                start = end = 0
            paragraph = unit_segments[0].paragraph_idx if unit_segments else 0
            sentence = (
                unit_segments[0].sentence_idx if unit.kind == "sentence" and unit_segments else None
            )
            text = doc.clean_text[unit.spoken_start : unit.spoken_end]
            descriptor = AudioUnitDescriptor(
                index=index,
                paragraph_idx=paragraph or 0,
                char_start=unit.spoken_start,
                char_end=unit.spoken_end,
                text=text,
                text_hash=_unit_text_hash(
                    paragraph or 0,
                    unit.spoken_start,
                    unit.spoken_end,
                    text,
                    phoneme_segments[start:end],
                    cfg,
                    marker_events,
                ),
                segment_ids=tuple(unit.segment_ids),
                phoneme_segment_ids=tuple(segment.id for segment in phoneme_segments[start:end]),
                unit_kind=unit.kind,
                sentence_idx=sentence,
                marker_names=tuple(event.attrs["marker"] for event in marker_events),
            )
            groups.append(_PreparedUnitGroup(descriptor, start, end, tuple(marker_events)))
            previous_end = end - 1
        return tuple(groups)

    def _build_unit_groups(
        self,
        doc: Any,
        segments: list[Segment],
        phoneme_segments: list[PhonemeSegment],
        cfg: PipelineConfig,
        unit: AudioUnitKind,
    ) -> tuple[_PreparedUnitGroup, ...]:
        if not phoneme_segments:
            return ()

        groups_data: list[tuple[tuple[object, ...], int, int | None, int, int]] = []
        closed_keys: set[tuple[object, ...]] = set()
        current_key: tuple[object, ...] | None = None
        current_paragraph: int | None = None
        for phoneme_index, segment in enumerate(phoneme_segments):
            paragraph = segment.paragraph_idx
            if paragraph is None:
                paragraph = current_paragraph if current_paragraph is not None else 0
            key: tuple[object, ...]
            if unit == "paragraph":
                key = ("paragraph", paragraph)
                sentence = None
            elif segment.sentence_idx is not None:
                sentence = segment.sentence_idx
                key = ("sentence", paragraph, sentence)
            else:
                sentence = None
                key = ("sentence-fallback", paragraph, segment.segment_id)
            if current_key != key:
                if key in closed_keys:
                    raise RuntimeError(
                        f"Prepared {unit} {key!r} is disjoint in phoneme segment order"
                    )
                if current_key is not None:
                    closed_keys.add(current_key)
                groups_data.append((key, paragraph, sentence, phoneme_index, phoneme_index + 1))
                current_key = key
                current_paragraph = paragraph
            else:
                group_key, group_paragraph, group_sentence, start, _ = groups_data[-1]
                groups_data[-1] = (
                    group_key,
                    group_paragraph,
                    group_sentence,
                    start,
                    phoneme_index + 1,
                )

        segment_by_id = {segment.id: segment for segment in segments}
        groups: list[_PreparedUnitGroup] = []
        for index, (_, paragraph, sentence, start, end) in enumerate(groups_data):
            group_phonemes = phoneme_segments[start:end]
            segment_ids = tuple(dict.fromkeys(segment.segment_id for segment in group_phonemes))
            group_segments = [
                segment_by_id[segment_id]
                for segment_id in segment_ids
                if segment_id in segment_by_id
            ]
            if not group_segments:
                char_start = min(segment.char_start for segment in group_phonemes)
                char_end = max(segment.char_end for segment in group_phonemes)
                spoken_text = " ".join(
                    segment.text.strip() for segment in group_phonemes if segment.text.strip()
                )
            else:
                char_start = min(segment.char_start for segment in group_segments)
                char_end = max(segment.char_end for segment in group_segments)
                spoken_text = " ".join(
                    segment.text.strip() for segment in group_segments if segment.text.strip()
                )
            descriptor = AudioUnitDescriptor(
                index=index,
                paragraph_idx=paragraph,
                char_start=char_start,
                char_end=char_end,
                text=spoken_text,
                text_hash=_unit_text_hash(
                    paragraph,
                    char_start,
                    char_end,
                    spoken_text,
                    group_phonemes,
                    cfg,
                    (),
                ),
                segment_ids=segment_ids,
                phoneme_segment_ids=tuple(segment.id for segment in group_phonemes),
                unit_kind=unit,
                sentence_idx=sentence,
            )
            groups.append(_PreparedUnitGroup(descriptor, start, end, ()))

        marker_groups: list[list[Any]] = [[] for _ in groups]
        for boundary in doc.boundary_events:
            if getattr(boundary, "kind", None) != "marker":
                continue
            owner = _owner_for_boundary(boundary.pos, groups, phoneme_segments, len(doc.clean_text))
            if owner is not None:
                marker_groups[owner].append(boundary)

        finalized: list[_PreparedUnitGroup] = []
        for group, marker_events in zip(groups, marker_groups, strict=True):
            marker_names = tuple(
                str(event.attrs["marker"]) for event in marker_events if event.attrs.get("marker")
            )
            descriptor = replace(
                group.descriptor,
                marker_names=marker_names,
                text_hash=_unit_text_hash(
                    group.descriptor.paragraph_idx,
                    group.descriptor.char_start,
                    group.descriptor.char_end,
                    group.descriptor.text,
                    phoneme_segments[group.phoneme_start : group.phoneme_end],
                    cfg,
                    marker_events,
                ),
            )
            finalized.append(
                _PreparedUnitGroup(
                    descriptor,
                    group.phoneme_start,
                    group.phoneme_end,
                    tuple(marker_events),
                )
            )
        return tuple(finalized)

    def _render_prepared_unit(self, prepared: _PreparedDocument, index: int) -> AudioUnitResult:
        group = prepared.groups[index]
        source = prepared.phoneme_segments[group.phoneme_start : group.phoneme_end]
        generated: list[PhonemeSegment] = source
        try:
            with trace_timing(prepared.trace, "audio_generation", "generate"):
                logger.debug(
                    "Generating audio for unit %d (%d phoneme segments)",
                    index,
                    len(source),
                )
                generated = prepared.audio_generator.generate(source, prepared.cfg, prepared.trace)
            with trace_timing(prepared.trace, "audio_postprocessing", "postprocess"):
                audio = prepared.audio_postprocessor.postprocess(
                    generated, prepared.cfg, prepared.trace
                )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "unit.finish index=%d unit_kind=%s samples=%d sample_rate=%d segments=%d",
                    index,
                    group.descriptor.unit_kind,
                    int(np.asarray(audio).size),
                    SAMPLE_RATE,
                    len(generated),
                )
            markers = _collect_marker_offsets(
                list(group.marker_events),
                generated,
                base_sample_offset=0,
                descriptor=group.descriptor,
            )
            segment_by_id = {segment.id: segment for segment in prepared.segments}
            unit_segments = [
                segment_by_id[segment_id]
                for segment_id in group.descriptor.segment_ids
                if segment_id in segment_by_id
            ]
            prepared.trace.events.append(
                TraceEvent(
                    stage="unit",
                    name="render",
                    ms=0.0,
                    details={
                        "unit_index": index,
                        "unit_kind": group.descriptor.unit_kind,
                        "sentence_idx": group.descriptor.sentence_idx,
                        "paragraph_idx": group.descriptor.paragraph_idx,
                        "phoneme_segment_count": len(generated),
                        "character_count": len(group.descriptor.text),
                    },
                )
            )
            result = AudioUnitResult(
                descriptor=group.descriptor,
                audio=audio,
                sample_rate=SAMPLE_RATE,
                segments=unit_segments,
                phoneme_segments=generated,
                markers=markers,
                trace=prepared.trace if prepared.cfg.return_trace else None,
                document_metadata=_copy_metadata_value(
                    {
                        "title": prepared.doc.header.get("title"),
                        "voice_bindings": prepared.doc.header.get("voice_bindings", {}),
                        "pause_defaults": prepared.doc.header.get("pause_defaults", {}),
                        **prepared.doc.metadata,
                    }
                ),
                word_timings=_collect_unit_word_timings(generated),
            )
            if not prepared.cfg.retain_segment_audio:
                result.release_segment_audio()
            return result
        except Exception:
            seen: set[int] = set()
            for segment in source + (generated if generated is not source else []):
                if id(segment) in seen:
                    continue
                seen.add(id(segment))
                segment.raw_audio = None
                segment.processed_audio = None
            raise

    def _audio_result_from_prepared(self, prepared: _PreparedDocument) -> AudioResult:
        prepared_units = PreparedAudioUnits(self, prepared)
        self._prepared_objects.append(prepared_units)
        cfg = prepared.cfg
        final_audio: list[Any] = []
        markers: list[dict[str, Any]] = []
        word_timings: list[Any] = []
        retained_phonemes: list[PhonemeSegment] = []
        base_offset = 0
        try:
            for unit_result in prepared_units.render():
                final_audio.append(unit_result.audio)
                markers.extend(
                    {
                        "name": marker["name"],
                        "char_offset": marker["char_offset"],
                        "sample_offset": marker["sample_offset"] + base_offset,
                    }
                    for marker in unit_result.markers
                )
                word_timings.extend(
                    replace(
                        timing,
                        start_sample=timing.start_sample + base_offset,
                        end_sample=timing.end_sample + base_offset,
                    )
                    for timing in unit_result.word_timings
                )
                if cfg.retain_segment_audio:
                    retained_phonemes.extend(
                        _copy_phoneme_segment(segment) for segment in unit_result.phoneme_segments
                    )
                base_offset += len(unit_result.audio)
            source_segments = list(prepared.segments)
            source_phonemes = (
                retained_phonemes if cfg.retain_segment_audio else list(prepared.phoneme_segments)
            )
            audio = np.concatenate(final_audio) if final_audio else np.array([], dtype=np.float32)
            metadata = dict(prepared_units.document_metadata)
            compatibility_job = waveform_to_audio_job(
                audio,
                sample_rate=SAMPLE_RATE,
                markers=markers,
                word_timings=word_timings,
                loudness=cfg.loudness,
                producer={"name": "pykokoro"},
                source={"clean_text": prepared_units.clean_text},
            )
            audio = compose_audio_job(compatibility_job).audio
            return AudioResult(
                audio=audio,
                sample_rate=SAMPLE_RATE,
                segments=source_segments,
                phoneme_segments=source_phonemes,
                trace=prepared.trace if cfg.return_trace else None,
                document_metadata=metadata,
                markers=markers,
                word_timings=word_timings,
                clean_text=prepared_units.clean_text,
                source_text=prepared_units.source_text,
            )
        finally:
            prepared_units.close()

    def _prepare_audio_job(self, text: str, **overrides: Any) -> _PreparedAudioJobContext:
        cfg = _copy_config_for_preparation(self._resolve_run_config(overrides))
        requested_retain_segment_audio = cfg.retain_segment_audio
        cfg = replace(cfg, retain_segment_audio=True)
        prepared = self._prepare_document(text, cfg, "paragraph")
        marker_positions: dict[str, int] = {}
        boundaries: list[BoundaryEvent] = []
        for group in prepared.groups:
            boundaries.extend(group.marker_events)
        for index, boundary in enumerate(boundaries):
            if boundary.kind == "marker" and boundary.attrs.get("marker"):
                marker_positions[f"marker:{index}"] = boundary.pos
        try:
            rendered_audio: list[np.ndarray] = []
            for index in range(len(prepared.groups)):
                rendered = self._render_prepared_unit(prepared, index)
                rendered_audio.append(np.asarray(rendered.audio, dtype=np.float32))
            document_metadata = {
                "title": prepared.doc.header.get("title"),
                "voice_bindings": prepared.doc.header.get("voice_bindings", {}),
                "pause_defaults": prepared.doc.header.get("pause_defaults", {}),
                **prepared.doc.metadata,
            }
            segments = list(prepared.segments)
            phoneme_segments = [
                _copy_phoneme_segment(segment) for segment in prepared.phoneme_segments
            ]
            source_metadata = {
                "clean_text": prepared.doc.clean_text,
                "source_text": prepared.doc.structural_clean_text,
                "metadata": _copy_metadata_value(document_metadata),
            }
            has_segment_audio = any(
                segment.processed_audio is not None or segment.raw_audio is not None
                for segment in prepared.phoneme_segments
            )
            if has_segment_audio:
                job = rendered_segments_to_audio_job(
                    prepared.phoneme_segments,
                    boundaries=boundaries,
                    sample_rate=SAMPLE_RATE,
                    loudness=cfg.loudness,
                    producer={"name": "pykokoro"},
                    source=source_metadata,
                )
            else:
                job = waveform_to_audio_job(
                    np.concatenate(rendered_audio)
                    if rendered_audio
                    else np.zeros(0, dtype=np.float32),
                    sample_rate=SAMPLE_RATE,
                    loudness=cfg.loudness,
                    producer={"name": "pykokoro"},
                    source=source_metadata,
                )
            return _PreparedAudioJobContext(
                job=job,
                cfg=replace(cfg, retain_segment_audio=requested_retain_segment_audio),
                segments=segments,
                phoneme_segments=phoneme_segments,
                trace=prepared.trace,
                document_metadata=_copy_metadata_value(document_metadata),
                clean_text=prepared.doc.clean_text,
                source_text=prepared.doc.structural_clean_text,
                marker_positions=marker_positions,
            )
        finally:
            for segment in prepared.phoneme_segments:
                segment.raw_audio = None
                segment.processed_audio = None
            prepared.segments.clear()

    def to_audio_job(self, text: str, **overrides: Any) -> AudioJob:
        """Prepare and render text as an explicit AudioCompose job."""
        return self._prepare_audio_job(text, **overrides).job


    def to_audio_job_from_plan(self, plan: UtterancePlan, **overrides: Any) -> AudioJob:
        """Create an AudioJob from an existing UtterancePlan without replanning.

        This method validates the plan, rejects planning-only overrides,
        reuses the prepare_plan_units path, and preserves plan segment IDs
        and markers in the AudioJob provenance.

        The input plan is never mutated, and UtterancePlanner.plan() is never called.
        """
        assert_renderer_overrides(overrides)
        cfg = _copy_config_for_preparation(self._resolve_run_config(overrides))
        if cfg.generation.is_phonemes:
            raise ConfigurationError(
                "to_audio_job_from_plan() requires a text UtterancePlan; phoneme input plans are unsupported"
            )
        unit = self._plan_unit_kind(plan)
        prepared = self._prepare_document_from_plan(plan, cfg, unit)
        rendered_audio: list[np.ndarray] = []
        for index in range(len(prepared.groups)):
            rendered = self._render_prepared_unit(prepared, index)
            rendered_audio.append(np.asarray(rendered.audio, dtype=np.float32))
        document_metadata = {
            "title": prepared.doc.header.get("title"),
            "voice_bindings": prepared.doc.header.get("voice_bindings", {}),
            "pause_defaults": prepared.doc.header.get("pause_defaults", {}),
            **prepared.doc.metadata,
        }
        boundaries: list[BoundaryEvent] = []
        for group in prepared.groups:
            boundaries.extend(group.marker_events)
        marker_positions: dict[str, int] = {}
        for index, boundary in enumerate(boundaries):
            if boundary.kind == "marker" and boundary.attrs.get("marker"):
                marker_positions[f"marker:{index}"] = boundary.pos
        source_metadata = {
            "clean_text": prepared.doc.clean_text,
            "source_text": prepared.doc.structural_clean_text,
            "metadata": _copy_metadata_value(document_metadata),
        }
        has_segment_audio = any(
            segment.processed_audio is not None or segment.raw_audio is not None
            for segment in prepared.phoneme_segments
        )
        if has_segment_audio:
            job = rendered_segments_to_audio_job(
                prepared.phoneme_segments,
                boundaries=boundaries,
                sample_rate=SAMPLE_RATE,
                loudness=cfg.loudness,
                producer={"name": "pykokoro"},
                source=source_metadata,
            )
        else:
            job = waveform_to_audio_job(
                np.concatenate(rendered_audio)
                if rendered_audio
                else np.zeros(0, dtype=np.float32),
                sample_rate=SAMPLE_RATE,
                loudness=cfg.loudness,
                producer={"name": "pykokoro"},
                source=source_metadata,
            )
        # Add plan provenance to the job
        from dataclasses import replace as _dc_replace

        plan_provenance = {
            "plan_id": plan.plan_id,
            "plan_schema_version": plan.schema_version,
            "plan_producer": dict(plan.producer),
            "segment_ids": tuple(segment.id for segment in plan.segments),
            "unit_ids": tuple(unit.id for unit in plan.units),
        }
        existing_provenance = dict(job.provenance) if job.provenance else {}
        existing_provenance["utterplan"] = plan_provenance
        job = _dc_replace(job, provenance=existing_provenance)
        # Clean up segment audio
        for segment in prepared.phoneme_segments:
            segment.raw_audio = None
            segment.processed_audio = None
        prepared.segments.clear()
        return job
    def run(self, text: str, **overrides: Any) -> AudioResult:
        """Render text by preparing one job and composing it once with AudioCompose."""
        context = self._prepare_audio_job(text, **overrides)
        composed = compose_audio_job(context.job)
        trace = context.trace
        trace.events.append(
            TraceEvent(
                stage="composition",
                name="audiocompose",
                ms=0.0,
                details={
                    "sample_rate": composed.sample_rate,
                    "item_count": len(composed.items),
                    "marker_count": len(composed.markers),
                    "span_count": len(composed.spans),
                    "diagnostics": [diagnostic.code for diagnostic in composed.diagnostics],
                    **dict(composed.provenance),
                },
            )
        )
        marker_positions = context.marker_positions
        markers = [
            {
                "name": marker.name,
                "char_offset": marker_positions.get(marker.id),
                "sample_offset": marker.sample_offset,
            }
            for marker in composed.markers
        ]
        spans = {span.id: span for span in composed.spans if span.id is not None}
        word_timings: list[Any] = []
        for segment in context.phoneme_segments:
            for index, timing in enumerate(segment.word_timings):
                span = spans.get(f"word:{segment.id}:{index}")
                if span is None:
                    word_timings.append(timing)
                else:
                    word_timings.append(
                        replace(
                            timing,
                            start_sample=span.sample_start,
                            end_sample=span.sample_end,
                        )
                    )
        if not context.cfg.retain_segment_audio:
            for segment in context.phoneme_segments:
                segment.raw_audio = None
                segment.processed_audio = None
        return AudioResult(
            audio=np.asarray(composed.audio, dtype=np.float32),
            sample_rate=composed.sample_rate,
            segments=context.segments,
            phoneme_segments=context.phoneme_segments,
            trace=trace if context.cfg.return_trace else None,
            document_metadata=context.document_metadata,
            markers=markers,
            word_timings=word_timings,
            clean_text=context.clean_text,
            source_text=context.source_text,
        )

    def play_streaming(
        self,
        text: str,
        *,
        unit: AudioUnitKind = "sentence",
        device: int | str | None = None,
        queue_size: int = 2,
        **overrides: Any,
    ) -> None:
        """Generate and play selected units through one persistent output stream."""
        config = self._resolve_run_config(dict(overrides))
        if config.loudness.target_lufs is not None:
            raise ConfigurationError(
                "Complete-output loudness normalization requires the complete waveform and "
                "cannot be used with play_streaming(). Render the complete AudioResult first."
            )
        from .playback import play_prepared_units

        with self.prepare_units(text, unit=unit, **overrides) as prepared:
            play_prepared_units(prepared, device=device, queue_size=queue_size)

    def __call__(self, text: str, **overrides: Any) -> AudioResult:
        return self.run(text, **overrides)


def _copy_metadata_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _copy_metadata_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_metadata_value(item) for item in value]
    if isinstance(value, tuple):
        return [_copy_metadata_value(item) for item in value]
    return value


def _collect_marker_offsets(
    boundaries: list[Any],
    phoneme_segments: list[Any],
    *,
    base_sample_offset: int = 0,
    descriptor: AudioUnitDescriptor | None = None,
) -> list[dict[str, Any]]:
    """Map marker boundaries to deterministic local or aggregate sample offsets."""

    marker_boundaries = [
        boundary for boundary in boundaries if getattr(boundary, "kind", None) == "marker"
    ]
    if not marker_boundaries:
        return []
    offsets: list[dict[str, Any]] = []
    for boundary in marker_boundaries:
        sample_offset = 0
        for segment in phoneme_segments:
            segment_audio = getattr(segment, "processed_audio", None)
            segment_samples = len(segment_audio) if segment_audio is not None else 0
            if boundary.pos <= getattr(segment, "char_start", 0):
                sample_offset += round(getattr(segment, "pause_before", 0.0) * SAMPLE_RATE)
                break
            sample_offset += round(getattr(segment, "pause_before", 0.0) * SAMPLE_RATE)
            sample_offset += segment_samples
            if boundary.pos <= getattr(segment, "char_end", 0):
                break
            sample_offset += round(getattr(segment, "pause_after", 0.0) * SAMPLE_RATE)
        marker = boundary.attrs.get("marker")
        if marker:
            offsets.append(
                {
                    "name": marker,
                    "char_offset": boundary.pos,
                    "sample_offset": base_sample_offset + sample_offset,
                    **(
                        {
                            "paragraph_idx": descriptor.paragraph_idx,
                            "unit_kind": descriptor.unit_kind,
                            "sentence_idx": descriptor.sentence_idx,
                            "unit_index": descriptor.index,
                        }
                        if descriptor is not None
                        else {}
                    ),
                }
            )
    return offsets


def _owner_for_boundary(
    position: int,
    groups: list[_PreparedUnitGroup],
    phoneme_segments: list[PhonemeSegment],
    doc_end: int,
) -> int | None:
    """Assign a marker to one group using clean-text positions and group order."""
    if not groups:
        return None
    for index, group in enumerate(groups):
        values = phoneme_segments[group.phoneme_start : group.phoneme_end]
        start = min(segment.char_start for segment in values)
        end = max(segment.char_end for segment in values)
        if start <= position < end:
            return index
        if position < start:
            return index
    return len(groups) - 1 if position <= doc_end else None


def _unit_text_hash(
    paragraph_idx: int,
    char_start: int,
    char_end: int,
    text: str,
    phoneme_segments: Sequence[PhonemeSegment],
    cfg: PipelineConfig,
    marker_events: Sequence[Any],
) -> str:
    """Create a stable identity hash from prepared audio-semantic content.

    The ``pykokoro-audio-unit-v1`` schema intentionally excludes tracing,
    retention, cache-directory, provider-session, and machine-local path state.
    Callers changing the hash schema must use a new schema prefix.
    """
    payload = {
        "schema": "pykokoro-audio-unit-v1",
        "paragraph_idx": paragraph_idx,
        "char_start": char_start,
        "char_end": char_end,
        "text": text,
        "segments": [
            {
                "id": segment.id,
                "text": segment.text,
                "phonemes": segment.phonemes,
                "lang": segment.lang,
                "voice": (
                    segment.voice_name,
                    segment.voice_language,
                    segment.voice_gender,
                    segment.voice_variant,
                ),
                "prosody": _freeze_config_value(segment.ssmd_metadata),
                "pause_before": segment.pause_before,
                "pause_after": segment.pause_after,
            }
            for segment in phoneme_segments
        ],
        "markers": [
            (event.pos, event.attrs.get("marker"), _freeze_config_value(event.attrs))
            for event in marker_events
        ],
        "config": _audio_identity_config(cfg),
    }
    encoded = json.dumps(payload, sort_keys=True, default=repr, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _audio_identity_config(cfg: PipelineConfig) -> dict[str, object]:
    """Project pipeline settings that can change rendered unit audio."""
    return {
        "voice": _freeze_config_value(cfg.voice),
        "generation": _freeze_config_value(cfg.generation),
        "ssmd": _freeze_config_value(cfg.ssmd),
        "prosody": _freeze_config_value(cfg.prosody),
        "loudness": _freeze_config_value(cfg.loudness),
        "model_quality": cfg.model_quality,
        "model_source": cfg.model_source,
        "model_variant": cfg.model_variant,
        "model_identity": cfg.model_identity,
        "provider": cfg.provider,
        "tokenizer_config": _freeze_config_value(cfg.tokenizer_config),
        "espeak_config": _freeze_config_value(cfg.espeak_config),
        "short_sentence_config": _freeze_config_value(cfg.short_sentence_config),
        "overlap_mode": cfg.overlap_mode,
    }


def _copy_phoneme_segment(segment: PhonemeSegment) -> PhonemeSegment:
    """Copy structural segment state while retaining independent array references."""
    return copy(segment)


def _freeze_config_value(value: Any) -> object:
    """Return an immutable snapshot for values used in backend cache keys."""
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        items: list[tuple[object, object]] = []
        for key, item in value.items():
            items.append((_freeze_config_value(key), _freeze_config_value(item)))
        return ("mapping", tuple(sorted(items, key=lambda item: repr(item[0]))))
    if isinstance(value, (list, tuple)):
        return (
            type(value).__name__,
            tuple(_freeze_config_value(item) for item in value),
        )
    if isinstance(value, (set, frozenset)):
        set_items: list[object] = [_freeze_config_value(item) for item in value]
        return (type(value).__name__, tuple(sorted(set_items, key=repr)))
    if is_dataclass(value):
        return (
            type(value).__qualname__,
            tuple(
                (field.name, _freeze_config_value(getattr(value, field.name)))
                for field in fields(value)
            ),
        )
    return (type(value).__qualname__, id(value))
