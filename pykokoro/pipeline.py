from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import Self

from .constants import SAMPLE_RATE
from .emphasis import apply_emphasis_policy
from .generation_config import GenerationConfig
from .pipeline_config import PipelineConfig, require_document_language, resolve_model_defaults
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
    return _freeze_config_value(replace(config, lexicons=None))


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


class PreparedAudioUnits:
    """A globally prepared document that can render selected units sequentially."""

    def __init__(self, pipeline: KokoroPipeline, prepared: _PreparedDocument) -> None:
        self._pipeline = pipeline
        self._prepared: _PreparedDocument | None = prepared
        self._units = tuple(group.descriptor for group in prepared.groups)
        self._unit_kind = prepared.unit_kind
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
        ssmd_value = data.pop("ssmd", None)
        tokenizer_value = data.pop("tokenizer_config", None)

        _coerce_paths_inplace(data)
        cfg = PipelineConfig(**data)

        if gen_value is not None:
            cfg = replace(cfg, generation=_coerce_generation(cfg.generation, gen_value))
        if ssmd_value is not None:
            cfg = replace(cfg, ssmd=_coerce_ssmd(cfg.ssmd, ssmd_value))
        if tokenizer_value is not None:
            cfg = replace(
                cfg,
                tokenizer_config=_coerce_tokenizer(cfg.tokenizer_config, tokenizer_value),
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
    ssmd_value = data.pop("ssmd", None)
    tokenizer_value = data.pop("tokenizer_config", None)

    _coerce_paths_inplace(data)
    cfg = replace(base, **data)

    if gen_value is not None:
        cfg = replace(cfg, generation=_coerce_generation(cfg.generation, gen_value))
    if ssmd_value is not None:
        cfg = replace(cfg, ssmd=_coerce_ssmd(cfg.ssmd, ssmd_value))
    if tokenizer_value is not None:
        cfg = replace(
            cfg,
            tokenizer_config=_coerce_tokenizer(cfg.tokenizer_config, tokenizer_value),
        )

    return cfg


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
            pipeline.phoneme_processing = onnx_phoneme_processor(backend)
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
            pipeline.phoneme_processing = onnx_phoneme_processor(kokoro)
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

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

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
        cfg = resolve_model_defaults(cfg)
        kokoro_key = self._kokoro_key(cfg)
        if self._kokoro is not None and self._kokoro_config_key == kokoro_key:
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
        )
        self._kokoro = new_kokoro
        self._kokoro_config_key = kokoro_key
        self._owns_kokoro = True
        if previous_kokoro is not None and previous_owned:
            try:
                previous_kokoro.close()
            except Exception:
                logger.warning("Failed to close replaced Kokoro backend", exc_info=True)
        return new_kokoro, True

    def _resolve_run_config(self, overrides: dict[str, Any]) -> PipelineConfig:
        if not overrides:
            return resolve_model_defaults(self.config)
        overrides = dict(overrides)
        lang = overrides.pop("lang", None)
        has_generation_override = "generation" in overrides
        ssmd_value = overrides.pop("ssmd", None)
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
        cfg = deepcopy(self._resolve_run_config(overrides))
        prepared = self._prepare_document(text, cfg, unit)
        result = PreparedAudioUnits(self, prepared)
        self._prepared_objects.append(result)
        return result

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
                    stage="linguistics",
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

    def _prepare_document(
        self, text: str, cfg: PipelineConfig, unit: AudioUnitKind
    ) -> _PreparedDocument:
        language = require_document_language(cfg)
        trace = Trace()
        with trace_timing(trace, "doc", "parse"):
            logger.debug("Parsing document")
            doc = self.doc_parser.parse(text, cfg, trace)
            trace.warnings.extend(doc.warnings)
            state = LinguisticRequestState()
            doc.linguistic_state = state
            state.source_plan = build_language_plan(
                doc.clean_text, doc.annotation_spans, default_language=language
            )
            trace.events.append(
                TraceEvent(
                    stage="language_plan",
                    name="source",
                    ms=0.0,
                    details={
                        "runs": len(state.source_plan),
                        "languages": [run.language for run in state.source_plan],
                    },
                )
            )
            state.source_analysis = self._analyze_runs(
                doc.clean_text, state.source_plan, cfg, trace, "pass_a"
            )

        try:
            with trace_timing(trace, "text_preparation", "prepare"):
                doc = self.text_preparer.prepare(doc, cfg, trace)
        except Exception:
            state.release_docs()
            doc.linguistic_state = None
            raise
        doc.linguistic_state = state
        trace.warnings.extend(doc.preparation.warnings if doc.preparation else ())
        state.prepared_plan = build_language_plan(
            doc.clean_text, doc.annotation_spans, default_language=language
        )
        trace.events.append(
            TraceEvent(
                stage="language_plan",
                name="prepared",
                ms=0.0,
                details={
                    "runs": len(state.prepared_plan),
                    "languages": [run.language for run in state.prepared_plan],
                },
            )
        )
        try:
            state.prepared_analysis = self._analyze_runs(
                doc.clean_text, state.prepared_plan, cfg, trace, "pass_b"
            )
        except Exception:
            state.release_docs()
            doc.linguistic_state = None
            raise
        state.release_source_docs()

        try:
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

        try:
            with trace_timing(trace, "g2p", "phonemize"):
                logger.debug("Phonemizing %d segments", len(segments))
                phoneme_segments = self.g2p.phonemize(segments, doc, cfg, trace)
        finally:
            state.release_docs()
            doc.linguistic_state = None
        phoneme_processor, audio_generator, audio_postprocessor = self._resolve_stages(cfg)

        with trace_timing(trace, "phoneme_processing", "preprocess"):
            logger.debug("Preprocessing %d phoneme segments", len(phoneme_segments))
            phoneme_segments = phoneme_processor.process(phoneme_segments, cfg, trace)

        apply_emphasis_policy(phoneme_segments, cfg, trace)

        groups = self._build_unit_groups(doc, segments, phoneme_segments, cfg, unit)
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

        if phoneme_processor is None or audio_generator is None or audio_postprocessor is None:
            kokoro, kokoro_changed = self._ensure_kokoro(cfg)
            (
                onnx_phoneme_processor,
                onnx_audio_generation,
                onnx_audio_postprocessing,
            ) = _load_default_onnx_adapters()
            if phoneme_processor is None or (kokoro_changed and self._owns_phoneme_processing):
                phoneme_processor = onnx_phoneme_processor(kokoro)
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
                word_timings=[timing for segment in generated for timing in segment.word_timings],
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

    def run(self, text: str, **overrides: Any) -> AudioResult:
        with self.prepare_units(text, unit="paragraph", **overrides) as prepared:
            cfg = prepared._prepared.cfg if prepared._prepared is not None else self.config
            final_audio: list[Any] = []
            markers: list[dict[str, Any]] = []
            word_timings: list[Any] = []
            retained_phonemes: list[PhonemeSegment] = []
            base_offset = 0
            for unit_result in prepared.render():
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

            source_segments = list(prepared._prepared.segments) if prepared._prepared else []
            source_phonemes = (
                retained_phonemes
                if cfg.retain_segment_audio
                else list(prepared._prepared.phoneme_segments)
                if prepared._prepared
                else []
            )
            audio = np.concatenate(final_audio) if final_audio else np.array([], dtype=np.float32)
            trace = prepared._prepared.trace if prepared._prepared is not None else None
            metadata = dict(prepared.document_metadata)
        return AudioResult(
            audio=audio,
            sample_rate=SAMPLE_RATE,
            segments=source_segments,
            phoneme_segments=source_phonemes,
            trace=trace if cfg.return_trace else None,
            document_metadata=metadata,
            markers=markers,
            word_timings=word_timings,
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
