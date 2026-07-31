from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

from .config_types import LANG_CODE_TO_ONNX
from .constants import SAMPLE_RATE
from .emphasis import apply_emphasis_policy
from .generation_config import GenerationConfig
from .pipeline_config import PipelineConfig
from .runtime.tracing import trace_timing
from .spacy_models import SpacyModelSize
from .ssmd_config import SSMDRenderConfig
from .stages.audio_generation.onnx import OnnxAudioGenerationAdapter
from .stages.audio_postprocessing.onnx import OnnxAudioPostprocessingAdapter
from .stages.doc_parsers.ssmd import SsmdDocumentParser
from .stages.g2p.kokorog2p import KokoroG2PAdapter
from .stages.phoneme_processing.onnx import OnnxPhonemeProcessorAdapter
from .stages.protocols import (
    AudioGeneratorStage,
    AudioPostprocessor,
    DocumentParser,
    G2PAdapter,
    PhonemeProcessor,
)
from .types import AudioResult, Segment, Trace

if TYPE_CHECKING:
    from .onnx_backend import Kokoro
    from .tokenizer import TokenizerConfig

logger = logging.getLogger(__name__)


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


def _coerce_paths_inplace(data: dict[str, Any]) -> None:
    # Convenience: accept str paths in config dict.
    for key in ("model_path", "voices_path"):
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

        _coerce_paths_inplace(data)
        cfg = PipelineConfig(**data)

        if gen_value is not None:
            cfg = replace(cfg, generation=_coerce_generation(cfg.generation, gen_value))
        if ssmd_value is not None:
            cfg = replace(cfg, ssmd=_coerce_ssmd(cfg.ssmd, ssmd_value))

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

    _coerce_paths_inplace(data)
    cfg = replace(base, **data)

    if gen_value is not None:
        cfg = replace(cfg, generation=_coerce_generation(cfg.generation, gen_value))
    if ssmd_value is not None:
        cfg = replace(cfg, ssmd=_coerce_ssmd(cfg.ssmd, ssmd_value))

    return cfg


def with_spacy_model_size(
    config: PipelineConfig | Mapping[str, Any] | None = None,
    *,
    size: SpacyModelSize = "md",
    model: str = "auto",
) -> PipelineConfig:
    """Return PipelineConfig with tokenizer spaCy model settings.

    This helper is useful when you want language-based model resolution in G2P:
    keep ``model="auto"`` and pick the package tier via ``size``.
    """
    cfg = _coerce_pipeline_config(config)

    from .tokenizer import TokenizerConfig

    tokenizer_config: TokenizerConfig = cfg.tokenizer_config or TokenizerConfig()
    tokenizer_config = replace(
        tokenizer_config,
        spacy_model=model,
        spacy_model_size=size,
    )
    return replace(cfg, tokenizer_config=tokenizer_config)


def build_pipeline(
    *,
    config: PipelineConfig | Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
    backend: Kokoro | None = None,
    eager: bool = False,
    # stage overrides for advanced usage/testing
    doc_parser: DocumentParser | None = None,
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
        g2p=g2p or KokoroG2PAdapter(),
        phoneme_processing=phoneme_processing,
        audio_generation=audio_generation,
        audio_postprocessing=audio_postprocessing,
    )

    # If backend injected: bind default stages to it
    # (unless user already provided stages)
    if backend is not None:
        if pipeline.phoneme_processing is None:
            pipeline.phoneme_processing = OnnxPhonemeProcessorAdapter(backend)
        if pipeline.audio_generation is None:
            pipeline.audio_generation = OnnxAudioGenerationAdapter(backend)
        if pipeline.audio_postprocessing is None:
            pipeline.audio_postprocessing = OnnxAudioPostprocessingAdapter(backend)
        return pipeline

    # Eager warmup: create backend now + bind stages + own/close them
    if eager:
        kokoro, _ = pipeline._ensure_kokoro(cfg)

        if pipeline.phoneme_processing is None:
            pipeline.phoneme_processing = OnnxPhonemeProcessorAdapter(kokoro)
            pipeline._owns_phoneme_processing = True

        if pipeline.audio_generation is None:
            pipeline.audio_generation = OnnxAudioGenerationAdapter(kokoro)
            pipeline._owns_audio_generation = True

        if pipeline.audio_postprocessing is None:
            pipeline.audio_postprocessing = OnnxAudioPostprocessingAdapter(kokoro)
            pipeline._owns_audio_postprocessing = True

    return pipeline


def _default_lang_from_voice(cfg: PipelineConfig) -> PipelineConfig:
    if not isinstance(cfg.voice, str):
        return cfg
    if "," in cfg.voice or ":" in cfg.voice:
        return cfg
    default_lang = GenerationConfig().lang
    if cfg.generation.lang != default_lang:
        return cfg
    voice_key = cfg.voice.split("_", 1)[0].strip().lower()
    if not voice_key:
        return cfg
    voice_lang = LANG_CODE_TO_ONNX.get(voice_key[0])
    if not voice_lang or voice_lang == cfg.generation.lang:
        return cfg
    generation = replace(cfg.generation, lang=voice_lang)
    return replace(cfg, generation=generation)


class KokoroPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        doc_parser: DocumentParser | None = None,
        g2p: G2PAdapter | None = None,
        phoneme_processing: PhonemeProcessor | None = None,
        audio_generation: AudioGeneratorStage | None = None,
        audio_postprocessing: AudioPostprocessor | None = None,
    ) -> None:
        self.config = config
        self.doc_parser = doc_parser or SsmdDocumentParser()
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

    def _kokoro_key(self, cfg: PipelineConfig) -> tuple[object, ...]:
        model_path = str(cfg.model_path) if cfg.model_path else None
        voices_path = str(cfg.voices_path) if cfg.voices_path else None
        return (
            model_path,
            voices_path,
            cfg.model_quality,
            cfg.model_source,
            cfg.model_variant,
            cfg.provider,
            _freeze_config_value(cfg.provider_options),
            _freeze_config_value(cfg.session_options),
            _freeze_config_value(cfg.tokenizer_config),
            _freeze_config_value(cfg.espeak_config),
            _freeze_config_value(cfg.short_sentence_config),
        )

    @staticmethod
    def _close_stage(stage: object | None) -> None:
        if stage is None:
            return
        close = getattr(stage, "close", None)
        if callable(close):
            close()

    def _ensure_kokoro(self, cfg: PipelineConfig) -> tuple[Kokoro, bool]:
        kokoro_key = self._kokoro_key(cfg)
        if self._kokoro is not None and self._kokoro_config_key == kokoro_key:
            return self._kokoro, False
        from .onnx_backend import Kokoro

        previous_kokoro = self._kokoro
        previous_owned = self._owns_kokoro
        new_kokoro = Kokoro(
            model_path=Path(cfg.model_path) if cfg.model_path else None,
            voices_path=Path(cfg.voices_path) if cfg.voices_path else None,
            model_quality=cfg.model_quality,
            model_source=cfg.model_source,
            model_variant=cfg.model_variant,
            provider=cfg.provider,
            provider_options=cfg.provider_options,
            session_options=cfg.session_options,
            tokenizer_config=cfg.tokenizer_config,
            espeak_config=cfg.espeak_config,
            short_sentence_config=cfg.short_sentence_config,
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
            return _default_lang_from_voice(self.config)
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
        return _default_lang_from_voice(replace(self.config, **overrides))

    def run(self, text: str, **overrides: Any) -> AudioResult:
        cfg = self._resolve_run_config(overrides)
        trace = Trace()

        with trace_timing(trace, "doc", "parse"):
            logger.debug("Parsing document")
            doc = self.doc_parser.parse(text, cfg, trace)
            trace.warnings.extend(doc.warnings)
        segments = doc.segments
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

        with trace_timing(trace, "g2p", "phonemize"):
            logger.debug("Phonemizing %d segments", len(segments))
            phoneme_segments = self.g2p.phonemize(segments, doc, cfg, trace)

        phoneme_processor = self.phoneme_processing
        audio_generator = self.audio_generation
        audio_postprocessor = self.audio_postprocessing

        if phoneme_processor is None or audio_generator is None or audio_postprocessor is None:
            kokoro, kokoro_changed = self._ensure_kokoro(cfg)
            if phoneme_processor is None or (kokoro_changed and self._owns_phoneme_processing):
                phoneme_processor = OnnxPhonemeProcessorAdapter(kokoro)
                if self.phoneme_processing is None or self._owns_phoneme_processing:
                    self.phoneme_processing = phoneme_processor
                    self._owns_phoneme_processing = True
            if audio_generator is None or (kokoro_changed and self._owns_audio_generation):
                audio_generator = OnnxAudioGenerationAdapter(kokoro)
                if self.audio_generation is None or self._owns_audio_generation:
                    self.audio_generation = audio_generator
                    self._owns_audio_generation = True
            if audio_postprocessor is None or (kokoro_changed and self._owns_audio_postprocessing):
                audio_postprocessor = OnnxAudioPostprocessingAdapter(kokoro)
                if self.audio_postprocessing is None or self._owns_audio_postprocessing:
                    self.audio_postprocessing = audio_postprocessor
                    self._owns_audio_postprocessing = True

        assert phoneme_processor is not None
        assert audio_generator is not None
        assert audio_postprocessor is not None

        with trace_timing(trace, "phoneme_processing", "preprocess"):
            logger.debug("Preprocessing %d phoneme segments", len(phoneme_segments))
            phoneme_segments = phoneme_processor.process(phoneme_segments, cfg, trace)

        apply_emphasis_policy(phoneme_segments, cfg, trace)

        with trace_timing(trace, "audio_generation", "generate"):
            logger.debug("Generating audio for %d phoneme segments", len(phoneme_segments))
            phoneme_segments = audio_generator.generate(phoneme_segments, cfg, trace)

        with trace_timing(trace, "audio_postprocessing", "postprocess"):
            logger.debug("Postprocessing %d phoneme segments", len(phoneme_segments))
            audio = audio_postprocessor.postprocess(phoneme_segments, cfg, trace)

        markers = _collect_marker_offsets(doc.boundary_events, phoneme_segments)
        result = AudioResult(
            audio=audio,
            sample_rate=SAMPLE_RATE,
            segments=segments,
            phoneme_segments=phoneme_segments,
            trace=trace if cfg.return_trace else None,
            document_metadata={
                "title": doc.header.get("title"),
                "voice_bindings": _copy_metadata_value(doc.header.get("voice_bindings", {})),
                "pause_defaults": _copy_metadata_value(doc.header.get("pause_defaults", {})),
            },
            markers=markers,
        )
        if not cfg.retain_segment_audio:
            result.release_segment_audio()
        return result

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
    boundaries: list[Any], phoneme_segments: list[Any]
) -> list[dict[str, Any]]:
    """Map clean-text marker boundaries to deterministic concatenated sample offsets."""

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
                    "sample_offset": sample_offset,
                }
            )
    return offsets


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
