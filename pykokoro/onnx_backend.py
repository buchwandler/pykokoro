"""ONNX backend for pykokoro - native ONNX TTS without external dependencies."""

from __future__ import annotations

import io
import json
import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from ._onnxvoice import (
    install_kokoro_model,
    open_installed_kokoro,
    open_local_kokoro,
    runtime_diagnostics,
)
from .asset_progress import AssetProgressCallback
from .audio_generator import AudioGenerator
from .config_types import (
    DEFAULT_MODEL_QUALITY,
    DEFAULT_MODEL_SOURCE,
    DEFAULT_MODEL_VARIANT,
    ModelQuality,
    ModelSource,
    ModelVariant,
    ProviderType,
)
from .exceptions import ConfigurationError
from .model_profiles import VOICE_ALIASES, get_model_profile
from .tokenizer import EspeakConfig, Tokenizer, TokenizerConfig
from .voice_level import VoiceCalibrationKey
from .voice_manager import VoiceBlend, VoiceManager

if TYPE_CHECKING:
    from .loudness_config import LoudnessConfig
    from .prosody_config import ProsodyConfig
    from .short_sentence_handler import ShortSentenceConfig
    from .types import PhonemeSegment, Trace

# Logger for debugging
logger = logging.getLogger(__name__)


def load_vocab_from_config(
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
    config_path: Path | None = None,
) -> dict[str, int]:
    """Load vocabulary from an already-resolved local config artifact.

    Args:
        variant: Model variant (v1.0 or v1.1-zh)
        config_path: Path to the local config/vocab JSON file.  Must be
            supplied by OnnxVoice resolution.  No download or cache-path
            fallback is performed.

    Returns:
        Dictionary mapping phoneme characters to token indices

    Raises:
        FileNotFoundError: If *config_path* is ``None`` or does not exist
        ValueError: If config doesn't contain a valid vocab mapping
    """
    if config_path is None:
        raise FileNotFoundError(
            f"Model config/vocabulary file required for variant '{variant}' "
            "but no config_path was resolved."
        )
    if not config_path.exists():
        raise FileNotFoundError(f"Resolved config path does not exist: {config_path}")

    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, ValueError) as e:
        raise ValueError(f"Failed to load vocabulary from {config_path}: {e}") from e

    # Release vocabularies are direct JSON maps; legacy configs nest
    # the map under "vocab".
    vocab = config.get("vocab") if isinstance(config, dict) else None
    if vocab is None and isinstance(config, dict):
        vocab = config
    if not isinstance(vocab, dict) or not all(
        isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool)
        for key, value in vocab.items()
    ):
        raise ValueError(
            f"Vocabulary at {config_path} must be a JSON object mapping phonemes to IDs"
        )
    logger.info(
        "Loaded vocabulary with %d tokens for variant '%s' from %s",
        len(vocab),
        variant,
        config_path.name,
    )
    return vocab


class Kokoro:
    """
    Native ONNX backend for TTS generation.

    This class provides direct ONNX inference without external dependencies.
    Includes embedded tokenizer for phoneme/token-based generation.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        voices_path: Path | None = None,
        model_config_path: Path | None = None,
        use_gpu: bool = False,
        provider: ProviderType | None = None,
        session_options: Any | None = None,
        provider_options: dict[str, Any] | None = None,
        vocab_version: str = "v1.0",
        espeak_config: EspeakConfig | None = None,
        tokenizer_config: TokenizerConfig | None = None,
        model_quality: ModelQuality | None = None,
        model_source: ModelSource = DEFAULT_MODEL_SOURCE,
        model_variant: ModelVariant = DEFAULT_MODEL_VARIANT,
        short_sentence_config: ShortSentenceConfig | None = None,
        waveform_validation: Literal["off", "warn", "strict"] = "off",
        inference_audio_diagnostics: bool = False,
        inference_cache_enabled: bool = True,
        inference_cache_max_bytes: int = 128 * 1024 * 1024,
        asset_progress: AssetProgressCallback | None = None,
    ) -> None:
        self._session: Any | None = None
        self._voice_manager: VoiceManager | None = None
        self._audio_generator: AudioGenerator | None = None
        self._runtime: Any | None = None
        self._np = np
        self._model_path_provided = model_path is not None
        self._voices_path_provided = voices_path is not None
        self._model_config_path = model_config_path
        self._model_config_path_provided = model_config_path is not None

        # Deprecation warning for use_gpu
        if use_gpu:
            logger.warning(
                "The 'use_gpu' parameter is deprecated and will be removed in a "
                "future version. Use 'provider' parameter instead. "
                "Example: Kokoro(provider='cuda') or Kokoro(provider='auto')"
            )

        self._use_gpu = use_gpu
        self._provider: ProviderType | None = provider
        self._session_options = session_options
        self._model_source: ModelSource = model_source

        # Store initial variant (before auto-detection)
        self._initial_model_variant: ModelVariant = model_variant
        self._model_variant: ModelVariant = model_variant
        self._auto_switched_variant = False

        # Load config for defaults
        from .utils import load_config

        cfg = load_config()

        # Resolve provider_options from config if not specified
        if provider_options is None and "provider_options" in cfg:
            provider_options = cfg.get("provider_options")
            logger.info("Loaded provider_options from config: %s", provider_options)

        self._provider_options = provider_options

        # Resolve model quality from config if not specified
        resolved_quality: ModelQuality = DEFAULT_MODEL_QUALITY
        if model_quality is not None:
            resolved_quality = model_quality
        else:
            quality_from_cfg = cfg.get("model_quality", DEFAULT_MODEL_QUALITY)
            resolved_quality = model_quality or cast(ModelQuality, quality_from_cfg)

        self._model_quality: ModelQuality = resolved_quality
        # Registry assets are resolved lazily as one atomic distribution.
        self._model_path = model_path
        self._voices_path = voices_path
        self._asset_progress = asset_progress
        # Voice database connection (for kokovoicelab integration)
        self._voice_db: sqlite3.Connection | None = None

        # Tokenizer for phoneme-based generation
        self._tokenizer: Tokenizer | None = None
        try:
            self._vocab_version = get_model_profile(
                self._model_variant, self._model_source
            ).tokenizer_vocab_version
        except ValueError:
            self._vocab_version = "1.0"
        self._espeak_config = espeak_config
        self._tokenizer_config = tokenizer_config

        # Short sentence handling configuration
        self._short_sentence_config = short_sentence_config
        if waveform_validation not in {"off", "warn", "strict"}:
            raise ValueError(f"Unsupported waveform validation mode: {waveform_validation!r}")
        self._waveform_validation = waveform_validation
        self._inference_audio_diagnostics = inference_audio_diagnostics
        self._inference_cache_enabled = inference_cache_enabled
        self._inference_cache_max_bytes = inference_cache_max_bytes

        self._init_lock = threading.RLock()

    def _get_vocabulary(self) -> dict[str, int]:
        """Get vocabulary for the current model variant.

        Returns:
            Dictionary mapping phoneme characters to token indices
        """
        from kokorog2p import get_kokoro_vocab

        profile = get_model_profile(self._model_variant, self._model_source)
        vocabulary_path = getattr(self, "_model_config_path", None)
        if profile.vocabulary_source in {"downloaded-config", "downloaded-release"}:
            if vocabulary_path is None:
                raise ConfigurationError(
                    f"Model config/vocabulary file required for {self._model_variant!r} but not resolved."
                )
            return load_vocab_from_config(self._model_variant, vocabulary_path)

        return get_kokoro_vocab()

    def _resolve_model_variant(self, lang: str) -> ModelVariant:
        """Resolve the appropriate model variant based on language.

        Automatically switches to v1.1-zh for Chinese languages unless
        user explicitly specified a variant.

        Args:
            lang: Language code for the text being synthesized

        Returns:
            Resolved model variant to use
        """
        # If user explicitly specified variant, don't auto-switch
        if self._initial_model_variant != DEFAULT_MODEL_VARIANT:
            return self._model_variant

        # Auto-detect: Switch to v1.1-zh for Chinese
        if is_chinese_language(lang) and self._model_source == "github":
            if not self._auto_switched_variant:
                logger.info(
                    "Detected Chinese language '%s'. "
                    "Automatically switching to model variant 'v1.1-zh'.",
                    lang,
                )
                self._auto_switched_variant = True
            return "v1.1-zh"

        # Otherwise use configured variant
        return self._model_variant

    @property
    def tokenizer(self) -> Tokenizer:
        """Get the tokenizer instance (lazily initialized)."""
        if self._tokenizer is None:
            vocab = self._get_vocabulary()
            logger.debug(
                "Initializing tokenizer with %d tokens for variant '%s'",
                len(vocab),
                self._model_variant,
            )
            self._tokenizer = Tokenizer(
                config=self._tokenizer_config,
                espeak_config=self._espeak_config,
                vocab_version=self._vocab_version,
                vocab=vocab,
            )
        return self._tokenizer

    def _init_kokoro(self) -> None:
        """Initialize the ONNX session and load voices once, safely across threads."""
        with self._init_lock:
            self._init_kokoro_locked()

    def _init_kokoro_locked(self) -> None:
        """Resolve the installation and open the shared OnnxVoice runtime."""
        if self._runtime is not None or self._audio_generator is not None:
            return

        started = time.perf_counter()
        logger.info(
            "tts.init.start model=%s source=%s quality=%s provider=%s",
            self._model_variant,
            self._model_source,
            self._model_quality,
            self._provider,
        )
        provider = self._provider or ("auto" if self._use_gpu else "cpu")
        if self._model_path is not None or self._voices_path is not None:
            if self._model_path is None or self._voices_path is None:
                raise ConfigurationError(
                    "Explicit Kokoro local loading requires both model_path and voices_path"
                )
            runtime = open_local_kokoro(
                model=self._model_path,
                voices=self._voices_path,
                config=self._model_config_path,
                providers=provider,
                provider_options=self._provider_options,
                session_options=self._session_options,
            )
            resolved = runtime.resolved
        else:
            resolved = install_kokoro_model(
                self._model_variant,
                quality=str(self._model_quality),
                source=self._model_source,
                cache_dir=None,
                progress=self._asset_progress,
            )
            runtime = open_installed_kokoro(
                resolved,
                providers=provider,
                provider_options=self._provider_options,
                session_options=self._session_options,
            )
        diagnostic = runtime_diagnostics(runtime)
        timing_contract = diagnostic.get("timing_contract", {})
        logger.info(
            "runtime.open model=%s source=%s quality=%s distribution=%s timings=%s timing_layout=%s",
            resolved.model_id or self._model_variant,
            self._model_source,
            resolved.quality or self._model_quality,
            resolved.distribution,
            timing_contract.get("supports_timings"),
            timing_contract.get("layout"),
        )
        self._runtime = runtime
        self._model_path = resolved.model_paths[0] if resolved.model_paths else None
        self._voices_path = resolved.voices_path
        self._model_config_path = resolved.config_path
        if self._voices_path is None:
            raise ConfigurationError("OnnxVoice Kokoro installation has no voices artifact")
        voice_manager = VoiceManager(model_source=self._model_source)
        voice_manager.load_voices(voices_path=self._voices_path)
        self._voice_manager = voice_manager
        self._audio_generator = AudioGenerator(
            runtime=runtime,
            tokenizer=self.tokenizer,
            model_source=self._model_source,
            short_sentence_config=self._short_sentence_config,
            waveform_validation=self._waveform_validation,
            inference_audio_diagnostics=self._inference_audio_diagnostics,
            inference_cache_enabled=self._inference_cache_enabled,
            inference_cache_max_bytes=self._inference_cache_max_bytes,
        )
        logger.info(
            "tts.init.finish model=%s elapsed_ms=%.3f",
            self._model_variant,
            (time.perf_counter() - started) * 1000.0,
        )

    def warmup(self) -> None:
        """Synchronously initialize backend assets, session, voices, and tokenizer."""
        self._init_kokoro()

    def get_voices(self) -> list[str]:
        self._init_kokoro()
        assert self._voice_manager is not None
        return self._voice_manager.get_voices()

    def _voice_manager_voice_name(self, voice_name: str) -> str:
        """Map a registry voice alias to the name stored in a voice archive."""
        return next(
            (
                archive_name
                for (variant, archive_name), registry_name in VOICE_ALIASES.items()
                if variant == self._model_variant and registry_name == voice_name
            ),
            voice_name,
        )

    def get_voice_style(self, voice_name: str) -> np.ndarray:
        self._init_kokoro()
        assert self._voice_manager is not None
        return self._voice_manager.get_voice_style(self._voice_manager_voice_name(voice_name))

    def create_blended_voice(self, blend: VoiceBlend) -> np.ndarray:
        """Create a blended voice style vector from a VoiceBlend."""
        self._init_kokoro()
        assert self._voice_manager is not None
        return self._voice_manager.create_blended_voice(blend)

    def _resolve_voice_style(self, voice: str | np.ndarray | VoiceBlend) -> np.ndarray:
        """Resolve voice parameter to a voice style array."""
        self._init_kokoro()
        if self._voice_manager is None and self._runtime is not None:
            if not isinstance(voice, str):
                raise ConfigurationError(
                    "Voice blending/arrays are not supported by this runtime layout"
                )
            try:
                return np.asarray(self._runtime.voices[voice], dtype=np.float32)[:, None, :]
            except KeyError as exc:
                raise KeyError(f"Voice {voice!r} not found") from exc
        assert self._voice_manager is not None
        if isinstance(voice, str):
            voice = self._voice_manager_voice_name(voice)
        return self._voice_manager.resolve_voice(
            voice,
            voice_db_lookup=self.get_voice_from_database,
        )

    def resolve_voice_style(self, voice: str | np.ndarray | VoiceBlend) -> np.ndarray:
        """Resolve voice parameter to a voice style array."""
        return self._resolve_voice_style(voice)

    def preprocess_segments(
        self,
        segments: list[PhonemeSegment],
        enable_short_sentence_override: bool | None,
        random_seed: int | None = None,
        context_phonemizer: Callable[[str, str], Any] | None = None,
    ) -> list[PhonemeSegment]:
        """Preprocess phoneme segments for short sentence handling."""
        self._init_kokoro()
        assert self._audio_generator is not None
        return self._audio_generator._preprocess_segments(
            segments,
            enable_short_sentence_override,
            random_seed,
            context_phonemizer=context_phonemizer,
        )

    def generate_raw_audio_segments(
        self,
        segments: list[PhonemeSegment],
        voice_style: np.ndarray,
        speed: float,
        voice_resolver: Callable[[str], np.ndarray] | None,
        *,
        default_voice_name: str | None = None,
        trace: Trace | None = None,
    ) -> list[PhonemeSegment]:
        """Generate raw audio for each phoneme segment."""
        self._init_kokoro()
        if getattr(self, "_audio_generator", None) is None and self._runtime is not None:
            default_voice = default_voice_name or next(iter(self._runtime.voices))
            for segment in segments:
                voice_name = segment.voice_name or default_voice
                if (
                    getattr(self, "_model_source", None) is not None
                    and getattr(self, "_model_variant", None) is not None
                    and getattr(self, "_model_quality", None) is not None
                ):
                    segment.render_voice_key = VoiceCalibrationKey(
                        str(self._model_source),
                        str(self._model_variant),
                        str(self._model_quality),
                        voice_name,
                    )
                segment.raw_audio = self._runtime.synthesize(segment.text, voice_name, speed=speed)
            return segments
        for segment in segments:
            voice_name = segment.voice_name or default_voice_name
            if (
                voice_name
                and segment.render_voice_key is None
                and getattr(self, "_model_source", None) is not None
                and getattr(self, "_model_variant", None) is not None
                and getattr(self, "_model_quality", None) is not None
            ):
                segment.render_voice_key = VoiceCalibrationKey(
                    str(self._model_source),
                    str(self._model_variant),
                    str(self._model_quality),
                    voice_name,
                )
        assert self._audio_generator is not None
        return self._audio_generator._generate_raw_audio_segments(
            segments, voice_style, speed, voice_resolver, trace
        )

    def postprocess_audio_segments(
        self,
        segments: list[PhonemeSegment],
        trim_silence: bool,
        prosody_config: ProsodyConfig | None = None,
        trace: Trace | None = None,
        loudness_config: LoudnessConfig | None = None,
    ) -> list[PhonemeSegment]:
        """Trim/prosody-process raw audio segments."""
        self._init_kokoro()
        assert self._audio_generator is not None
        return self._audio_generator._postprocess_audio_segments(
            segments,
            trim_silence,
            prosody_config,
            trace,
            loudness_config,
        )

    def concatenate_audio_segments(
        self,
        segments: list[PhonemeSegment],
        prosody_config: ProsodyConfig | None = None,
        trace: Trace | None = None,
    ) -> np.ndarray:
        """Concatenate processed segments into a single waveform."""
        self._init_kokoro()
        assert self._audio_generator is not None
        return self._audio_generator._concatenate_audio_segments(
            segments,
            prosody_config,
            trace,
        )

    # Voice Database Integration (from kokovoicelab)

    def load_voice_database(self, db_path: Path) -> None:
        """Load a voice database for custom/synthetic voices."""
        if self._voice_db is not None:
            self._voice_db.close()
        sqlite3.register_converter("array", self._convert_array)
        self._voice_db = sqlite3.connect(str(db_path), detect_types=sqlite3.PARSE_DECLTYPES)

    def _convert_array(self, blob: bytes) -> np.ndarray:
        """Convert binary blob back to numpy array."""
        out = io.BytesIO(blob)
        return np.load(out)

    def get_voice_from_database(self, voice_name: str) -> np.ndarray | None:
        """Get a voice style vector from the database."""
        if self._voice_db is None:
            return None
        cursor = self._voice_db.cursor()
        cursor.execute(
            "SELECT style_vector FROM voices WHERE name = ?",
            (voice_name,),
        )
        row = cursor.fetchone()
        if row:
            return row[0]
        return None

    def list_database_voices(self) -> list[dict[str, Any]]:
        """List all voices in the database."""
        if self._voice_db is None:
            return []
        cursor = self._voice_db.cursor()
        cursor.execute(
            """
            SELECT name, gender, language, quality, is_synthetic, notes
            FROM voices
            ORDER BY quality DESC
            """
        )
        voices = []
        for row in cursor.fetchall():
            voices.append(
                {
                    "name": row[0],
                    "gender": row[1],
                    "language": row[2],
                    "quality": row[3],
                    "is_synthetic": bool(row[4]),
                    "notes": row[5],
                }
            )
        return voices

    def interpolate_voices(
        self,
        voice1: str | np.ndarray,
        voice2: str | np.ndarray,
        factor: float = 0.5,
    ) -> np.ndarray:
        """Interpolate between two voices."""
        self._init_kokoro()
        assert self._voice_manager is not None

        style1 = self._voice_manager.resolve_voice(
            voice1, voice_db_lookup=self.get_voice_from_database
        )
        style2 = self._voice_manager.resolve_voice(
            voice2, voice_db_lookup=self.get_voice_from_database
        )

        diff_vector = style2 - style1
        midpoint = (style1 + style2) / 2
        return midpoint + (diff_vector * factor / 2)

    def _generate_from_segments(
        self,
        segments: list[PhonemeSegment],
        voice_style: np.ndarray,
        speed: float,
        trim_silence: bool,
        enable_short_sentence_override: bool | None = None,
        random_seed: int | None = None,
        prosody_config: ProsodyConfig | None = None,
    ) -> np.ndarray:
        """Delegate to AudioGenerator with voice resolution support."""
        self._init_kokoro()
        assert self._audio_generator is not None

        def voice_resolver(voice_name: str) -> np.ndarray:
            """Resolve voice name to style vector."""
            assert self._voice_manager is not None
            return self._voice_manager.resolve_voice(
                voice_name, voice_db_lookup=self.get_voice_from_database
            )

        return self._audio_generator.generate_from_segments(
            segments,
            voice_style,
            speed,
            trim_silence,
            voice_resolver=voice_resolver,
            enable_short_sentence_override=enable_short_sentence_override,
            random_seed=random_seed,
            prosody_config=prosody_config,
        )

    @property
    def runtime_metadata(self) -> dict[str, Any]:
        """Return selected distribution and local artifact identity."""
        runtime = self._runtime
        resolved = getattr(runtime, "resolved", None)
        if resolved is not None:
            return {
                "model_id": resolved.model_id or self._model_variant,
                "ref": resolved.ref,
                "quality": resolved.quality,
                "distribution_id": resolved.distribution,
                "storage_id": resolved.storage_id,
                "release_tag": resolved.metadata.get("release_tag"),
            }

        paths = [path for path in (self._model_path, self._voices_path) if path is not None]
        return {
            "model_id": self._model_variant,
            "ref": None,
            "quality": None,
            "distribution_id": None,
            "storage_id": None,
            "release_tag": None,
            "artifacts": [
                {"path": str(path), "size": path.stat().st_size} for path in paths if path.is_file()
            ],
        }

    def runtime_diagnostics(self) -> dict[str, Any]:
        """Return JSON-safe runtime and timing contract diagnostics."""
        self._init_kokoro()
        runtime = self._runtime
        if runtime is None:
            return {"model_id": self._model_variant, "timing_contract": {"supports_timings": False}}
        return runtime_diagnostics(runtime)

    def close(self) -> None:
        """Release database, tokenizer, voice, generator, and ONNX resources."""
        voice_db, self._voice_db = getattr(self, "_voice_db", None), None

        audio_generator = getattr(self, "_audio_generator", None)
        if callable(getattr(audio_generator, "close", None)):
            audio_generator.close()
        self._audio_generator = None
        self._tokenizer = None
        self._voice_manager = None
        self._session = None
        runtime = getattr(self, "_runtime", None)
        if callable(getattr(runtime, "close", None)):
            runtime.close()
        self._runtime = None

        if voice_db is not None:
            voice_db.close()


def is_chinese_language(lang: str) -> bool:
    """Check if language code is Chinese.

    Args:
        lang: Language code (e.g., 'zh', 'cmn', 'zh-cn')

    Returns:
        True if language is Chinese, False otherwise
    """
    lang_lower = lang.lower().strip()
    return lang_lower in ["zh", "cmn", "zh-cn", "zh-tw", "zh-hans", "zh-hant"]
