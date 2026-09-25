"""Short sentence handling for pykokoro using single-word context approach.

This module provides functionality to improve audio quality for short phrases
 by applying a "context-prepending" technique during phoneme creation.

See ShortSentenceConfig for details.

This approach produces better prosody and intonation compared to generating
very short sentences directly, as neural TTS models typically need more context
to produce natural-sounding speech.

Longer phrases will NOT use this handler, as they already have
sufficient context for natural prosody.
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from .constants import MAX_PHONEME_LENGTH, SUPPORTED_LANGUAGES
from .short_sentence_cutters import cut_phrase_audio
from .short_sentence_phrases import (
    ShortSentencePhraseSet,
    resolve_short_sentence_phrase_language,
    resolve_short_sentence_phrase_set,
)

if TYPE_CHECKING:
    from .types import PhonemeSegment

logger = logging.getLogger(__name__)

SHORT_SENTENCE_META_KEY = "__short_sentence"
ResolveModeName = str | Literal[False]
PhraseSelection = Literal["auto", "neutral", "end"]


def _validate_phrase_cutter_settings(
    frame_duration_ms: int,
    energy_threshold: float,
    min_silence_seconds: float,
    search_radius_ms: float,
    context_guard_ms: float,
    analysis_window_ms: float,
) -> None:
    if (
        isinstance(frame_duration_ms, bool)
        or not isinstance(frame_duration_ms, int)
        or frame_duration_ms <= 0
    ):
        raise ValueError("frame_duration_ms must be a positive integer")
    numeric_values = {
        "energy_threshold": energy_threshold,
        "min_silence_seconds": min_silence_seconds,
        "search_radius_ms": search_radius_ms,
        "context_guard_ms": context_guard_ms,
        "analysis_window_ms": analysis_window_ms,
    }
    for name, value in numeric_values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number")
    if not 0 <= energy_threshold <= 1:
        raise ValueError("energy_threshold must be between 0 and 1")
    if min_silence_seconds < 0:
        raise ValueError("min_silence_seconds must be non-negative")
    if search_radius_ms <= 0:
        raise ValueError("search_radius_ms must be positive")
    if context_guard_ms < 0:
        raise ValueError("context_guard_ms must be non-negative")
    if analysis_window_ms <= 0:
        raise ValueError("analysis_window_ms must be positive")


@dataclass
class WrapResolveMode:
    """Configuration for phoneme pretext wrapping."""

    kind: Literal["wrap"] = "wrap"
    phoneme_pretext: str = "—"


@dataclass
class PhraseResolveMode:
    """Configuration for phrase generation and cutting."""

    kind: Literal["phrase"] = "phrase"
    phrase_selection: PhraseSelection = "auto"
    neutral_phrase: str = "The conversation stopped, {segment}, before someone answered."
    end_phrase: str = "The conversation stopped after one last reply: {segment}"
    frame_duration_ms: int = 5
    energy_threshold: float = 0.05
    silence_threshold: float = 1e-4
    min_silence_seconds: float = 0.02
    cutter: Literal["energy-valley", "timestamp-adaptive"] = "timestamp-adaptive"
    search_radius_ms: float = 35.0
    context_guard_ms: float = 8.0
    analysis_window_ms: float = 5.0

    def __post_init__(self) -> None:
        _validate_phrase_cutter_settings(
            self.frame_duration_ms,
            self.energy_threshold,
            self.min_silence_seconds,
            self.search_radius_ms,
            self.context_guard_ms,
            self.analysis_window_ms,
        )


@dataclass
class RandomizedPhraseResolveMode:
    """Configuration for randomized phrase generation and cutting."""

    kind: Literal["randomized-phrase"] = "randomized-phrase"
    phrase_selection: PhraseSelection = "auto"
    neutral_phrases: list[str] = field(
        default_factory=lambda: [
            "The conversation stopped, {segment}, before someone answered.",
            "The hallway went quiet; {segment}; then footsteps resumed.",
            "The radio paused; {segment}; then the broadcast continued.",
            "The student thought, {segment}, before the teacher continued.",
            "He paused…: {segment}…? … Is that you?",
            "The screen changed; {segment}; then the next slide appeared.",
            "The transcript paused…: {segment}; the next entry followed.",
            "The music stopped: {segment}. Then the singer continued.",
            "He paused…: {segment}? Is that you?",
            "The clerk paused, {segment}, before the next name was called.",
        ]
    )
    end_phrases: list[str] = field(
        default_factory=lambda: [
            "The conversation stopped after one last reply: {segment}",
            "The teacher waited for a response. {segment}",
            "The announcement ended like this: {segment}",
            "There was a pause before the answer came: {segment}",
            "The report concludes with this note: {segment}",
            "The recording trails off after the words, … {segment}",
            "The lesson ended when the teacher asked, {segment}",
            "The host asked again, more quietly this time: {segment}",
            "The letter closed with this unfinished thought — {segment}",
            "The note on the desk simply said, {segment}",
            "At last, the guide called out, {segment}",
        ]
    )
    question_phrases: list[str] = field(
        default_factory=lambda: [
            "The question was asked plainly: {segment}",
            "A quiet voice asked: {segment}",
        ]
    )
    exclamation_phrases: list[str] = field(
        default_factory=lambda: [
            "The speaker called out: {segment}",
            "The announcement ended with: {segment}",
        ]
    )
    ellipsis_phrases: list[str] = field(
        default_factory=lambda: [
            "The thought trailed off with: {segment}",
            "The unfinished sentence was: {segment}",
        ]
    )
    fragment_phrases: list[str] = field(
        default_factory=lambda: [
            "The note contained only these words: {segment}",
            "The short message read: {segment}",
        ]
    )
    frame_duration_ms: int = 5
    energy_threshold: float = 0.05
    silence_threshold: float = 1e-4
    min_silence_seconds: float = 0.02
    cutter: Literal["energy-valley", "timestamp-adaptive"] = "timestamp-adaptive"
    search_radius_ms: float = 35.0
    context_guard_ms: float = 8.0
    analysis_window_ms: float = 5.0

    def __post_init__(self) -> None:
        _validate_phrase_cutter_settings(
            self.frame_duration_ms,
            self.energy_threshold,
            self.min_silence_seconds,
            self.search_radius_ms,
            self.context_guard_ms,
            self.analysis_window_ms,
        )


ShortSentenceResolveMode = WrapResolveMode | PhraseResolveMode | RandomizedPhraseResolveMode


@dataclass
class ShortSentenceApplication:
    """Result of applying a short sentence resolve mode."""

    phonemes: str
    tokens: list[int]
    metadata: dict[str, object] | None = None


@dataclass
class ShortSentenceTimingToken:
    """Serializable token metadata used to map timestamped model durations."""

    text: str
    phonemes: str
    whitespace: str
    is_target: bool = False
    char_start: int | None = None
    char_end: int | None = None
    model_token_count: int | None = None
    model_span_token_count: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "phonemes": self.phonemes,
            "whitespace": self.whitespace,
            "is_target": self.is_target,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "model_token_count": self.model_token_count,
            "model_span_token_count": self.model_span_token_count,
        }


@dataclass
class ShortSentenceConfig:
    """Configuration for short sentence handling using single-word context.

    Short or single-word phrases (< 30 phonemes) often sound robotic
    when generated alone in most voices. E.g. "Oh!" or "One step."
    This module improves quality by applying workarounds.

    Mode: phrase and randomized-phrase (default when the model exposes duration timestamps)
    1. Add a full sentence around the phrase. "Phrase" uses a fixed
       sentence, "Randomized-Phrase" chooses from a list for variety.
    2. Cut out the phrase when timestamp geometry and a legal waveform boundary are available.
    3. Retry with another surrounding phrase only when alignment or timestamps are unusable.
    This mode works best, but can increase computation time.
    Accuracy is voice dependent, but a less accurate voice will only
    slow it down, not stop it from working.
    Phrase-based modes require an ONNX duration/timestamp output. When the
    implicit default is used with a model that lacks timestamps, the runtime
    selects `wrap`. An explicitly requested phrase mode also falls back to
    `wrap` and emits a warning.

    Mode: Wrap
    1. Add phoneme pretext around the phrase. (e.g. "—" or "…")
    This mode is faster and still an improvement over no short-sentence handling.

    Attributes:
        min_phoneme_length: Threshold below which sentences are
            considered "short" based on token count and will use
            context extraction. Default: 30 (decent for most voices).
            Set as low as you can without having garbled or stretched
            short phrases with your voice.
        phoneme_pretext: Phoneme(s) to add before and after the target word
            when generating combined audio for context. Default: "—".
        enabled: Whether short sentence handling is enabled. Default: True.
        resolve_mode: Resolve mode to apply to all short sentences. Default:
            "randomized-phrase".
        phrase_fallback_tries: Number of alternate phrase templates to
            try when alignment or timestamp geometry is unusable,
            before falling back to wrap mode. Default: 1
            Higher=more robust and possibly slower for less-accurate
            voices, Lower=falls back to wrap mode quicker.

    """

    min_phoneme_length: int = 30
    phoneme_pretext: str = "—"
    enabled: bool = True
    resolve_mode: ResolveModeName = "randomized-phrase"
    resolve_modes: dict[str, ShortSentenceResolveMode] = field(
        default_factory=lambda: {
            "wrap": WrapResolveMode(),
            "phrase": PhraseResolveMode(),
            "randomized-phrase": RandomizedPhraseResolveMode(),
        }
    )
    phrase_fallback_tries: int = 1

    phrase_catalog: dict[str, ShortSentencePhraseSet] | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_phoneme_length, bool)
            or not isinstance(self.min_phoneme_length, int)
            or self.min_phoneme_length < 0
        ):
            raise ValueError(
                "min_phoneme_length must be a non-negative integer, "
                f"got {self.min_phoneme_length!r}"
            )
        if (
            isinstance(self.phrase_fallback_tries, bool)
            or not isinstance(self.phrase_fallback_tries, int)
            or self.phrase_fallback_tries < 0
        ):
            raise ValueError(
                "phrase_fallback_tries must be a non-negative integer, "
                f"got {self.phrase_fallback_tries!r}"
            )
        if not isinstance(self.resolve_mode, (str, bool)) or (
            isinstance(self.resolve_mode, bool) and self.resolve_mode
        ):
            raise ValueError(
                f"resolve_mode must be a mode name or False, got {self.resolve_mode!r}"
            )
        if self.resolve_mode is not False and self.resolve_mode not in self.resolve_modes:
            raise ValueError(
                f"resolve_mode {self.resolve_mode!r} is not configured; "
                f"choose one of {sorted(self.resolve_modes)} or False"
            )
        for mode in self.resolve_modes.values():
            templates: list[str] = []
            if isinstance(mode, PhraseResolveMode):
                templates = [mode.neutral_phrase, mode.end_phrase]
            elif isinstance(mode, RandomizedPhraseResolveMode):
                templates = [
                    *mode.neutral_phrases,
                    *mode.end_phrases,
                    *mode.question_phrases,
                    *mode.exclamation_phrases,
                    *mode.ellipsis_phrases,
                    *mode.fragment_phrases,
                ]
            if any(template.count("{segment}") != 1 for template in templates):
                raise ValueError(
                    "phrase templates must contain exactly one '{segment}' placeholder"
                )
        if self.phrase_catalog is not None:
            for key, phrase_set in self.phrase_catalog.items():
                if not isinstance(phrase_set, ShortSentencePhraseSet):
                    raise ValueError("phrase_catalog values must be ShortSentencePhraseSet")
                if key.strip().lower().replace("_", "-") != phrase_set.language.lower():
                    raise ValueError("phrase catalog key must match its declared language")

    def should_use_pause_surrounding(self, phoneme_length: int, text: str) -> bool:
        """Check if segment should use pause surrounding.

        Args:
            phoneme_length: Token count for the segment
            text: The text content to check for single-word status

        Returns:
            True if pause-surrounding should be applied
            (sentence is short AND single-word)
        """
        return self.get_resolve_mode_name(phoneme_length) is not False

    def get_resolve_mode_name(self, phoneme_length: int) -> ResolveModeName:
        """Return the configured resolve mode name for a token length."""
        if not self.enabled:
            return False

        if phoneme_length < self.min_phoneme_length:
            return self.resolve_mode
        return False

    def get_resolve_mode(self, phoneme_length: int) -> ShortSentenceResolveMode | None:
        """Return the configured resolve mode for a token length."""
        mode_name = self.get_resolve_mode_name(phoneme_length)
        if mode_name is False:
            return None
        return self.resolve_modes.get(mode_name)

    def contains_only_punctuation(self, phoneme: str) -> bool:
        """Check if segment contains only pounctions.

        Args:
            phoneme_length: Number of phonemes in the segment
            text: The text content to check for single-word status

        Returns:
            True if segment skipping should be applied
            (sentence is short AND single-word)
        """
        contains_only = ';:,.!?—…"()“” '

        return (
            self.enabled
            and len(phoneme) < self.min_phoneme_length
            and all(char in contains_only for char in phoneme)
        )


def is_segment_empty(
    segment: PhonemeSegment,
    config: ShortSentenceConfig | None = None,
) -> bool:
    """Check if segment contains only .

    Checks if segment is BOTH short (<10 phonemes) AND contains only pounctions.

    Args:
        segment: PhonemeSegment to check
        config: Configuration (uses defaults if None)

    Returns:
        True if segment should be skipped
    """
    if config is None:
        config = ShortSentenceConfig()

    # Skip empty segments
    if not segment.phonemes.strip():
        return False
    return config.contains_only_punctuation(segment.phonemes)


def is_segment_short(
    segment: PhonemeSegment,
    config: ShortSentenceConfig | None = None,
) -> bool:
    """Check if segment should use context-prepending.

    Checks if segment is short (e.g. <30 phonemes).

    Args:
        segment: PhonemeSegment to check
        config: Configuration (uses defaults if None)

    Returns
        True if segment should use pause-surrounding (short AND single-word)
    """
    if config is None:
        config = ShortSentenceConfig()

    # Skip empty segments
    if not segment.phonemes.strip():
        return False

    token_length = len(segment.tokens) if segment.tokens else len(segment.phonemes)
    return config.should_use_pause_surrounding(token_length, segment.text)


def phonemize_short_sentence_phrase(
    segment: PhonemeSegment,
    phrase_template: str,
    context_phonemizer: Callable[[str, str], Any] | None = None,
    *,
    phrase_language: str | None = None,
    tokenize: Callable[[str], list[int]] | None = None,
) -> tuple[str, list[int], list[dict[str, object]]]:
    """Phonemize a phrase containing the short segment text."""
    phrase_text = phrase_template.replace("{segment}", segment.text)
    segment_start = phrase_template.find("{segment}")
    segment_end = segment_start + len(segment.text) if segment_start >= 0 else -1
    segment_language = resolve_short_sentence_phrase_language(segment.lang) or segment.lang
    resolved_phrase_language = (
        resolve_short_sentence_phrase_language(phrase_language)
        if phrase_language is not None
        else segment_language
    )
    if resolved_phrase_language != segment_language:
        raise ValueError(
            f"short sentence phrase language {phrase_language!r} does not match "
            f"segment language {segment.lang!r}"
        )
    lang = SUPPORTED_LANGUAGES.get(segment.lang, segment.lang)
    started = time.perf_counter()
    logger.debug(
        "short_sentence.context_g2p.start mode=phrase segment_language=%s phrase_language=%s "
        "terminal_form=%s reuse=%s",
        segment_language,
        resolved_phrase_language,
        _terminal_form(segment.text),
        context_phonemizer is not None,
    )
    if context_phonemizer is None:
        import kokorog2p

        result = kokorog2p.phonemize(
            phrase_text,
            language=lang,
            return_phonemes=True,
            return_ids=True,
        )
    else:
        result = context_phonemizer(phrase_text, lang)
    logger.debug(
        "short_sentence.context_g2p.finish elapsed_ms=%.3f",
        (time.perf_counter() - started) * 1000.0,
    )
    phonemes = getattr(result, "phonemes", None) or getattr(result, "phoneme", "")
    tokens = getattr(result, "ids", None) or getattr(result, "token_ids", [])
    timing_tokens = _build_timing_tokens(
        getattr(result, "tokens", []),
        segment_start=segment_start,
        segment_end=segment_end,
        tokenize=tokenize,
    )
    return str(phonemes), list(tokens), timing_tokens


def _build_wrap_application(
    segment: PhonemeSegment,
    phonemes: str,
    tokens: list[int],
    config: ShortSentenceConfig,
    mode_name: str,
    tokenize: Callable[[str], list[int]],
    cut_failure_reason: str | None = None,
) -> ShortSentenceApplication:
    wrap_mode = config.resolve_modes.get("wrap")
    pretext = (
        wrap_mode.phoneme_pretext
        if isinstance(wrap_mode, WrapResolveMode)
        else config.phoneme_pretext
    )
    if pretext == "—":
        pretext = config.phoneme_pretext
    wrapped = f"{pretext}{phonemes}{pretext}"
    wrapped_tokens = tokenize(wrapped)
    metadata: dict[str, object] = {
        "mode": mode_name,
        "kind": "wrap",
        "original_token_count": len(tokens),
        "generated_token_count": len(wrapped_tokens),
    }
    if cut_failure_reason is not None:
        metadata["cut_failure_reason"] = cut_failure_reason
    return ShortSentenceApplication(wrapped, wrapped_tokens, metadata)


def apply_short_sentence_mode(
    segment: PhonemeSegment,
    phonemes: str,
    tokens: list[int],
    config: ShortSentenceConfig,
    tokenize: Callable[[str], list[int]],
    rng: random.Random | None = None,
    context_phonemizer: Callable[[str, str], Any] | None = None,
) -> ShortSentenceApplication:
    """Apply the configured short sentence resolve mode to a segment."""
    mode_name = config.get_resolve_mode_name(len(tokens))
    if mode_name is False:
        return ShortSentenceApplication(phonemes, tokens)

    mode = config.resolve_modes.get(mode_name)
    if mode is None:
        logger.warning("Unknown short sentence resolve mode '%s'", mode_name)
        return ShortSentenceApplication(phonemes, tokens)

    if mode.kind == "wrap":
        return _build_wrap_application(segment, phonemes, tokens, config, mode_name, tokenize)
    phrase_set, phrase_language, catalog_source = resolve_short_sentence_phrase_set(
        segment.lang, config.phrase_catalog
    )
    if phrase_set is None or phrase_language is None:
        logger.info(
            "short_sentence.phrase.unavailable language=%s action=wrap reason=no-localized-phrase-catalog",
            segment.lang,
        )
        return _build_wrap_application(
            segment,
            phonemes,
            tokens,
            config,
            mode_name,
            tokenize,
            cut_failure_reason="no-localized-phrase-catalog",
        )
    phrase_mode = _configured_phrase_mode(config)
    phrase_template = _select_phrase_template(
        segment.text,
        mode,
        phrase_mode=phrase_mode,
        phrase_set=phrase_set,
        rng=rng,
    )
    phrase_fallback_templates = _select_phrase_fallback_templates(
        segment.text,
        mode,
        phrase_mode=phrase_mode,
        phrase_set=phrase_set,
        used_templates=[phrase_template],
        limit=config.phrase_fallback_tries,
    )
    try:
        phrase_result = phonemize_short_sentence_phrase(
            segment,
            phrase_template,
            context_phonemizer=context_phonemizer,
            tokenize=tokenize,
        )
    except (RuntimeError, ValueError, KeyError) as exc:
        logger.warning(
            "Failed to phonemize short sentence phrase for '%s': %s",
            segment.text[:50],
            exc,
        )
        return ShortSentenceApplication(phonemes, tokens)

    phrase_phonemes, phrase_tokens, timing_tokens = _coerce_phrase_result(phrase_result)
    if len(phrase_tokens) > MAX_PHONEME_LENGTH:
        logger.warning(
            "Short sentence phrase for '%s' exceeded max token length; using original segment",
            segment.text[:50],
        )
        return ShortSentenceApplication(phonemes, tokens)

    fallback_phonemes = _wrap_phonemes(phonemes, config)
    metadata = _build_short_sentence_metadata(
        mode_name=mode_name,
        mode=mode,
        original_token_count=len(tokens),
        generated_token_count=len(phrase_tokens),
        phrase_template=phrase_template,
        phrase_language=phrase_language,
        phrase_terminal_form=_terminal_form(segment.text),
        phrase_catalog_source=catalog_source,
        phrase_fallback_templates=phrase_fallback_templates,
        phrase_fallback_tries=config.phrase_fallback_tries,
        timing_tokens=timing_tokens,
        fallback_phonemes=fallback_phonemes,
        fallback_tokens=tokenize(fallback_phonemes),
    )
    return ShortSentenceApplication(phrase_phonemes, phrase_tokens, metadata)


def cut_short_sentence_phrase_audio(
    audio: np.ndarray, metadata: dict[str, object]
) -> np.ndarray | None:
    """Cut phrase-generated short sentence audio using the configured cutter."""
    kind = metadata.get("kind")
    if kind not in {"phrase", "randomized-phrase"}:
        return audio
    if audio.size == 0:
        return audio
    metadata["cutter_invoked"] = True
    result = cut_phrase_audio(audio, metadata)
    if result is not None:
        metadata["cutter_reached"] = True
    return result


def build_short_sentence_phrase_retry(
    segment: PhonemeSegment,
    phrase_template: str,
    base_metadata: dict[str, object],
    context_phonemizer: Callable[[str, str], Any] | None = None,
    tokenize: Callable[[str], list[int]] | None = None,
) -> ShortSentenceApplication | None:
    """Build a retry phrase application using the original phrase-cut settings."""
    try:
        phrase_result = phonemize_short_sentence_phrase(
            segment,
            phrase_template,
            context_phonemizer=context_phonemizer,
            tokenize=tokenize,
        )
    except (RuntimeError, ValueError, KeyError) as exc:
        logger.warning(
            "Failed to phonemize short sentence fallback phrase for '%s': %s",
            segment.text[:50],
            exc,
        )
        return None

    phrase_phonemes, phrase_tokens, timing_tokens = _coerce_phrase_result(phrase_result)
    if len(phrase_tokens) > MAX_PHONEME_LENGTH:
        logger.warning(
            "Short sentence fallback phrase for '%s' exceeded max token length; "
            "skipping phrase fallback",
            segment.text[:50],
        )
        return None

    metadata = _build_short_sentence_retry_metadata(
        base_metadata,
        generated_token_count=len(phrase_tokens),
        phrase_template=phrase_template,
        timing_tokens=timing_tokens,
    )
    return ShortSentenceApplication(phrase_phonemes, phrase_tokens, metadata)


def _select_phrase_template(
    segment_text: str,
    mode: ShortSentenceResolveMode,
    *,
    phrase_mode: PhraseResolveMode | None = None,
    phrase_set: ShortSentencePhraseSet | None = None,
    rng: random.Random | None = None,
) -> str:
    use_end_phrase = _uses_end_phrase(segment_text, mode)
    if isinstance(mode, RandomizedPhraseResolveMode):
        effective_phrase_set = phrase_set if mode == RandomizedPhraseResolveMode() else None
        choices = _phrase_choices(segment_text, mode, phrase_mode, effective_phrase_set)
        chooser = rng if rng is not None else random
        return chooser.choice(choices)
    if isinstance(mode, PhraseResolveMode):
        default_mode = PhraseResolveMode()
        uses_builtin_templates = (
            mode.neutral_phrase == default_mode.neutral_phrase
            and mode.end_phrase == default_mode.end_phrase
        )
        effective_phrase_set = phrase_set if uses_builtin_templates else None
        if effective_phrase_set is not None and use_end_phrase:
            choices = effective_phrase_set.declarative or effective_phrase_set.neutral
            return choices[0]
        if effective_phrase_set is not None:
            choices = effective_phrase_set.neutral or effective_phrase_set.fragment
            return choices[0]
        return mode.end_phrase if use_end_phrase else mode.neutral_phrase
    return ""


def _select_phrase_fallback_templates(
    segment_text: str,
    mode: ShortSentenceResolveMode,
    *,
    phrase_mode: PhraseResolveMode | None = None,
    phrase_set: ShortSentencePhraseSet | None = None,
    used_templates: list[str],
    limit: int,
) -> list[str]:
    limit = max(0, int(limit))
    if limit == 0:
        return []

    used = set(used_templates)
    if isinstance(mode, PhraseResolveMode):
        effective_phrase_set = (
            phrase_set if phrase_set is not None and phrase_set.language != "en" else None
        )
        choices = _default_ranked_phrase_choices(segment_text, effective_phrase_set)
        return _unique_phrase_templates(choices, used, limit)

    if isinstance(mode, RandomizedPhraseResolveMode):
        choices = _phrase_choices(
            segment_text,
            mode,
            phrase_mode,
            phrase_set if mode == RandomizedPhraseResolveMode() else None,
        )
        selected = used_templates[-1] if used_templates else ""
        try:
            selected_index = choices.index(selected)
        except ValueError:
            selected_index = -1
        ordered = [
            choices[(selected_index + offset) % len(choices)]
            for offset in range(1, len(choices) + 1)
        ]
        return _unique_phrase_templates(ordered, used, limit)

    return []


def _phrase_choices(
    segment_text: str,
    mode: RandomizedPhraseResolveMode,
    phrase_defaults: PhraseResolveMode | None = None,
    phrase_set: ShortSentencePhraseSet | None = None,
) -> list[str]:
    if phrase_defaults is None:
        phrase_defaults = PhraseResolveMode()
    if mode.phrase_selection == "neutral":
        if phrase_set is not None:
            return list(phrase_set.neutral or phrase_set.fragment)
        return mode.neutral_phrases or [phrase_defaults.neutral_phrase]
    if mode.phrase_selection == "end":
        if phrase_set is not None:
            return list(phrase_set.declarative or phrase_set.neutral)
        return mode.end_phrases or [phrase_defaults.end_phrase]
    terminal_form = _terminal_form(segment_text)
    if phrase_set is not None:
        choices = {
            "question": phrase_set.question,
            "exclamation": phrase_set.exclamation,
            "ellipsis": phrase_set.ellipsis,
            "fragment": phrase_set.fragment,
            "declarative": phrase_set.declarative,
        }[terminal_form]
        fallback = phrase_set.declarative if terminal_form == "declarative" else phrase_set.neutral
        return list(choices or fallback)
    fragment_choices = mode.fragment_phrases
    defaults = RandomizedPhraseResolveMode()
    if (
        mode.neutral_phrases != defaults.neutral_phrases
        and mode.fragment_phrases == defaults.fragment_phrases
    ):
        fragment_choices = mode.neutral_phrases
    choices = {
        "question": mode.question_phrases,
        "exclamation": mode.exclamation_phrases,
        "ellipsis": mode.ellipsis_phrases,
        "fragment": fragment_choices,
        "declarative": mode.end_phrases,
    }[terminal_form]
    fallback = (
        phrase_defaults.end_phrase
        if terminal_form == "declarative"
        else phrase_defaults.neutral_phrase
    )
    return choices or [fallback]


def _configured_phrase_mode(config: ShortSentenceConfig) -> PhraseResolveMode | None:
    mode = config.resolve_modes.get("phrase")
    if isinstance(mode, PhraseResolveMode):
        return mode
    return None


def _default_ranked_phrase_choices(
    segment_text: str,
    phrase_set: ShortSentencePhraseSet | None = None,
) -> list[str]:
    if phrase_set is None and _terminal_form(segment_text) == "fragment":
        return list(RandomizedPhraseResolveMode().neutral_phrases)
    defaults = (
        RandomizedPhraseResolveMode(fragment_phrases=())
        if phrase_set is None
        else RandomizedPhraseResolveMode()
    )
    return _phrase_choices(segment_text, defaults, phrase_set=phrase_set)


def _unique_phrase_templates(
    choices: list[str],
    used: set[str],
    limit: int,
) -> list[str]:
    selected: list[str] = []
    for choice in choices:
        if choice in used:
            continue
        used.add(choice)
        selected.append(choice)
        if len(selected) >= limit:
            break
    return selected


def _terminal_form(
    segment_text: str,
) -> Literal["declarative", "question", "exclamation", "ellipsis", "fragment"]:
    text = segment_text.rstrip()
    if text.endswith(("…", "...")):
        return "ellipsis"
    if text.endswith(("?", "？", "؟")):
        return "question"
    if text.endswith(("!", "！")):
        return "exclamation"
    if text.endswith((".", "。", "।")):
        return "declarative"
    return "fragment"


def _uses_end_phrase(segment_text: str, mode: ShortSentenceResolveMode) -> bool:
    if isinstance(mode, (PhraseResolveMode, RandomizedPhraseResolveMode)):
        if mode.phrase_selection == "end":
            return True
        if mode.phrase_selection == "neutral":
            return False
    return _terminal_form(segment_text) != "fragment"


def _build_short_sentence_metadata(
    *,
    mode_name: str,
    mode: ShortSentenceResolveMode,
    original_token_count: int,
    generated_token_count: int,
    phrase_template: str | None = None,
    phrase_language: str | None = None,
    phrase_terminal_form: str | None = None,
    phrase_catalog_source: str | None = None,
    phrase_fallback_templates: list[str] | None = None,
    phrase_fallback_tries: int | None = None,
    timing_tokens: list[dict[str, object]] | None = None,
    fallback_phonemes: str | None = None,
    fallback_tokens: list[int] | None = None,
) -> dict[str, object]:
    if isinstance(mode, WrapResolveMode):
        frame_duration_ms = 5
        energy_threshold = 0.05
        silence_threshold = 1e-4
        min_silence_seconds = 0.02
        cutter = "energy-valley"
        search_radius_ms = 35.0
        context_guard_ms = 8.0
        analysis_window_ms = 5.0
    else:
        frame_duration_ms = mode.frame_duration_ms
        energy_threshold = mode.energy_threshold
        silence_threshold = mode.silence_threshold
        min_silence_seconds = mode.min_silence_seconds
        cutter = mode.cutter
        search_radius_ms = mode.search_radius_ms
        context_guard_ms = mode.context_guard_ms
        analysis_window_ms = mode.analysis_window_ms
    expected_cut_ratio = 1.0
    if generated_token_count > 0:
        expected_cut_ratio = original_token_count / generated_token_count
    metadata: dict[str, object] = {
        "mode": mode_name,
        "kind": mode.kind,
        "phrase_template": phrase_template,
        "phrase_language": phrase_language,
        "phrase_terminal_form": phrase_terminal_form,
        "phrase_catalog_source": phrase_catalog_source,
        "original_token_count": original_token_count,
        "generated_token_count": generated_token_count,
        "expected_cut_ratio": max(0.01, min(0.99, expected_cut_ratio)),
        "frame_duration_ms": frame_duration_ms,
        "energy_threshold": energy_threshold,
        "silence_threshold": silence_threshold,
        "min_silence_seconds": min_silence_seconds,
        "cutter": cutter,
        "search_radius_ms": search_radius_ms,
        "context_guard_ms": context_guard_ms,
        "analysis_window_ms": analysis_window_ms,
    }
    if timing_tokens:
        metadata["timing_tokens"] = timing_tokens
    if phrase_fallback_templates:
        metadata["phrase_fallback_templates"] = phrase_fallback_templates
    if phrase_fallback_tries is not None:
        metadata["phrase_fallback_tries"] = max(0, int(phrase_fallback_tries))
    if fallback_phonemes is not None:
        metadata["fallback_phonemes"] = fallback_phonemes
    if fallback_tokens is not None:
        metadata["fallback_tokens"] = fallback_tokens
    return metadata


def _build_short_sentence_retry_metadata(
    base_metadata: dict[str, object],
    *,
    generated_token_count: int,
    phrase_template: str,
    timing_tokens: list[dict[str, object]],
) -> dict[str, object]:
    original_token_count = int(cast(Any, base_metadata.get("original_token_count", 0)))
    expected_cut_ratio = 1.0
    if generated_token_count > 0 and original_token_count > 0:
        expected_cut_ratio = original_token_count / generated_token_count

    metadata: dict[str, object] = {
        "mode": base_metadata.get("mode"),
        "kind": base_metadata.get("kind"),
        "phrase_template": phrase_template,
        "phrase_language": base_metadata.get("phrase_language"),
        "phrase_terminal_form": base_metadata.get("phrase_terminal_form"),
        "phrase_catalog_source": base_metadata.get("phrase_catalog_source"),
        "original_token_count": original_token_count,
        "generated_token_count": generated_token_count,
        "expected_cut_ratio": max(0.01, min(0.99, expected_cut_ratio)),
        "frame_duration_ms": base_metadata.get("frame_duration_ms", 5),
        "energy_threshold": base_metadata.get("energy_threshold", 0.05),
        "silence_threshold": base_metadata.get("silence_threshold", 1e-4),
        "min_silence_seconds": base_metadata.get("min_silence_seconds", 0.02),
        "cutter": base_metadata.get("cutter", "energy-valley"),
        "search_radius_ms": base_metadata.get("search_radius_ms", 35.0),
        "context_guard_ms": base_metadata.get("context_guard_ms", 8.0),
        "analysis_window_ms": base_metadata.get("analysis_window_ms", 5.0),
    }
    if timing_tokens:
        metadata["timing_tokens"] = timing_tokens
    for key in (
        "fallback_phonemes",
        "fallback_tokens",
        "phrase_fallback_tries",
        "short_sentence_attempts",
    ):
        if key in base_metadata:
            metadata[key] = base_metadata[key]
    return metadata


def _wrap_phonemes(phonemes: str, config: ShortSentenceConfig) -> str:
    wrap_mode = config.resolve_modes.get("wrap")
    pretext = wrap_mode.phoneme_pretext if isinstance(wrap_mode, WrapResolveMode) else "â€”"
    if pretext == "â€”":
        pretext = config.phoneme_pretext
    return f"{pretext}{phonemes}{pretext}"


def _coerce_phrase_result(
    phrase_result: tuple[str, list[int]] | tuple[str, list[int], list[dict[str, object]]],
) -> tuple[str, list[int], list[dict[str, object]]]:
    """Accept legacy two-item monkeypatched test tuples and new timing tuples."""
    if len(phrase_result) == 2:
        phrase_phonemes, phrase_tokens = phrase_result
        return phrase_phonemes, phrase_tokens, []
    phrase_phonemes, phrase_tokens, timing_tokens = phrase_result
    return phrase_phonemes, phrase_tokens, timing_tokens


def _build_timing_tokens(
    tokens: object,
    *,
    segment_start: int,
    segment_end: int,
    tokenize: Callable[[str], list[int]] | None = None,
) -> list[dict[str, object]]:
    timing_tokens: list[dict[str, object]] = []
    for token in cast(list[object], tokens or []):
        phonemes = str(_token_attr(token, "phonemes") or _token_attr(token, "phoneme") or "")
        text = str(_token_attr(token, "text") or "")
        whitespace = str(_token_attr(token, "whitespace") or "")
        char_start = _token_attr(token, "char_start")
        char_end = _token_attr(token, "char_end")
        is_target = _token_overlaps_segment(
            char_start,
            char_end,
            segment_start=segment_start,
            segment_end=segment_end,
        )
        source_start = (
            max(0, char_start - segment_start)
            if isinstance(char_start, int) and is_target
            else None
        )
        source_end = (
            max(0, char_end - segment_start) if isinstance(char_end, int) and is_target else None
        )
        raw_model_token_count = _token_attr(token, "model_token_count")
        if (
            isinstance(raw_model_token_count, int)
            and not isinstance(raw_model_token_count, bool)
            and raw_model_token_count >= 0
        ):
            model_token_count = raw_model_token_count
        elif tokenize is not None:
            model_token_count = len(tokenize(phonemes))
        else:
            model_token_count = None
        raw_model_span_count = _token_attr(token, "model_span_token_count")
        if (
            isinstance(raw_model_span_count, int)
            and not isinstance(raw_model_span_count, bool)
            and raw_model_span_count >= 0
        ):
            model_span_token_count = raw_model_span_count
        elif tokenize is not None:
            model_span_token_count = len(tokenize(phonemes + whitespace))
        elif isinstance(model_token_count, int):
            model_span_token_count = model_token_count + (1 if whitespace else 0)
        else:
            model_span_token_count = None
        timing_tokens.append(
            ShortSentenceTimingToken(
                text=text,
                phonemes=phonemes,
                whitespace=whitespace,
                is_target=is_target,
                char_start=source_start,
                char_end=source_end,
                model_token_count=model_token_count,
                model_span_token_count=model_span_token_count,
            ).to_dict()
        )
    return timing_tokens


def _token_attr(token: object, name: str) -> object:
    value = getattr(token, name, None)
    if value is not None:
        return value
    meta = getattr(token, "meta", None)
    if isinstance(meta, dict) and name in meta:
        return meta[name]
    get = getattr(token, "get", None)
    if callable(get):
        return get(name)
    return None


def _token_overlaps_segment(
    char_start: object,
    char_end: object,
    *,
    segment_start: int,
    segment_end: int,
) -> bool:
    if segment_start < 0 or segment_end < 0:
        return False
    if not isinstance(char_start, int) or not isinstance(char_end, int):
        return False
    return char_start < segment_end and char_end > segment_start
