"""SSMD (Speech Synthesis Markdown) parser for pykokoro.

This module provides integration with the SSMD library to support
rich markup syntax for TTS generation including:
- Breaks/Pauses: ...c (comma), ...s (sentence), ...p (paragraph), ...500ms
- Emphasis: *text* (moderate), **text** (strong)
- Prosody: +loud+, >fast>, ^high^, etc.
- Language switching: [Bonjour](fr)
- Phonetic pronunciation: [tomato](ph: təˈmeɪtoʊ)
- Substitution: [H2O](sub: water)
- Say-as: [123](as: cardinal), [3rd](as: ordinal), [+1-555-0123](as: telephone)
- Voice directives: <div voice="name"> ... </div> and [text]{voice="name"}
- Markers: @marker_name

This module uses SSMD's parse_paragraphs() API to extract structured data
and maps it to PyKokoro's internal segment representation.
"""

from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ssmd import SSMDSegment as SSMDParserSegment

_SSMD_IMPORTS: tuple[type, Any] | None = None


def _load_ssmd() -> tuple[type, Any]:
    global _SSMD_IMPORTS
    if _SSMD_IMPORTS is None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "ssmd.parser_types is deprecated; import from ssmd.types, "
                    "ssmd.segment, or ssmd.sentence instead."
                ),
                category=DeprecationWarning,
            )
            from ssmd import TTSCapabilities, parse_paragraphs

        _SSMD_IMPORTS = (TTSCapabilities, parse_paragraphs)
    return _SSMD_IMPORTS


ANNOTATION_RE = re.compile(
    r"""
    \[
        [^\]]+                      # [text]
    \]
    \{
        \s*\w+\s*=\s*
        (?:'[^']+'|"[^"]+")         # 'value' OR "value"
        (?:\s+\w+\s*=\s*(?:'[^']+'|"[^"]+"))*
        \s*
    \}
    """,
    re.VERBOSE,
)

LANG_SHORTHAND_RE = re.compile(r"\[([^\]]+)\]\(([a-zA-Z]{2,3}(?:-[a-zA-Z]{2})?)\)")
BREAK_TIME_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s)\s*$", re.IGNORECASE)
BREAK_MARKER_RE = re.compile(
    r"\.\.\.\s*(?P<token>(?:[nwcsp])|(?:\d+(?:\.\d+)?\s*(?:ms|s)))"
    r"(?=(?:\s|$|[\"'\)\]\}.,!?]))",
    re.IGNORECASE,
)
DIV_TAG_RE = re.compile(r"</?div\b[^>]*>", re.IGNORECASE)

DEFAULT_PAUSE_NONE = 0.0
DEFAULT_PAUSE_WEAK = 0.15
DEFAULT_PAUSE_CLAUSE = 0.3
DEFAULT_PAUSE_SENTENCE = 0.6
DEFAULT_PAUSE_PARAGRAPH = 1.0

logger = logging.getLogger(__name__)


@dataclass
class SSMDMetadata:
    """Metadata extracted from SSMD markup for a text segment.

    Attributes:
        emphasis: Emphasis level ("moderate", "strong", or None)
        prosody_volume: Volume level (0-5 scale or relative like "+6dB")
        prosody_rate: Rate/speed level (1-5 scale or relative like "+20%")
        prosody_pitch: Pitch level (1-5 scale or relative like "+15%")
        language: Language code override for this segment
        phonemes: Explicit phoneme string (bypasses G2P)
        substitution: Substitution text (replaces original before G2P)
        say_as_interpret: Say-as interpretation type (e.g., "telephone", "date")
        say_as_format: Say-as format attribute (e.g., "mdy" for dates)
        say_as_detail: Say-as detail attribute (e.g., "2" for cardinal detail)
        markers: List of marker names in this segment
        voice_name: Voice name for this segment (e.g., "af_sarah", "Joanna")
        voice_language: Voice language attribute (e.g., "en-US", "fr-FR")
        voice_gender: Voice gender attribute ("male", "female", "neutral")
        voice_variant: Voice variant number for multi-variant voices
        audio_src: Audio file source URL or path (for audio segments)
        audio_alt_text: Fallback text if audio cannot be played
    """

    emphasis: str | None = None
    prosody_volume: str | None = None
    prosody_rate: str | None = None
    prosody_pitch: str | None = None
    language: str | None = None
    phonemes: str | None = None
    substitution: str | None = None
    say_as_interpret: str | None = None
    say_as_format: str | None = None
    say_as_detail: str | None = None
    markers: list[str] = field(default_factory=list)
    voice_name: str | None = None
    voice_language: str | None = None
    voice_gender: str | None = None
    voice_variant: str | None = None
    audio_src: str | None = None
    audio_alt_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "emphasis": self.emphasis,
            "prosody_volume": self.prosody_volume,
            "prosody_rate": self.prosody_rate,
            "prosody_pitch": self.prosody_pitch,
            "language": self.language,
            "phonemes": self.phonemes,
            "substitution": self.substitution,
            "say_as_interpret": self.say_as_interpret,
            "say_as_format": self.say_as_format,
            "say_as_detail": self.say_as_detail,
            "markers": self.markers,
            "voice_name": self.voice_name,
            "voice_language": self.voice_language,
            "voice_gender": self.voice_gender,
            "voice_variant": self.voice_variant,
            "audio_src": self.audio_src,
            "audio_alt_text": self.audio_alt_text,
        }


@dataclass
class SSMDSegment:
    """A parsed segment from SSMD markup.

    Attributes:
        text: Processed text (after substitutions, stripped of markup)
        pause_before: Pause duration before this segment in seconds (e.g., for headings)
        pause_after: Pause duration after this segment in seconds
        metadata: SSMD metadata (emphasis, prosody, etc.)
        paragraph: Paragraph index this segment belongs to
        sentence: Sentence index this segment belongs to
    """

    text: str
    pause_before: float = 0.0
    pause_after: float = 0.0
    metadata: SSMDMetadata = field(default_factory=SSMDMetadata)
    paragraph: int = 0
    sentence: int = 0


def has_ssmd_markup(text: str) -> bool:
    """Check if text contains SSMD markup.

    Args:
        text: Input text to check

    Returns:
        True if text contains any SSMD markup patterns
    """
    # Break markers
    if BREAK_MARKER_RE.search(text):
        return True

    # Emphasis (must have word character adjacent to asterisk)
    if re.search(r"\*\w[^*]*\*|\*[^*]*\w\*", text):
        return True

    # Annotations: [text]{annotation)
    if bool(ANNOTATION_RE.search(text)):
        return True

    # Markers: @name
    if re.search(r"(?:^|\s)@\w+", text):
        return True

    return "<div" in text.lower() and bool(
        re.search(
            r"<div\b[^>]*=.*?>.*?</div\s*>",
            text,
            re.IGNORECASE | re.DOTALL,
        )
    )


def _normalize_div_directives(text: str) -> str:
    if "<div" not in text.lower():
        return text

    def _add_linebreaks(match: re.Match[str]) -> str:
        tag = match.group(0)
        return f"\n{tag}\n"

    return DIV_TAG_RE.sub(_add_linebreaks, text)


def _convert_break_strength_to_duration(
    strength: str | None,
    time: str | None,
    pause_none: float = DEFAULT_PAUSE_NONE,
    pause_weak: float = DEFAULT_PAUSE_WEAK,
    pause_clause: float = DEFAULT_PAUSE_CLAUSE,
    pause_sentence: float = DEFAULT_PAUSE_SENTENCE,
    pause_paragraph: float = DEFAULT_PAUSE_PARAGRAPH,
) -> float:
    """Convert SSMD BreakAttrs to pause duration in seconds.

    Args:
        strength: Break strength ('none', 'x-weak', 'weak', 'medium',
            'strong', 'x-strong')
        time: Break time ('500ms', '2s')
        pause_none: Duration for 'none' strength
        pause_weak: Duration for 'x-weak'/'weak' strength
        pause_clause: Duration for 'medium' strength
        pause_sentence: Duration for 'strong' strength
        pause_paragraph: Duration for 'x-strong' strength

    Returns:
        Pause duration in seconds
    """
    # If explicit time is provided, use it
    if time:
        duration = _parse_break_time(time)
        if duration is not None:
            return duration
        logger.warning(
            "Invalid SSMD break duration '%s'; falling back to strength.", time
        )

    # Otherwise use strength mapping
    if strength:
        strength_map = {
            "none": pause_none,
            "x-weak": pause_weak,
            "weak": pause_weak,
            "medium": pause_clause,
            "strong": pause_sentence,
            "x-strong": pause_paragraph,
        }
        return strength_map.get(strength, 0.0)

    return 0.0


def _break_duration_from_token(
    token: str,
    *,
    pause_none: float,
    pause_weak: float,
    pause_clause: float,
    pause_sentence: float,
    pause_paragraph: float,
) -> float | None:
    normalized = token.strip().lower()
    if normalized == "n":
        return pause_none
    if normalized == "w":
        return pause_weak
    if normalized == "c":
        return pause_clause
    if normalized == "s":
        return pause_sentence
    if normalized == "p":
        return pause_paragraph
    match = BREAK_TIME_RE.match(normalized)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "ms":
        return value / 1000.0
    return value


def _scan_break_markers(
    text: str,
    *,
    pause_none: float,
    pause_weak: float,
    pause_clause: float,
    pause_sentence: float,
    pause_paragraph: float,
) -> list[float]:
    durations: list[float] = []
    for match in BREAK_MARKER_RE.finditer(text):
        duration = _break_duration_from_token(
            match.group("token"),
            pause_none=pause_none,
            pause_weak=pause_weak,
            pause_clause=pause_clause,
            pause_sentence=pause_sentence,
            pause_paragraph=pause_paragraph,
        )
        if duration is not None:
            durations.append(duration)
    return durations


def _collect_pause_durations(segments: list[SSMDSegment]) -> list[float]:
    durations: list[float] = []
    for segment in segments:
        if segment.pause_before > 0:
            durations.append(segment.pause_before)
        if segment.pause_after > 0:
            durations.append(segment.pause_after)
    return durations


def _breaks_satisfied(
    expected: list[float], actual: list[float], tolerance: float = 1e-6
) -> bool:
    remaining = list(actual)
    for target in expected:
        match_index = next(
            (
                idx
                for idx, value in enumerate(remaining)
                if abs(value - target) <= tolerance
            ),
            None,
        )
        if match_index is None:
            return False
        remaining.pop(match_index)
    return True


def _fallback_segments_from_chunk(
    chunk: str, paragraph_idx: int, sentence_idx: int
) -> tuple[list[SSMDSegment], int, int]:
    segments: list[SSMDSegment] = []
    paragraphs = re.split(r"\n\s*\n", chunk)
    for idx, paragraph in enumerate(paragraphs):
        paragraph_text = paragraph.strip()
        if paragraph_text:
            segments.append(
                SSMDSegment(
                    text=paragraph_text,
                    paragraph=paragraph_idx,
                    sentence=sentence_idx,
                )
            )
            sentence_idx += 1
        if idx < len(paragraphs) - 1:
            paragraph_idx += 1
    return segments, paragraph_idx, sentence_idx


def _fallback_break_segments(
    text: str,
    *,
    pause_none: float,
    pause_weak: float,
    pause_clause: float,
    pause_sentence: float,
    pause_paragraph: float,
) -> tuple[float, list[SSMDSegment]]:
    segments: list[SSMDSegment] = []
    initial_pause = 0.0
    paragraph_idx = 0
    sentence_idx = 0
    cursor = 0
    for match in BREAK_MARKER_RE.finditer(text):
        chunk = text[cursor : match.start()]
        duration = _break_duration_from_token(
            match.group("token"),
            pause_none=pause_none,
            pause_weak=pause_weak,
            pause_clause=pause_clause,
            pause_sentence=pause_sentence,
            pause_paragraph=pause_paragraph,
        )
        duration = duration or 0.0
        if chunk.strip():
            chunk_segments, paragraph_idx, sentence_idx = _fallback_segments_from_chunk(
                chunk, paragraph_idx, sentence_idx
            )
            if chunk_segments:
                chunk_segments[-1].pause_after = max(
                    chunk_segments[-1].pause_after, duration
                )
                segments.extend(chunk_segments)
            elif segments:
                segments[-1].pause_after = max(segments[-1].pause_after, duration)
            else:
                initial_pause = max(initial_pause, duration)
        else:
            if segments:
                segments[-1].pause_after = max(segments[-1].pause_after, duration)
            else:
                initial_pause = max(initial_pause, duration)
        cursor = match.end()

    tail = text[cursor:]
    if tail.strip():
        tail_segments, paragraph_idx, sentence_idx = _fallback_segments_from_chunk(
            tail, paragraph_idx, sentence_idx
        )
        segments.extend(tail_segments)
    return initial_pause, segments


def _build_segments_from_paragraphs(
    paragraphs: Sequence[Any],
    *,
    lang: str,
    pause_none: float,
    pause_weak: float,
    pause_clause: float,
    pause_sentence: float,
    pause_paragraph: float,
) -> list[SSMDSegment]:
    pykokoro_segments: list[SSMDSegment] = []
    sentence_counter = 0

    for paragraph_idx, paragraph in enumerate(paragraphs):
        for sentence in paragraph.sentences:
            # Extract voice context for this sentence
            voice_metadata = SSMDMetadata()
            if sentence.voice:
                voice_metadata.voice_name = sentence.voice.name
                voice_metadata.voice_language = sentence.voice.language
                voice_metadata.voice_gender = sentence.voice.gender
                voice_metadata.voice_variant = (
                    str(sentence.voice.variant) if sentence.voice.variant else None
                )

            resolved_paragraph = getattr(sentence, "paragraph_index", paragraph_idx)
            resolved_sentence = getattr(sentence, "sentence_index", None)
            if resolved_sentence is None:
                resolved_sentence = sentence_counter
                sentence_counter += 1
            else:
                sentence_counter = max(sentence_counter, resolved_sentence + 1)

            # Process each segment in sentence
            for seg_idx, ssmd_seg in enumerate(sentence.segments):
                # Determine language for this segment
                # Priority: segment lang > sentence lang > default
                segment_lang = ssmd_seg.language or lang

                # Map SSMD segment to PyKokoro metadata
                seg_text, metadata = _map_ssmd_segment_to_metadata(
                    ssmd_seg, segment_lang
                )

                # Apply voice context if segment doesn't have its own voice
                if not metadata.voice_name and voice_metadata.voice_name:
                    metadata.voice_name = voice_metadata.voice_name
                    metadata.voice_language = voice_metadata.voice_language
                    metadata.voice_gender = voice_metadata.voice_gender
                    metadata.voice_variant = voice_metadata.voice_variant

                # Calculate pause before this segment (for headings)
                pause_before = 0.0

                # Check for breaks before this segment
                if ssmd_seg.breaks_before:
                    # Use the last break if multiple
                    last_break = ssmd_seg.breaks_before[-1]
                    pause_before = _convert_break_strength_to_duration(
                        last_break.strength,
                        last_break.time,
                        pause_none=pause_none,
                        pause_weak=pause_weak,
                        pause_clause=pause_clause,
                        pause_sentence=pause_sentence,
                        pause_paragraph=pause_paragraph,
                    )

                # Calculate pause after this segment
                pause_after = 0.0

                # Check for breaks after this segment
                if ssmd_seg.breaks_after:
                    # Use the last break if multiple
                    last_break = ssmd_seg.breaks_after[-1]
                    pause_after = _convert_break_strength_to_duration(
                        last_break.strength,
                        last_break.time,
                        pause_none=pause_none,
                        pause_weak=pause_weak,
                        pause_clause=pause_clause,
                        pause_sentence=pause_sentence,
                        pause_paragraph=pause_paragraph,
                    )

                # If this is the last segment in the sentence,
                # check sentence-level breaks
                if seg_idx == len(sentence.segments) - 1 and sentence.breaks_after:
                    last_break = sentence.breaks_after[-1]
                    sentence_pause = _convert_break_strength_to_duration(
                        last_break.strength,
                        last_break.time,
                        pause_none=pause_none,
                        pause_weak=pause_weak,
                        pause_clause=pause_clause,
                        pause_sentence=pause_sentence,
                        pause_paragraph=pause_paragraph,
                    )
                    pause_after = max(pause_after, sentence_pause)

                # Create PyKokoro SSMDSegment with paragraph tracking
                pykokoro_segments.append(
                    SSMDSegment(
                        text=seg_text,
                        pause_before=pause_before,
                        pause_after=pause_after,
                        metadata=metadata,
                        paragraph=resolved_paragraph,
                        sentence=resolved_sentence,
                    )
                )
    return pykokoro_segments


def _parse_break_time(time: str) -> float | None:
    match = BREAK_TIME_RE.match(time)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "ms":
        return value / 1000.0
    return value


def _map_ssmd_segment_to_metadata(
    ssmd_seg: SSMDParserSegment,
    lang: str = "en-us",
) -> tuple[str, SSMDMetadata]:
    """Map SSMD parser segment to PyKokoro metadata.

    Args:
        ssmd_seg: SSMDSegment from SSMD parser
        lang: Language code for say-as normalization

    Returns:
        Tuple of (text, metadata)
    """
    from .say_as import normalize_say_as

    metadata = SSMDMetadata()

    # Handle text transformations (priority order: audio >
    # say-as > sub > phoneme > text)
    text = ssmd_seg.text

    # Audio segments - store metadata and use alt_text fallback
    if ssmd_seg.audio:
        metadata.audio_src = ssmd_seg.audio.src
        metadata.audio_alt_text = ssmd_seg.audio.alt_text
        # Use alt_text for phonemization fallback
        text = ssmd_seg.audio.alt_text or ""
    elif ssmd_seg.say_as:
        # Store say-as metadata
        metadata.say_as_interpret = ssmd_seg.say_as.interpret_as
        metadata.say_as_format = ssmd_seg.say_as.format
        metadata.say_as_detail = ssmd_seg.say_as.detail

        # Normalize text based on interpret-as type
        text = normalize_say_as(
            text,
            interpret_as=ssmd_seg.say_as.interpret_as,
            lang=lang,
            format_str=ssmd_seg.say_as.format,
            detail=ssmd_seg.say_as.detail,
        )
    elif ssmd_seg.substitution:
        text = ssmd_seg.substitution
        metadata.substitution = ssmd_seg.substitution
    elif ssmd_seg.phoneme:
        # Access phoneme string from PhonemeAttrs object
        # ssmd_seg.phoneme has .ph (phoneme string) and .alphabet ("ipa" or "x-sampa")
        metadata.phonemes = ssmd_seg.phoneme.ph
        # Keep original text for display, phoneme will override during synthesis

    # Emphasis - SSMD supports: True, "moderate", "strong", "reduced", "none"
    if ssmd_seg.emphasis:
        if isinstance(ssmd_seg.emphasis, bool):
            metadata.emphasis = "moderate"  # True maps to moderate (default)
        elif isinstance(ssmd_seg.emphasis, str):
            metadata.emphasis = ssmd_seg.emphasis  # Use explicit level

    # Language
    if ssmd_seg.language:
        metadata.language = ssmd_seg.language

    # Voice (inline annotations)
    if ssmd_seg.voice:
        metadata.voice_name = ssmd_seg.voice.name
        metadata.voice_language = ssmd_seg.voice.language
        metadata.voice_gender = ssmd_seg.voice.gender
        metadata.voice_variant = (
            str(ssmd_seg.voice.variant) if ssmd_seg.voice.variant is not None else None
        )

    # Prosody
    if ssmd_seg.prosody:
        metadata.prosody_volume = ssmd_seg.prosody.volume
        metadata.prosody_rate = ssmd_seg.prosody.rate
        metadata.prosody_pitch = ssmd_seg.prosody.pitch

    # Markers (from marks_before and marks_after)
    markers = []
    if ssmd_seg.marks_before:
        markers.extend(ssmd_seg.marks_before)
    if ssmd_seg.marks_after:
        markers.extend(ssmd_seg.marks_after)
    if markers:
        metadata.markers = markers

    return text, metadata


def parse_ssmd_to_segments(
    text: str,
    lang: str = "en-us",
    pause_none: float = DEFAULT_PAUSE_NONE,
    pause_weak: float = DEFAULT_PAUSE_WEAK,
    pause_clause: float = DEFAULT_PAUSE_CLAUSE,
    pause_sentence: float = DEFAULT_PAUSE_SENTENCE,
    pause_paragraph: float = DEFAULT_PAUSE_PARAGRAPH,
    model_size: str | None = None,
    use_spacy: bool | None = None,
    heading_levels: dict | None = None,
    parse_yaml_header: bool = False,
) -> tuple[float, list[SSMDSegment]]:
    """Parse SSMD markup and convert to segments with metadata.

    This function uses SSMD's parse_paragraphs() API to extract structured data
    and maps it to PyKokoro's internal segment representation.

    Features supported:
    - Text segments with substitutions applied
    - Pause durations from break markers (...c, ...s, ...p, ...500ms)
    - Metadata (emphasis, prosody, language, phonemes, voice, etc.)
    - Voice markers (@voice: name) with proper propagation
    - Text transformations (say-as, substitution, phoneme, audio)
    - Markers (@marker_name)
    - Audio segments ([audio](file.mp3))

    Args:
        text: Input text with SSMD markup
        lang: Default language code
        pause_none: Duration for 'none' break strength in seconds
        pause_weak: Duration for 'weak' break strength in seconds
        pause_clause: Duration for 'medium' break strength in seconds
        pause_sentence: Duration for 'strong' break strength in seconds
        pause_paragraph: Duration for 'x-strong' break strength in seconds
        model_size: Size of spacy model for sentence detection ('sm', 'md', 'lg')
        use_spacy: Force use of spacy for sentence detection
        heading_levels: Custom heading configurations (overrides default)
        parse_yaml_header: If True, parse YAML front matter and apply config

    Returns:
        Tuple of (initial_pause, segments) where segments is a list of SSMDSegment

    Example:
        >>> segments = parse_ssmd_to_segments(
        ...     "Hello ...c *important* ...s [Bonjour](fr)",
        ... )
        >>> segments = parse_ssmd_to_segments(
        ...     "@voice: sarah\\nHello!\\n\\n@voice: michael\\nWorld!",
        ... )
    """
    text = _normalize_div_directives(text)

    # Convert shorthand language annotations like [Bonjour](fr)
    text = LANG_SHORTHAND_RE.sub(r'[\1]{lang="\2"}', text)
    expected_breaks = _scan_break_markers(
        text,
        pause_none=pause_none,
        pause_weak=pause_weak,
        pause_clause=pause_clause,
        pause_sentence=pause_sentence,
        pause_paragraph=pause_paragraph,
    )

    TTSCapabilities, parse_paragraphs = _load_ssmd()

    # Use SSMD's parse_paragraphs to get structured data
    # Enable heading detection for markdown-style headings (# ## ###)
    caps = TTSCapabilities()
    caps.heading_emphasis = True

    def parse_with_spacy(use_spacy_override: bool | None) -> list[Any]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "Importing 'parser.split_arg_string' is deprecated, "
                    "it will only be available in 'shell_completion' in Click 9.0."
                ),
                category=DeprecationWarning,
            )
            return parse_paragraphs(
                text,
                sentence_detection=True,
                language=lang,
                capabilities=caps,
                model_size=model_size,
                use_spacy=use_spacy_override,
                heading_levels=heading_levels,
                parse_yaml_header=parse_yaml_header,
            )

    paragraphs = parse_with_spacy(use_spacy)
    if not paragraphs:
        return 0.0, []

    pykokoro_segments = _build_segments_from_paragraphs(
        paragraphs,
        lang=lang,
        pause_none=pause_none,
        pause_weak=pause_weak,
        pause_clause=pause_clause,
        pause_sentence=pause_sentence,
        pause_paragraph=pause_paragraph,
    )
    expected_nonzero = [duration for duration in expected_breaks if duration > 0]
    if expected_nonzero:
        actual_durations = _collect_pause_durations(pykokoro_segments)
        if not _breaks_satisfied(expected_nonzero, actual_durations):
            if use_spacy is not False:
                fallback_paragraphs = parse_with_spacy(False)
                if fallback_paragraphs:
                    fallback_segments = _build_segments_from_paragraphs(
                        fallback_paragraphs,
                        lang=lang,
                        pause_none=pause_none,
                        pause_weak=pause_weak,
                        pause_clause=pause_clause,
                        pause_sentence=pause_sentence,
                        pause_paragraph=pause_paragraph,
                    )
                    fallback_durations = _collect_pause_durations(fallback_segments)
                    if _breaks_satisfied(expected_nonzero, fallback_durations):
                        return 0.0, fallback_segments
            logger.warning(
                "SSMD parser dropped explicit breaks; using fallback parser."
            )
            return _fallback_break_segments(
                text,
                pause_none=pause_none,
                pause_weak=pause_weak,
                pause_clause=pause_clause,
                pause_sentence=pause_sentence,
                pause_paragraph=pause_paragraph,
            )

    return 0.0, pykokoro_segments
