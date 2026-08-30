"""Tests for SSMD (Speech Synthesis Markdown) integration in pykokoro."""


class TestSSMDDetection:
    """Tests for SSMD markup detection."""

    def test_has_ssmd_markup_breaks(self):
        """Test detection of SSMD break markers."""
        from pykokoro.ssmd_parser import has_ssmd_markup

        assert has_ssmd_markup("Hello ...c world")
        assert has_ssmd_markup("Test ...500ms pause")
        assert has_ssmd_markup("Wait ...2s")
        assert has_ssmd_markup("Wait ...0.5s")
        assert not has_ssmd_markup("Hello... world")  # Bare ellipsis
        assert not has_ssmd_markup("Plain text")

    def test_has_ssmd_markup_emphasis(self):
        """Test detection of emphasis markers."""
        from pykokoro.ssmd_parser import has_ssmd_markup

        assert has_ssmd_markup("This is *important*")
        assert has_ssmd_markup("This is **very important**")
        assert not has_ssmd_markup("This has * asterisks * but not emphasis")

    def test_has_ssmd_markup_annotations(self):
        """Test detection of annotations."""
        from pykokoro.ssmd_parser import has_ssmd_markup

        assert has_ssmd_markup('[Bonjour]{lang="fr"}')
        assert has_ssmd_markup('[Bonjour]{voice-lang="fr-FR"}')
        assert has_ssmd_markup("[Bonjour]{lang=fr}")
        assert has_ssmd_markup(r'[quote]{sub="say \\"hello\\""}')
        assert has_ssmd_markup('[Bonjour]{ph="abc"}')
        assert has_ssmd_markup('[]{src="beep.mp3"}')
        assert not has_ssmd_markup("[Bonjour](fr)")
        assert not has_ssmd_markup("No markup here")

    def test_has_ssmd_markup_markers(self):
        """Test detection of markers."""
        from pykokoro.ssmd_parser import has_ssmd_markup

        assert has_ssmd_markup("Text with @marker")
        assert not has_ssmd_markup("@voice: af_sarah")
        assert not has_ssmd_markup("Email@example.com")  # @ in email
        assert not has_ssmd_markup("Plain text")


class TestSSMDSegmentConversion:
    """Tests for SSMD segment parsing and conversion."""

    def test_parse_ssmd_to_segments_basic(self):
        """Test basic SSMD parsing to segments."""
        from pykokoro.ssmd_parser import parse_ssmd_to_segments

        initial, segments = parse_ssmd_to_segments(
            "Hello ...c world",
        )

        assert initial == 0.0
        assert len(segments) == 2
        assert segments[0].text == "Hello"
        assert segments[0].pause_after == 0.3
        assert segments[1].text == "world"
        assert segments[1].pause_after == 0.0

    def test_parse_ssmd_to_segments_with_markup(self):
        """Test SSMD parsing strips markup from text."""
        from pykokoro.ssmd_parser import parse_ssmd_to_segments

        _initial, segments = parse_ssmd_to_segments(
            "This is *important* ...s Really!",
        )

        # SSMD splits on emphasis markers, creating segments for each part
        assert len(segments) == 3
        # First segment: text before emphasis
        assert segments[0].text == "This is"
        # Second segment: emphasized text (markup stripped)
        assert segments[1].text == "important"
        assert "*" not in segments[1].text  # Markup removed
        # Third segment: text after pause
        assert "Really!" in segments[2].text

    def test_parse_ssmd_to_segments_without_markup(self):
        """Test SSMD parsing strips markup from text."""
        from pykokoro.ssmd_parser import parse_ssmd_to_segments

        _initial, segments = parse_ssmd_to_segments(
            "Hello this is great. Really!",
        )

        assert len(segments) == 2
        # Markup should be stripped from text
        assert "Hello this is great." in segments[0].text
        assert "Really!" in segments[1].text


class TestSSMDNativeSyntax:
    """Tests for the native SSMD 0.7 annotation and directive contract."""

    def test_legacy_markdown_link_syntax_is_literal_text(self):
        """Do not reinterpret ordinary Markdown links as SSMD metadata."""
        from pykokoro.ssmd_parser import parse_ssmd_to_segments

        _initial, segments = parse_ssmd_to_segments("[Bonjour](fr)")

        assert len(segments) == 1
        assert segments[0].text == "[Bonjour](fr)"
        assert segments[0].metadata.language is None

    def test_div_context_is_inherited_and_inline_values_override(self):
        """Sentence-level div attributes flow into each adapted segment."""
        from pykokoro.ssmd_parser import parse_ssmd_to_segments

        text = (
            '<div lang="fr" volume="loud" rate="slow" pitch="high" '
            'voice="af_sarah" voice-lang="fr-FR">\n'
            '[Bonjour]{pitch="low" gender="female" variant="2"}\n'
            "</div>"
        )
        _initial, segments = parse_ssmd_to_segments(text)

        assert len(segments) == 1
        metadata = segments[0].metadata
        assert metadata.language == "fr"
        assert metadata.prosody_volume == "loud"
        assert metadata.prosody_rate == "slow"
        assert metadata.prosody_pitch == "low"
        assert metadata.voice_name == "af_sarah"
        assert metadata.voice_language == "fr-FR"
        assert metadata.voice_gender == "female"
        assert metadata.voice_variant == "2"

    def test_xsampa_override_uses_ssmd_converter(self, monkeypatch):
        """Convert SSMD X-SAMPA before passing a phoneme override downstream."""
        from ssmd import segment as ssmd_segment

        from pykokoro.ssmd_parser import parse_ssmd_to_segments

        monkeypatch.setattr(
            ssmd_segment,
            "xsampa_to_ipa",
            lambda value: f"ipa:{value}",
        )

        _initial, segments = parse_ssmd_to_segments('[dictionary]{sampa="dIkS@n@ri"}')

        assert segments[0].metadata.phonemes == "ipa:dIkS@n@ri"

    def test_adjacent_breaks_are_cumulative(self):
        """Preserve multiple SSMD breaks at the same segment boundary."""
        import pytest

        from pykokoro.ssmd_parser import parse_ssmd_to_segments

        _initial, segments = parse_ssmd_to_segments("Hello ...c ...s world")

        assert segments[0].pause_after == pytest.approx(0.9)


class TestSSMDMetadata:
    """Tests for SSMD metadata structures."""

    def test_ssmd_metadata_creation(self):
        """Test creating SSMD metadata."""
        from pykokoro.ssmd_parser import SSMDMetadata

        metadata = SSMDMetadata(
            emphasis="strong",
            language="fr",
            phonemes="bɔ̃ʒuʁ",
        )

        assert metadata.emphasis == "strong"
        assert metadata.language == "fr"
        assert metadata.phonemes == "bɔ̃ʒuʁ"

    def test_ssmd_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        from pykokoro.ssmd_parser import SSMDMetadata

        metadata = SSMDMetadata(emphasis="moderate")
        data = metadata.to_dict()

        assert isinstance(data, dict)
        assert data["emphasis"] == "moderate"
        assert "prosody_volume" in data
        assert "language" in data

    def test_ssmd_segment_creation(self):
        """Test creating SSMD segment."""
        from pykokoro.ssmd_parser import SSMDMetadata, SSMDSegment

        segment = SSMDSegment(
            text="Hello",
            pause_after=0.5,
            metadata=SSMDMetadata(emphasis="strong"),
        )

        assert segment.text == "Hello"
        assert segment.pause_after == 0.5
        assert segment.metadata.emphasis == "strong"


class TestSSMDBreakParsing:
    """Tests for SSMD break duration parsing."""

    def test_break_time_parsing_formats(self):
        from pykokoro.ssmd_parser import _convert_break_strength_to_duration

        assert _convert_break_strength_to_duration(None, "500ms") == 0.5
        assert _convert_break_strength_to_duration(None, "2s") == 2.0
        assert _convert_break_strength_to_duration(None, "0.5s") == 0.5
        assert _convert_break_strength_to_duration(None, "500 ms") == 0.5
        assert _convert_break_strength_to_duration(None, "0.5 s") == 0.5
        assert _convert_break_strength_to_duration(None, "1.25s") == 1.25

    def test_break_time_invalid_falls_back_to_strength(self):
        from pykokoro.ssmd_parser import _convert_break_strength_to_duration

        assert _convert_break_strength_to_duration("weak", "fast") == 0.15
        assert _convert_break_strength_to_duration(None, "fast") == 0.0


class TestSSMDAudioSegments:
    """Tests for SSMD audio segment behavior."""

    def test_audio_segment_uses_alt_text(self):
        from pykokoro.ssmd_parser import SSMDMetadata, SSMDSegment
        from pykokoro.stages.doc_parsers.ssmd import SsmdDocumentParser
        from pykokoro.types import Trace

        parser = SsmdDocumentParser()
        trace = Trace()
        segment = SSMDSegment(
            text="Hello",
            metadata=SSMDMetadata(audio_src="clip.wav", audio_alt_text="Hello"),
        )

        clean_text, spans, boundaries, doc_segments = parser._build_document([segment], 0.0, trace)

        assert clean_text == "Hello"
        assert spans[0].attrs["audio_src"] == "clip.wav"
        assert "alt_text" in "".join(trace.warnings)
        assert boundaries == []
        assert len(doc_segments) == 1

    def test_audio_segment_without_alt_text_is_skipped(self):
        from pykokoro.ssmd_parser import SSMDMetadata, SSMDSegment
        from pykokoro.stages.doc_parsers.ssmd import SsmdDocumentParser
        from pykokoro.types import Trace

        parser = SsmdDocumentParser()
        trace = Trace()
        segment = SSMDSegment(
            text="",
            metadata=SSMDMetadata(audio_src="clip.wav", audio_alt_text=""),
        )

        clean_text, spans, boundaries, doc_segments = parser._build_document([segment], 0.0, trace)

        assert clean_text == ""
        assert spans == []
        assert boundaries == []
        assert "no alt_text" in "".join(trace.warnings)
        assert doc_segments == []


def test_ssmd_parser_keeps_ellipsis_text():
    from pykokoro.generation_config import GenerationConfig
    from pykokoro.pipeline_config import PipelineConfig
    from pykokoro.stages.doc_parsers.ssmd import SsmdDocumentParser
    from pykokoro.tokenizer import TokenizerConfig
    from pykokoro.types import Trace

    text = (
        "\"Don't I?\" He smirked. \"I'd've thought you'd've figured "
        "it out by now... People like you—you're\n"
        "all the same. You won't listen, you can't comprehend, and you "
        "shouldn't even bother trying!\""
    )
    cfg = PipelineConfig(
        generation=GenerationConfig(lang="en-us"),
        tokenizer_config=TokenizerConfig(spacy_model_size="sm"),
    )
    doc = SsmdDocumentParser().parse(text, cfg, Trace())

    assert "People like you" in doc.clean_text
    assert not [segment for segment in doc.segments if segment.sentence_idx is not None]
    assert doc.segments[0].text == doc.clean_text


class TestSSMDVoiceSwitching:
    """Tests for per-segment voice switching functionality."""

    def test_parse_ssmd_with_voice_creates_metadata(self):
        """Test that parsing SSMD text with voice creates proper metadata."""
        from pykokoro.ssmd_parser import parse_ssmd_to_segments

        # Test 1: Block directives (<div voice="name">)
        text = '<div voice="af_sarah">Hello ...s</div>\n\n<div voice="am_michael">World</div>'
        _initial_pause, segments = parse_ssmd_to_segments(text)

        assert any(seg.metadata.voice_name == "af_sarah" for seg in segments)
        assert any(seg.metadata.voice_name == "am_michael" for seg in segments)

    def test_parse_ssmd_with_inline_voice_annotations(self):
        """Test that inline voice annotations work."""
        from pykokoro.ssmd_parser import parse_ssmd_to_segments

        # Test 2: Inline voice annotations ([text]{voice="name"})
        text = "[Hello]{voice='af_sarah'} ...s\n\n[World]{voice='am_michael'}"
        _initial_pause, segments = parse_ssmd_to_segments(text)

        assert any(seg.metadata.voice_name == "af_sarah" for seg in segments)
        assert any(seg.metadata.voice_name == "am_michael" for seg in segments)

    def test_voice_resolver_called_for_segment_with_voice(self):
        """Test AudioGenerator calls voice_resolver for voice metadata."""
        from unittest.mock import Mock

        import numpy as np

        from pykokoro.audio_generator import AudioGenerator
        from pykokoro.tokenizer import create_tokenizer
        from pykokoro.types import PhonemeSegment

        tokenizer = create_tokenizer()

        # Create mock session
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input_ids")]
        mock_session.run.return_value = [np.zeros((1, 100), dtype=np.float32)]

        generator = AudioGenerator(mock_session, tokenizer)

        # Create segments with voice metadata
        segments = [
            PhonemeSegment(
                id="seg0_ph0",
                segment_id="seg_0",
                phoneme_id=0,
                text="Hello",
                phonemes="hɛˈloʊ",
                tokens=[1, 2, 3],
                char_start=0,
                char_end=5,
                paragraph_idx=0,
                sentence_idx=0,
                clause_idx=0,
                ssmd_metadata={"voice_name": "af_sarah"},
            ),
            PhonemeSegment(
                id="seg1_ph0",
                segment_id="seg_1",
                phoneme_id=0,
                text="World",
                phonemes="wɝld",
                tokens=[4, 5],
                char_start=6,
                char_end=11,
                paragraph_idx=0,
                sentence_idx=0,
                clause_idx=0,
                ssmd_metadata={"voice_name": "am_michael"},
            ),
        ]

        # Mock voice resolver
        voice_calls = []

        def mock_voice_resolver(voice_name: str) -> np.ndarray:
            voice_calls.append(voice_name)
            return np.zeros((512, 1, 256), dtype=np.float32)

        default_voice = np.zeros((512, 1, 256), dtype=np.float32)

        # Generate with voice resolver
        audio = generator.generate_from_segments(
            segments,
            default_voice,
            speed=1.0,
            trim_silence=False,
            voice_resolver=mock_voice_resolver,
        )

        # Verify audio was generated
        assert isinstance(audio, np.ndarray)

        # Verify voice_resolver was called for each segment
        assert len(voice_calls) == 2
        assert voice_calls[0] == "af_sarah"
        assert voice_calls[1] == "am_michael"

    def test_voice_switching_without_resolver_uses_default(self):
        """Test that segments with voice metadata but no resolver use default voice."""
        from unittest.mock import Mock

        import numpy as np

        from pykokoro.audio_generator import AudioGenerator
        from pykokoro.tokenizer import create_tokenizer
        from pykokoro.types import PhonemeSegment

        tokenizer = create_tokenizer()

        # Create mock session
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input_ids")]
        mock_session.run.return_value = [np.zeros((1, 100), dtype=np.float32)]

        generator = AudioGenerator(mock_session, tokenizer)

        # Create segment with voice metadata
        segments = [
            PhonemeSegment(
                id="seg0_ph0",
                segment_id="seg_0",
                phoneme_id=0,
                text="Hello",
                phonemes="hɛˈloʊ",
                tokens=[1, 2, 3],
                char_start=0,
                char_end=5,
                paragraph_idx=0,
                sentence_idx=0,
                clause_idx=0,
                ssmd_metadata={"voice_name": "af_sarah"},
            ),
        ]

        default_voice = np.zeros((512, 1, 256), dtype=np.float32)

        # Generate WITHOUT voice resolver (should use default)
        audio = generator.generate_from_segments(
            segments,
            default_voice,
            speed=1.0,
            trim_silence=False,
            voice_resolver=None,  # No resolver
        )

        # Should succeed and use default voice
        assert isinstance(audio, np.ndarray)
