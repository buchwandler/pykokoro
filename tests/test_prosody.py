"""Tests for prosody audio processing."""

import numpy as np
import pytest

from pykokoro.prosody import (
    apply_pitch,
    apply_prosody,
    apply_rate,
    apply_volume,
    parse_pitch,
    parse_rate,
    parse_volume,
)


class TestVolumeParsing:
    """Tests for volume parsing."""

    def test_parse_volume_absolute_values(self):
        """Test parsing absolute volume values."""
        assert parse_volume("silent") == -float("inf")
        assert parse_volume("x-soft") == -12.0
        assert parse_volume("soft") == -6.0
        assert parse_volume("medium") == 0.0
        assert parse_volume("loud") == 6.0
        assert parse_volume("x-loud") == 12.0
        assert parse_volume("default") == 0.0

    def test_parse_volume_case_insensitive(self):
        """Test that volume parsing is case insensitive."""
        assert parse_volume("LOUD") == 6.0
        assert parse_volume("Loud") == 6.0
        assert parse_volume("LoUd") == 6.0

    def test_parse_volume_db_notation(self):
        """Test parsing dB notation."""
        assert parse_volume("+6dB") == 6.0
        assert parse_volume("-3dB") == -3.0
        assert parse_volume("6dB") == 6.0
        assert parse_volume("0dB") == 0.0
        assert parse_volume("+12.5dB") == 12.5

    def test_parse_volume_percentage_relative(self):
        """Test parsing relative percentage values."""
        # +20% means 20% louder: 20*log10(1.2) ≈ 1.58dB
        result = parse_volume("+20%")
        assert pytest.approx(result, abs=0.01) == 1.58

        # -10% means 10% quieter: 20*log10(0.9) ≈ -0.92dB
        result = parse_volume("-10%")
        assert pytest.approx(result, abs=0.01) == -0.92

    def test_parse_volume_percentage_absolute(self):
        """Test parsing absolute percentage values."""
        # 120% means 20% louder than baseline
        result = parse_volume("120%")
        assert pytest.approx(result, abs=0.01) == 1.58

        # 90% means 10% quieter than baseline
        result = parse_volume("90%")
        assert pytest.approx(result, abs=0.01) == -0.92

    def test_parse_volume_invalid_format(self):
        """Test that invalid formats raise ValueError."""
        with pytest.raises(ValueError, match="Invalid volume format"):
            parse_volume("invalid")
        with pytest.raises(ValueError, match="Invalid volume format"):
            parse_volume("123")
        with pytest.raises(ValueError, match="Invalid volume format"):
            parse_volume("abc%")


class TestRateParsing:
    """Tests for rate parsing."""

    def test_parse_rate_absolute_values(self):
        """Test parsing absolute rate values."""
        assert parse_rate("x-slow") == 0.5
        assert parse_rate("slow") == 0.75
        assert parse_rate("medium") == 1.0
        assert parse_rate("fast") == 1.25
        assert parse_rate("x-fast") == 1.5
        assert parse_rate("default") == 1.0

    def test_parse_rate_case_insensitive(self):
        """Test that rate parsing is case insensitive."""
        assert parse_rate("FAST") == 1.25
        assert parse_rate("Fast") == 1.25
        assert parse_rate("FaSt") == 1.25

    def test_parse_rate_percentage_relative(self):
        """Test parsing relative percentage values."""
        assert parse_rate("+20%") == 1.2
        assert parse_rate("-10%") == 0.9
        assert parse_rate("+50%") == 1.5

    def test_parse_rate_percentage_absolute(self):
        """Test parsing absolute percentage values."""
        assert parse_rate("120%") == 1.2
        assert parse_rate("80%") == 0.8
        assert parse_rate("100%") == 1.0

    def test_parse_rate_invalid_format(self):
        """Test that invalid formats raise ValueError."""
        with pytest.raises(ValueError, match="Invalid rate format"):
            parse_rate("invalid")
        with pytest.raises(ValueError, match="Invalid rate format"):
            parse_rate("123")
        with pytest.raises(ValueError, match="Invalid rate format"):
            parse_rate("abc%")


class TestPitchParsing:
    """Tests for pitch parsing."""

    def test_parse_pitch_absolute_values(self):
        """Test parsing absolute pitch values."""
        assert parse_pitch("x-low") == -4.0
        assert parse_pitch("low") == -2.0
        assert parse_pitch("medium") == 0.0
        assert parse_pitch("high") == 2.0
        assert parse_pitch("x-high") == 4.0
        assert parse_pitch("default") == 0.0

    def test_parse_pitch_case_insensitive(self):
        """Test that pitch parsing is case insensitive."""
        assert parse_pitch("HIGH") == 2.0
        assert parse_pitch("High") == 2.0
        assert parse_pitch("HiGh") == 2.0

    def test_parse_pitch_semitones(self):
        """Test parsing semitone notation."""
        assert parse_pitch("+2st") == 2.0
        assert parse_pitch("-1.5st") == -1.5
        assert parse_pitch("3st") == 3.0
        assert parse_pitch("0st") == 0.0

    def test_parse_pitch_percentage(self):
        """Test parsing percentage values."""
        # +10% means 10% higher pitch: 12*log2(1.1) ≈ 1.65 semitones
        result = parse_pitch("+10%")
        assert pytest.approx(result, abs=0.05) == 1.65

        # -10% means 10% lower pitch: 12*log2(0.9) ≈ -1.82 semitones
        result = parse_pitch("-10%")
        assert pytest.approx(result, abs=0.05) == -1.82

    def test_parse_pitch_invalid_format(self):
        """Test that invalid formats raise ValueError."""
        with pytest.raises(ValueError, match="Invalid pitch format"):
            parse_pitch("invalid")
        with pytest.raises(ValueError, match="Invalid pitch format"):
            parse_pitch("123")
        with pytest.raises(ValueError, match="Invalid pitch format"):
            parse_pitch("abc%")


class TestVolumeApplication:
    """Tests for volume application."""

    def test_apply_volume_increases(self):
        """Test that volume increase amplifies audio."""
        audio = np.ones(100, dtype=np.float32)
        result = apply_volume(audio, "+6dB")

        # +6dB = 10^(6/20) ≈ 2.0 amplitude multiplier
        expected_multiplier = 10 ** (6 / 20)
        assert pytest.approx(result[0], abs=0.01) == expected_multiplier

    def test_apply_volume_decreases(self):
        """Test that volume decrease attenuates audio."""
        audio = np.ones(100, dtype=np.float32)
        result = apply_volume(audio, "-6dB")

        # -6dB = 10^(-6/20) ≈ 0.5 amplitude multiplier
        expected_multiplier = 10 ** (-6 / 20)
        assert pytest.approx(result[0], abs=0.01) == expected_multiplier

    def test_apply_volume_silent(self):
        """Test that silent produces zero audio."""
        audio = np.ones(100, dtype=np.float32)
        result = apply_volume(audio, "silent")
        assert np.all(result == 0)

    def test_apply_volume_medium_no_change(self):
        """Test that medium volume doesn't change audio."""
        audio = np.random.randn(100).astype(np.float32)
        result = apply_volume(audio, "medium")
        np.testing.assert_array_almost_equal(result, audio)

    def test_apply_volume_invalid_returns_original(self):
        """Test that invalid volume returns original audio."""
        audio = np.random.randn(100).astype(np.float32)
        result = apply_volume(audio, "invalid_value")
        np.testing.assert_array_equal(result, audio)


class TestPitchApplication:
    """Tests for the required AudioSig pitch operation."""

    def test_apply_pitch_zero_no_change(self):
        """Test that zero pitch shift doesn't change audio significantly."""
        audio = np.random.randn(1000).astype(np.float32)
        result = apply_pitch(audio, "medium", 24000)

        # Should be very similar (allow small numerical differences)
        correlation = np.corrcoef(audio, result)[0, 1]
        assert correlation > 0.99

    def test_apply_pitch_changes_length_preservation(self):
        """Test that pitch shift preserves audio length."""
        audio = np.random.randn(1000).astype(np.float32)
        result = apply_pitch(audio, "+2st", 24000)

        assert len(result) == len(audio)

    def test_apply_pitch_invalid_returns_original(self):
        """Test that invalid pitch returns original audio."""
        audio = np.random.randn(100).astype(np.float32)
        result = apply_pitch(audio, "invalid_value", 24000)
        np.testing.assert_array_equal(result, audio)


class TestRateApplication:
    """Tests for the required AudioSig time-stretch operation."""

    def test_apply_rate_faster_shortens(self):
        """Test that faster rate shortens audio."""
        audio = np.random.randn(1000).astype(np.float32)
        result = apply_rate(audio, "fast")  # 1.25x speed

        assert len(result) == round(len(audio) / 1.25)

    def test_apply_rate_slower_lengthens(self):
        """Test that slower rate lengthens audio."""
        audio = np.random.randn(1000).astype(np.float32)
        result = apply_rate(audio, "slow")  # 0.75x speed

        assert len(result) == round(len(audio) / 0.75)

    def test_apply_rate_medium_no_change(self):
        """Test that medium rate doesn't change length significantly."""
        audio = np.random.randn(1000).astype(np.float32)
        result = apply_rate(audio, "medium")

        assert result is audio

    def test_apply_rate_invalid_returns_original(self):
        """Test that invalid rate returns original audio."""
        audio = np.random.randn(100).astype(np.float32)
        result = apply_rate(audio, "invalid_value")
        np.testing.assert_array_equal(result, audio)


class TestProsodyApplication:
    """Tests for combined prosody application."""

    def test_apply_prosody_all_parameters(self):
        """Test applying all prosody parameters together."""
        audio = np.random.randn(1000).astype(np.float32)
        result = apply_prosody(audio, sample_rate=24000, volume="+6dB", pitch="+2st", rate="fast")

        # Result should be different from input
        assert not np.array_equal(result, audio)
        # Result should not be all zeros
        assert np.any(result != 0)

    def test_apply_prosody_volume_only(self):
        """Test applying only volume."""
        audio = np.ones(100, dtype=np.float32)
        result = apply_prosody(audio, sample_rate=24000, volume="loud")

        # Should be louder (+6dB)
        assert np.mean(np.abs(result)) > np.mean(np.abs(audio))

    def test_apply_prosody_none_parameters(self):
        """Test that None parameters don't change audio."""
        audio = np.random.randn(1000).astype(np.float32)
        result = apply_prosody(audio, sample_rate=24000)

        # Should be identical
        np.testing.assert_array_equal(result, audio)

    def test_apply_prosody_order_of_operations(self):
        """Test that prosody operations are applied in correct order.

        Order should be: pitch -> rate -> volume
        """
        audio = np.ones(1000, dtype=np.float32)

        # Apply prosody with all parameters
        result = apply_prosody(audio, sample_rate=24000, volume="loud", pitch="+2st", rate="fast")

        # Length should be affected by rate (faster = shorter)
        # but not by volume or pitch (allow reasonable tolerance)
        assert len(result) < len(audio) * 0.95


class TestProsodyAudioSigBoundary:
    """Tests for AudioSig-backed policy boundaries."""

    def test_volume_works_with_core_dependency(self):
        """Volume is available without an optional prosody extra."""
        audio = np.ones(100, dtype=np.float32)
        result = apply_volume(audio, "loud")

        # Volume should still work
        expected_multiplier = 10 ** (6 / 20)
        assert pytest.approx(result[0], abs=0.01) == expected_multiplier

    def test_pitch_preserves_dtype_and_input(self):
        audio = np.random.randn(100).astype(np.float32)
        original = audio.copy()
        result = apply_pitch(audio, "+2st", 24000)
        assert result.dtype == audio.dtype
        assert len(result) == len(audio)
        np.testing.assert_array_equal(audio, original)
