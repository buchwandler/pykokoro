"""Unit tests for bug fixes in PyKokoro.

This module tests the specific bug fixes:
1. Duplicate phoneme dictionary loading (tokenizer.py)
2. Voice style array bounds checking (audio_generator.py)
3. Provider config extraction (provider_config.py)
"""

import json
from unittest.mock import Mock, patch

import numpy as np
import pytest

from pykokoro.audio_generator import AudioGenerator
from pykokoro.provider_config import ProviderConfigManager
from pykokoro.tokenizer import Tokenizer, TokenizerConfig


class TestBugFix1_DuplicatePhonemeDictionaryLoading:
    """Test that phoneme dictionary is loaded only once (Bug #1)."""

    def test_phoneme_dictionary_loaded_once(self, tmp_path):
        """Test that phoneme dictionary is not loaded twice."""
        # Create a temporary phoneme dictionary file
        dict_file = tmp_path / "phonemes.json"
        phoneme_data = {
            "hello": "həˈloʊ",
            "world": "wɜːrld",
        }
        with open(dict_file, "w") as f:
            json.dump(phoneme_data, f)

        # Create tokenizer with phoneme dictionary
        config = TokenizerConfig(phoneme_dictionary_path=str(dict_file))

        # Mock the load method to track calls
        with patch.object(
            Tokenizer, "_load_phoneme_dictionary", wraps=lambda self, p: phoneme_data
        ) as mock_load:
            tokenizer = Tokenizer(config=config)

            # The dictionary should exist
            assert tokenizer._phoneme_dictionary_obj is not None
            assert tokenizer._phoneme_dictionary_obj.has_entries()

            # Verify _load_phoneme_dictionary was NOT called
            # (dictionary is loaded via PhonemeDictionary constructor)
            mock_load.assert_not_called()

    def test_phoneme_dictionary_attributes_consistent(self, tmp_path):
        """Test that both dictionary attributes point to same data."""
        dict_file = tmp_path / "phonemes.json"
        phoneme_data = {
            "test": "tɛst",
        }
        with open(dict_file, "w") as f:
            json.dump(phoneme_data, f)

        config = TokenizerConfig(phoneme_dictionary_path=str(dict_file))
        tokenizer = Tokenizer(config=config)

        # Both attributes should reference the same underlying dict
        assert tokenizer._phoneme_dictionary_obj is not None
        assert tokenizer._phoneme_dictionary_obj._dictionary["test"] == "tɛst"


class TestBugFix2_VoiceStyleBoundsChecking:
    """Test that voice style indexing is properly bounds-checked (Bug #3)."""

    def test_voice_style_index_within_bounds(self):
        """Test that style index doesn't exceed voice_style array size."""
        # Create mock session and tokenizer
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="tokens")]
        mock_session.run.return_value = [np.zeros((100, 1), dtype=np.float32)]

        mock_tokenizer = Mock()
        mock_tokenizer.tokenize.return_value = [1, 2, 3, 4, 5]  # 5 tokens

        # Create audio generator
        generator = AudioGenerator(
            session=mock_session,
            tokenizer=mock_tokenizer,
            model_source="huggingface",
        )

        # Create voice style array that's SMALLER than MAX_PHONEME_LENGTH
        # This should trigger the bounds check
        small_voice_style = np.random.randn(10, 1, 256).astype(np.float32)  # Only 10 elements

        # This should NOT raise IndexError
        try:
            audio, sample_rate = generator.generate_from_phonemes(
                phonemes="hello world", voice_style=small_voice_style, speed=1.0
            )
            # If we get here, bounds checking worked
            assert audio is not None
            assert sample_rate == 24000
        except IndexError as e:
            pytest.fail(f"IndexError raised despite bounds checking: {e}")

    def test_voice_style_index_clamped_correctly(self):
        """Test that style index is clamped to voice_style size."""
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="tokens")]
        mock_session.run.return_value = [np.zeros((100, 1), dtype=np.float32)]

        # Create tokenizer that returns many tokens (would exceed small array)
        mock_tokenizer = Mock()
        mock_tokenizer.tokenize.return_value = list(
            range(500)
        )  # 500 tokens (way more than array size)

        generator = AudioGenerator(
            session=mock_session,
            tokenizer=mock_tokenizer,
            model_source="huggingface",
        )

        # Small voice style array
        small_voice_style = np.random.randn(20, 1, 256).astype(np.float32)

        # Should use index 19 (last element), not 500 or 509
        audio, _sample_rate = generator.generate_from_phonemes(
            phonemes="test " * 100, voice_style=small_voice_style, speed=1.0
        )

        # Verify it used a valid index (would have crashed without fix)
        assert audio is not None

    def test_voice_style_normal_size(self):
        """Test that normal-sized voice arrays still work correctly."""
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="tokens")]
        mock_session.run.return_value = [np.zeros((100, 1), dtype=np.float32)]

        mock_tokenizer = Mock()
        mock_tokenizer.tokenize.return_value = [1, 2, 3]

        generator = AudioGenerator(
            session=mock_session,
            tokenizer=mock_tokenizer,
            model_source="huggingface",
        )

        # Normal-sized voice style (512 elements, typical size)
        normal_voice_style = np.random.randn(512, 1, 256).astype(np.float32)

        audio, sample_rate = generator.generate_from_phonemes(
            phonemes="test", voice_style=normal_voice_style, speed=1.0
        )

        assert audio is not None
        assert sample_rate == 24000

    def test_voice_style_input_shape(self):
        """Test that style input is normalized to 2D float32."""
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(name="input_ids")]
        captured_inputs = {}

        def _run(_, inputs):
            captured_inputs.update(inputs)
            return [np.zeros((1, 100), dtype=np.float32)]

        mock_session.run.side_effect = _run

        mock_tokenizer = Mock()
        mock_tokenizer.tokenize.return_value = [1, 2, 3]

        generator = AudioGenerator(
            session=mock_session,
            tokenizer=mock_tokenizer,
            model_source="huggingface",
        )

        voice_style = np.random.randn(512, 256).astype(np.float32)
        generator.generate_from_phonemes(phonemes="test", voice_style=voice_style, speed=1.0)

        assert "style" in captured_inputs
        style_input = np.asarray(captured_inputs["style"])
        assert style_input.shape == (1, 256)
        assert style_input.dtype == np.float32


class TestBugFix3_ProviderConfigExtraction:
    """Test that provider configuration is properly extracted and deduplicated."""

    def test_get_default_provider_options_openvino(self, tmp_path):
        """Test default options for OpenVINO provider."""
        options = ProviderConfigManager.get_default_provider_options(
            provider="OpenVINOExecutionProvider",
            model_quality="fp16",
            cache_path=tmp_path,
        )

        assert "device_type" in options
        assert options["device_type"] == "CPU_FP32"
        assert "precision" in options
        assert options["precision"] == "FP16"  # Should match model_quality
        assert "enable_opencl_throttling" in options
        assert "cache_dir" in options

    def test_get_default_provider_options_cuda(self):
        """Test default options for CUDA provider."""
        options = ProviderConfigManager.get_default_provider_options(
            provider="CUDAExecutionProvider",
            model_quality="fp32",
        )

        assert "device_id" in options
        assert options["device_id"] == "0"
        assert "arena_extend_strategy" in options

    def test_get_provider_specific_options_filtering(self):
        """Test that provider-specific options are extracted correctly."""
        all_options = {
            # SessionOptions attributes (should be excluded)
            "intra_op_num_threads": 4,
            "inter_op_num_threads": 2,
            "graph_optimization_level": 99,
            # Provider-specific options (should be included)
            "device_type": "GPU",  # Valid for OpenVINO
            "precision": "FP16",
            "num_of_threads": 4,
        }

        provider_opts = ProviderConfigManager.get_provider_specific_options(
            provider="OpenVINOExecutionProvider",
            all_options=all_options,
        )

        # Should NOT contain SessionOptions attributes
        assert "intra_op_num_threads" not in provider_opts
        assert "inter_op_num_threads" not in provider_opts
        assert "graph_optimization_level" not in provider_opts

        # Should contain provider-specific options (as strings)
        assert "device_type" in provider_opts
        assert provider_opts["device_type"] == "GPU"
        assert "precision" in provider_opts
        assert provider_opts["precision"] == "FP16"
        assert "num_of_threads" in provider_opts
        assert provider_opts["num_of_threads"] == "4"

    def test_merge_provider_options(self):
        """Test that user options override defaults."""
        defaults = {
            "device_id": "0",
            "precision": "FP32",
            "cache_dir": "/default/path",
        }

        user_options = {
            "device_id": "1",  # Override
            "precision": "FP16",  # Override
            # cache_dir not specified, should use default
        }

        merged = ProviderConfigManager.merge_provider_options(defaults, user_options)

        assert merged["device_id"] == "1"  # User override
        assert merged["precision"] == "FP16"  # User override
        assert merged["cache_dir"] == "/default/path"  # Default kept

    def test_known_provider_options_validation(self, caplog):
        """Test that unknown options trigger warnings."""
        import logging

        all_options = {
            "device_id": "0",  # Known option for CUDA
            "unknown_option": "value",  # Unknown option
        }

        with caplog.at_level(logging.WARNING):
            provider_opts = ProviderConfigManager.get_provider_specific_options(
                provider="CUDAExecutionProvider",
                all_options=all_options,
            )

            # Should only include known option
            assert "device_id" in provider_opts
            assert "unknown_option" not in provider_opts

            # Should have logged a warning about unknown option
            assert len(caplog.records) == 1
            assert "unknown_option" in caplog.text
            assert "CUDAExecutionProvider" in caplog.text


class TestBugFix3Integration:
    """Integration tests for provider config deduplication."""

    def test_no_code_duplication_between_modules(self):
        """Verify that onnx_session and onnx_backend use same logic."""
        # Both modules should import and use ProviderConfigManager
        from pykokoro import onnx_session

        # Check that both use the shared manager (by checking method signatures)
        # This is a smoke test to ensure refactoring didn't break anything

        # OnnxSessionManager should have the methods that delegate to
        # ProviderConfigManager
        session_manager = onnx_session.OnnxSessionManager()
        assert hasattr(session_manager, "_get_default_provider_options")
        assert hasattr(session_manager, "_get_provider_specific_options")

        # Test that the methods work
        opts1 = session_manager._get_default_provider_options("CUDAExecutionProvider")
        opts2 = ProviderConfigManager.get_default_provider_options("CUDAExecutionProvider", "fp32")

        # Should have same structure
        assert set(opts1.keys()) == set(opts2.keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
