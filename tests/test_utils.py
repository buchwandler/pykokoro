"""Tests for pykokoro.utils module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from pykokoro.utils import (
    DEFAULT_ENCODING,
    detect_encoding,
    format_chapters_range,
    format_duration,
    format_filename_template,
    format_size,
    get_device,
    get_gpu_info,
    get_user_cache_path,
    get_user_config_path,
    load_config,
    reset_config,
    sanitize_filename,
    save_config,
)


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_zero_seconds(self):
        """Zero seconds should format correctly."""
        assert format_duration(0) == "00:00:00"

    def test_seconds_only(self):
        """Seconds only should format correctly."""
        assert format_duration(45) == "00:00:45"

    def test_minutes_and_seconds(self):
        """Minutes and seconds should format correctly."""
        assert format_duration(125) == "00:02:05"

    def test_hours_minutes_seconds(self):
        """Hours, minutes, and seconds should format correctly."""
        assert format_duration(3661) == "01:01:01"

    def test_large_duration(self):
        """Large durations should format correctly."""
        assert format_duration(36000) == "10:00:00"

    def test_float_seconds(self):
        """Float seconds should be truncated."""
        assert format_duration(65.9) == "00:01:05"


class TestFormatSize:
    """Tests for format_size function."""

    def test_bytes(self):
        """Small sizes should show in bytes."""
        assert format_size(500) == "500.0 B"

    def test_kilobytes(self):
        """KB range should format correctly."""
        assert format_size(1500) == "1.5 KB"

    def test_megabytes(self):
        """MB range should format correctly."""
        assert format_size(1500000) == "1.4 MB"

    def test_gigabytes(self):
        """GB range should format correctly."""
        assert format_size(1500000000) == "1.4 GB"

    def test_terabytes(self):
        """TB range should format correctly."""
        assert format_size(1500000000000) == "1.4 TB"

    def test_zero(self):
        """Zero should format as bytes."""
        assert format_size(0) == "0.0 B"


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_normal_string(self):
        """Normal strings should pass through."""
        assert sanitize_filename("hello_world") == "hello_world"

    def test_removes_invalid_chars(self):
        """Invalid characters should be removed."""
        assert sanitize_filename('file<>:"/\\|?*name') == "filename"

    def test_replaces_spaces_with_underscore(self):
        """Spaces should be replaced with underscores."""
        assert sanitize_filename("hello world") == "hello_world"

    def test_collapses_multiple_spaces(self):
        """Multiple spaces should collapse to single underscore."""
        assert sanitize_filename("hello   world") == "hello_world"

    def test_strips_leading_trailing_underscores(self):
        """Leading and trailing underscores should be stripped."""
        assert sanitize_filename("_hello_world_") == "hello_world"

    def test_max_length_truncation(self):
        """Long names should be truncated."""
        long_name = "a" * 150
        result = sanitize_filename(long_name, max_length=100)
        assert len(result) <= 100

    def test_empty_returns_output(self):
        """Empty or all-invalid strings should return 'output'."""
        assert sanitize_filename("") == "output"
        assert sanitize_filename("<>:") == "output"

    def test_custom_max_length(self):
        """Custom max length should be respected."""
        result = sanitize_filename("hello_world_test", max_length=10)
        assert len(result) <= 10


class TestGetDevice:
    """Tests for get_device function (ONNX Runtime providers)."""

    def test_cpu_when_disabled(self):
        """Should return CPUExecutionProvider when GPU disabled."""
        assert get_device(use_gpu=False) == "CPUExecutionProvider"

    def test_returns_valid_provider(self):
        """Should return a valid ONNX Runtime execution provider."""
        device = get_device(use_gpu=True)
        valid_providers = (
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "DmlExecutionProvider",
        )
        assert device in valid_providers


class TestGetGpuInfo:
    """Tests for get_gpu_info function."""

    def test_returns_tuple(self):
        """Should return a tuple of (message, available)."""
        result = get_gpu_info(enabled=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], bool)

    def test_disabled_message(self):
        """Should indicate when GPU is disabled."""
        message, _available = get_gpu_info(enabled=False)
        # Either GPU is available but disabled, or just not available
        assert isinstance(message, str)
        assert len(message) > 0


class TestConfigFunctions:
    """Tests for configuration functions."""

    def test_get_user_config_path_returns_path(self):
        """Should return a Path object."""
        path = get_user_config_path()
        assert isinstance(path, Path)
        assert path.name == "config.json"

    def test_get_user_cache_path_returns_path(self):
        """Should return a Path object."""
        path = get_user_cache_path()
        assert isinstance(path, Path)

    def test_get_user_cache_path_with_folder(self):
        """Should create subfolder in cache path."""
        path = get_user_cache_path("test_folder")
        assert isinstance(path, Path)
        assert "test_folder" in str(path)

    def test_load_config_returns_dict(self):
        """Should return a dictionary."""
        config = load_config()
        assert isinstance(config, dict)

    def test_load_config_has_defaults(self):
        """Should have default keys."""
        config = load_config()
        assert "model_quality" in config
        assert "use_gpu" in config
        assert "vocab_version" in config

    def test_save_and_load_config(self):
        """Should be able to save and load config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with patch("pykokoro.utils.get_user_config_path", return_value=config_path):
                test_config = {"test_key": "test_value", "model_quality": "fp16"}
                result = save_config(test_config)
                assert result is True

                loaded = load_config()
                assert loaded["test_key"] == "test_value"
                assert loaded["model_quality"] == "fp16"

    def test_reset_config_returns_defaults(self):
        """Reset should return default config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with patch("pykokoro.utils.get_user_config_path", return_value=config_path):
                # First save custom config
                save_config({"custom_key": "value"})

                # Reset should return defaults
                config = reset_config()
                assert "custom_key" not in config
                assert "model_quality" in config


class TestDetectEncoding:
    """Tests for detect_encoding function."""

    def test_detect_utf8(self):
        """Should detect UTF-8 encoding."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Hello, World! \u00e4\u00f6\u00fc")
            f.flush()
            filename = f.name
        try:
            encoding = detect_encoding(filename)
            assert encoding in ("utf-8", "ascii")
        finally:
            os.unlink(filename)

    def test_detect_ascii(self):
        """Should detect ASCII encoding."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="ascii"
        ) as f:
            f.write("Hello, World!")
            f.flush()
            filename = f.name
        try:
            encoding = detect_encoding(filename)
            assert encoding in ("utf-8", "ascii")  # ASCII is subset of UTF-8
        finally:
            os.unlink(filename)

    def test_returns_lowercase(self):
        """Encoding should be returned in lowercase."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            f.flush()
            filename = f.name
        try:
            encoding = detect_encoding(filename)
            assert encoding == encoding.lower()
        finally:
            os.unlink(filename)


class TestDefaultEncoding:
    """Tests for DEFAULT_ENCODING constant."""

    def test_default_encoding_is_string(self):
        """DEFAULT_ENCODING should be a string."""
        assert isinstance(DEFAULT_ENCODING, str)

    def test_default_encoding_is_valid(self):
        """DEFAULT_ENCODING should be a valid encoding."""
        # Should not raise an exception
        "test".encode(DEFAULT_ENCODING)


class TestFormatChaptersRange:
    """Tests for format_chapters_range function."""

    def test_all_chapters_returns_empty(self):
        """All chapters selected should return empty string."""
        assert format_chapters_range([0, 1, 2, 3, 4], 5) == ""

    def test_all_chapters_unordered_returns_empty(self):
        """All chapters selected in any order should return empty string."""
        assert format_chapters_range([4, 2, 0, 1, 3], 5) == ""

    def test_partial_range_returns_chapters_range(self):
        """Partial selection should return chapters_X-Y format."""
        assert format_chapters_range([0, 1, 2], 5) == "chapters_1-3"

    def test_single_chapter(self):
        """Single chapter should return chapters_X format."""
        assert format_chapters_range([2], 5) == "chapters_3"

    def test_non_contiguous_uses_min_max(self):
        """Non-contiguous selection should use min-max range."""
        assert format_chapters_range([0, 2, 4], 5) == "chapters_1-5"

    def test_first_chapter_only(self):
        """First chapter only should return chapters_1."""
        assert format_chapters_range([0], 10) == "chapters_1"

    def test_last_chapter_only(self):
        """Last chapter only should return correct chapter number."""
        assert format_chapters_range([9], 10) == "chapters_10"

    def test_empty_list_returns_empty(self):
        """Empty list should return empty string."""
        assert format_chapters_range([], 5) == ""

    def test_middle_range(self):
        """Middle range selection should work correctly."""
        assert format_chapters_range([2, 3, 4], 10) == "chapters_3-5"

    def test_two_chapters_at_ends(self):
        """Two chapters at opposite ends should show full range."""
        assert format_chapters_range([0, 9], 10) == "chapters_1-10"


class TestFormatFilenameTemplate:
    """Tests for format_filename_template function."""

    def test_book_title_only(self):
        """Simple book title template should work."""
        result = format_filename_template("{book_title}", book_title="My Book")
        assert result == "My_Book"

    def test_author_and_title(self):
        """Author and title template should work."""
        result = format_filename_template(
            "{author}_{book_title}", book_title="My Book", author="John Doe"
        )
        assert result == "John_Doe_My_Book"

    def test_chapter_number_formatting(self):
        """Chapter number with format spec should work."""
        result = format_filename_template(
            "{chapter_num:03d}_{chapter_title}",
            chapter_num=1,
            chapter_title="Introduction",
        )
        assert result == "001_Introduction"

    def test_full_chapter_template(self):
        """Full chapter template with book title should work."""
        result = format_filename_template(
            "{chapter_num:03d}_{book_title}_{chapter_title}",
            book_title="My Book",
            chapter_title="Chapter One",
            chapter_num=5,
        )
        assert result == "005_My_Book_Chapter_One"

    def test_empty_book_title_fallback_to_input_stem(self):
        """Empty book title should fall back to input_stem."""
        result = format_filename_template("{book_title}", book_title="", input_stem="my_file")
        assert result == "my_file"

    def test_empty_book_title_fallback_to_default(self):
        """Empty book title with no input_stem should fall back to default."""
        result = format_filename_template("{book_title}", book_title="", default_title="Untitled")
        assert result == "Untitled"

    def test_special_characters_sanitized(self):
        """Special characters in values should be sanitized."""
        result = format_filename_template("{book_title}", book_title="Test: Book/Name?")
        assert result == "Test_BookName"

    def test_spaces_replaced_with_underscores(self):
        """Spaces should be replaced with underscores."""
        result = format_filename_template("{book_title}", book_title="The Great Book")
        assert result == "The_Great_Book"

    def test_chapters_range_included(self):
        """Chapters range variable should work."""
        result = format_filename_template(
            "{book_title}_{chapters_range}",
            book_title="My Book",
            chapters_range="chapters_1-5",
        )
        assert result == "My_Book_chapters_1-5"

    def test_empty_chapters_range(self):
        """Empty chapters range should produce clean filename."""
        result = format_filename_template("{book_title}", book_title="My Book", chapters_range="")
        assert result == "My_Book"

    def test_input_stem_variable(self):
        """Input stem variable should work."""
        result = format_filename_template(
            "{input_stem}_{book_title}",
            book_title="My Book",
            input_stem="original_file",
        )
        assert result == "original_file_My_Book"

    def test_max_length_truncation(self):
        """Long filenames should be truncated to max_length."""
        result = format_filename_template("{book_title}", book_title="A" * 200, max_length=50)
        assert len(result) <= 50

    def test_invalid_template_variable_fallback(self):
        """Invalid template variable should fall back gracefully."""
        result = format_filename_template("{invalid_var}", book_title="My Book")
        # Should fall back to book title
        assert result == "My_Book"

    def test_invalid_format_spec_fallback(self):
        """Invalid format spec should fall back gracefully."""
        result = format_filename_template("{book_title:invalid}", book_title="My Book")
        # Should fall back to book title
        assert result == "My_Book"

    def test_chapter_num_without_format_spec(self):
        """Chapter number without format spec should work."""
        result = format_filename_template(
            "{chapter_num}_{chapter_title}", chapter_num=7, chapter_title="Test"
        )
        assert result == "7_Test"

    def test_all_variables_together(self):
        """All variables used together should work."""
        result = format_filename_template(
            "{author}_{book_title}_{chapter_num:03d}_{chapter_title}",
            book_title="Epic Story",
            author="Jane Smith",
            chapter_title="The Beginning",
            chapter_num=1,
            input_stem="file",
            chapters_range="",
        )
        assert result == "Jane_Smith_Epic_Story_001_The_Beginning"

    def test_unicode_characters(self):
        """Unicode characters should be handled properly."""
        result = format_filename_template("{book_title}", book_title="Über die Kunst")
        # Should sanitize but preserve valid unicode
        assert "ber" in result or "Uber" in result.lower()

    def test_empty_author(self):
        """Empty author should not cause issues."""
        result = format_filename_template("{author}_{book_title}", book_title="My Book", author="")
        # Empty author gets sanitized, result should still be valid
        assert "My_Book" in result

    def test_whitespace_only_title(self):
        """Whitespace-only title should fall back to default."""
        result = format_filename_template(
            "{book_title}", book_title="   ", default_title="Untitled"
        )
        assert result == "Untitled"

    def test_result_never_empty(self):
        """Result should never be empty string."""
        result = format_filename_template(
            "{book_title}", book_title="", input_stem="", default_title="Fallback"
        )
        assert result == "Fallback"
        assert len(result) > 0
