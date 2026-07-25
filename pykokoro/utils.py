"""Utility functions for pykokoro - config, GPU detection, encoding, etc."""

import json
import platform
import sys
from pathlib import Path
from typing import Any

from platformdirs import user_cache_dir, user_config_dir

from .constants import DEFAULT_CONFIG

# Default encoding for subprocess
DEFAULT_ENCODING = sys.getfilesystemencoding()


def get_user_config_path() -> Path:
    """Get path to user configuration file."""
    if platform.system() != "Windows":
        # On non-Windows, prefer ~/.config/pykokoro if it already exists
        custom_dir = Path.home() / ".config" / "pykokoro"
        if custom_dir.exists():
            config_dir = custom_dir
        else:
            config_dir = Path(
                user_config_dir(
                    "pykokoro", appauthor=False, roaming=True, ensure_exists=True
                )
            )
    else:
        config_dir = Path(
            user_config_dir(
                "pykokoro", appauthor=False, roaming=True, ensure_exists=True
            )
        )
    return config_dir / "config.json"


def get_user_cache_path(folder: str | None = None) -> Path:
    """Get path to user cache directory, optionally with a subfolder."""
    cache_dir = Path(
        user_cache_dir("pykokoro", appauthor=False, opinion=True, ensure_exists=True)
    )
    if folder:
        cache_dir = cache_dir / folder
        cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_config() -> dict[str, Any]:
    """Load configuration from file, returning defaults if not found."""
    try:
        config_path = get_user_config_path()
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                user_config = json.load(f)
                # Merge with defaults to ensure all keys exist
                return {**DEFAULT_CONFIG, **user_config}
    except (OSError, ValueError, KeyError):
        pass
    return DEFAULT_CONFIG.copy()


def save_config(config: dict[str, Any]) -> bool:
    """Save configuration to file. Returns True on success."""
    try:
        config_path = get_user_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return True
    except (OSError, ValueError):
        return False


def reset_config() -> dict[str, Any]:
    """Reset configuration to defaults and save."""
    save_config(DEFAULT_CONFIG)
    return DEFAULT_CONFIG.copy()


def detect_encoding(file_path: str | Path) -> str:
    """Detect the encoding of a file using chardet/charset_normalizer."""
    import chardet
    import charset_normalizer

    with open(file_path, "rb") as f:
        raw_data = f.read()

    detected_encoding = None
    for detector in (charset_normalizer, chardet):
        try:
            result = detector.detect(raw_data)
            if result and result.get("encoding"):
                detected_encoding = result["encoding"]
                break
        except (OSError, UnicodeDecodeError, LookupError):
            continue

    encoding = detected_encoding if detected_encoding else "utf-8"
    return encoding.lower()


def get_gpu_info(enabled: bool = True) -> tuple[str, bool]:
    """
    Check GPU acceleration availability for ONNX runtime.

    Args:
        enabled: Whether GPU acceleration is requested

    Returns:
        Tuple of (status message, is_gpu_available)
    """
    if not enabled:
        return "GPU disabled in config. Using CPU.", False

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()

        # Check for CUDA
        if "CUDAExecutionProvider" in providers:
            return "CUDA GPU available via ONNX Runtime.", True

        # Check for CoreML (Apple)
        if "CoreMLExecutionProvider" in providers:
            return "CoreML GPU available via ONNX Runtime.", True

        # Check for DirectML (Windows)
        if "DmlExecutionProvider" in providers:
            return "DirectML GPU available via ONNX Runtime.", True

        return f"No GPU providers available. Using CPU. (Available: {providers})", False
    except ImportError:
        return "ONNX Runtime not installed. Using CPU.", False
    except (RuntimeError, OSError) as e:
        return f"Error checking GPU: {e}", False


def get_device(use_gpu: bool = True) -> str:
    """
    Get the appropriate execution provider for ONNX Runtime.

    Args:
        use_gpu: Whether to attempt GPU usage

    Returns:
        Execution provider name: 'CUDAExecutionProvider',
        'CoreMLExecutionProvider', or 'CPUExecutionProvider'
    """
    if not use_gpu:
        return "CPUExecutionProvider"

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()

        # Prefer CUDA
        if "CUDAExecutionProvider" in providers:
            return "CUDAExecutionProvider"

        # CoreML for Apple
        if "CoreMLExecutionProvider" in providers:
            return "CoreMLExecutionProvider"

        # DirectML for Windows
        if "DmlExecutionProvider" in providers:
            return "DmlExecutionProvider"

    except ImportError:
        pass

    return "CPUExecutionProvider"


def sanitize_filename(name: str, max_length: int = 100) -> str:
    """
    Sanitize a string for use as a filename.

    Args:
        name: The string to sanitize
        max_length: Maximum length of the result

    Returns:
        Sanitized filename
    """
    import re

    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "", name)
    # Replace multiple spaces/underscores with single underscore
    sanitized = re.sub(r"[\s_]+", "_", sanitized).strip("_")
    # Truncate if needed
    if len(sanitized) > max_length:
        # Try to break at underscore
        pos = sanitized[:max_length].rfind("_")
        sanitized = sanitized[: pos if pos > 0 else max_length].rstrip("_")
    return sanitized or "output"


def format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def format_chapters_range(indices: list[int], total_chapters: int) -> str:
    """
    Format chapter indices into a range string for filenames.

    Returns empty string if all chapters are selected.
    Returns "chapters_X-Y" style string for partial selection (using min-max).

    Args:
        indices: 0-based chapter indices
        total_chapters: Total number of chapters in book

    Returns:
        Range string (e.g., "chapters_1-5") or empty string if all chapters
    """
    if not indices:
        return ""

    # Check if all chapters are selected
    if len(indices) == total_chapters and set(indices) == set(range(total_chapters)):
        return ""

    # Convert to 1-based and get min/max
    min_chapter = min(indices) + 1
    max_chapter = max(indices) + 1

    if min_chapter == max_chapter:
        return f"chapters_{min_chapter}"
    return f"chapters_{min_chapter}-{max_chapter}"


def generate_silence(duration: float, sample_rate: int = 24000) -> Any:
    """Generate silence array of specified duration.

    Args:
        duration: Duration in seconds
        sample_rate: Audio sample rate (default: 24000 for Kokoro)

    Returns:
        Numpy array of zeros (float32)
    """
    import numpy as np

    return np.zeros(int(duration * sample_rate), dtype=np.float32)


def format_filename_template(
    template: str,
    book_title: str = "",
    author: str = "",
    chapter_title: str = "",
    chapter_num: int = 0,
    input_stem: str = "",
    chapters_range: str = "",
    default_title: str = "Untitled",
    max_length: int = 100,
) -> str:
    """
    Format a filename template with the given variables.

    All values are sanitized before substitution.
    Falls back to input_stem or default_title if book_title is empty.

    Template variables:
        {book_title} - Sanitized book title
        {author} - Sanitized author name
        {chapter_title} - Sanitized chapter title
        {chapter_num} - Chapter number (1-based), supports format specs
        {input_stem} - Original input filename without extension
        {chapters_range} - Chapter range string (e.g., "chapters_1-5") or empty

    Args:
        template: Python format string (e.g., "{book_title}_{chapter_num:03d}")
        book_title: Book title from metadata
        author: Author name from metadata
        chapter_title: Chapter title
        chapter_num: 1-based chapter number
        input_stem: Original input filename without extension
        chapters_range: Chapter range string or empty
        default_title: Fallback title if book_title is empty
        max_length: Maximum length of final filename

    Returns:
        Formatted and sanitized filename (without extension)

    Examples:
        >>> format_filename_template("{book_title}", book_title="My Book")
        'My_Book'
        >>> format_filename_template(
        ...     "{chapter_num:03d}_{chapter_title}",
        ...     chapter_num=1,
        ...     chapter_title="Intro",
        ... )
        '001_Intro'
        >>> format_filename_template(
        ...     "{author}_{book_title}",
        ...     author="John Doe",
        ...     book_title="",
        ... )
        'John_Doe_Untitled'
    """
    # Determine effective book title with fallback
    effective_title = book_title.strip() if book_title else ""
    if not effective_title:
        effective_title = input_stem.strip() if input_stem else default_title

    # Sanitize all string values (but don't truncate yet - do that at the end)
    safe_book_title = sanitize_filename(effective_title, max_length=200)
    safe_author = sanitize_filename(author, max_length=100) if author else ""
    safe_chapter_title = (
        sanitize_filename(chapter_title, max_length=100) if chapter_title else ""
    )
    safe_input_stem = (
        sanitize_filename(input_stem, max_length=100) if input_stem else ""
    )
    safe_chapters_range = (
        sanitize_filename(chapters_range, max_length=50) if chapters_range else ""
    )

    # Build the format kwargs
    format_kwargs = {
        "book_title": safe_book_title,
        "author": safe_author,
        "chapter_title": safe_chapter_title,
        "chapter_num": chapter_num,
        "input_stem": safe_input_stem,
        "chapters_range": safe_chapters_range,
    }

    try:
        result = template.format(**format_kwargs)
    except KeyError:
        # Unknown template variable - fall back to book title
        result = safe_book_title
    except ValueError:
        # Invalid format spec - fall back to book title
        result = safe_book_title

    # Final sanitization and truncation
    result = sanitize_filename(result, max_length=max_length)

    # Ensure we have something
    return result or default_title
