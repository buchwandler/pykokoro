"""ONNX backend for pykokoro - native ONNX TTS without external dependencies."""

import contextlib
import io
import logging
import os
import shutil
import sqlite3
import tempfile
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import onnxruntime as rt
from huggingface_hub import hf_hub_download

from .audio_generator import AudioGenerator
from .exceptions import ConfigurationError
from .onnx_session import OnnxSessionManager
from .provider_config import ProviderConfigManager
from .tokenizer import EspeakConfig, Tokenizer, TokenizerConfig
from .utils import get_user_cache_path
from .voice_manager import VoiceBlend, VoiceManager, normalize_voice_style

if TYPE_CHECKING:
    from .short_sentence_handler import ShortSentenceConfig
    from .types import PhonemeSegment

# Logger for debugging
logger = logging.getLogger(__name__)

DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_RETRIES = 3
DOWNLOAD_TIMEOUT_SECONDS = 30
DOWNLOAD_LOCK_TIMEOUT_SECONDS = 30
MIN_ONNX_BYTES = 1_000_000
MIN_VOICE_ARCHIVE_BYTES = 1_000_000
MIN_VOICE_BIN_BYTES = 100_000
MIN_CONFIG_BYTES = 100

# Model quality type
ModelQuality = Literal[
    "fp32", "fp16", "fp16-gpu", "q8", "q8f16", "q4", "q4f16", "uint8", "uint8f16"
]
DEFAULT_MODEL_QUALITY: ModelQuality = "fp32"

# Provider type
ProviderType = Literal["auto", "cpu", "cuda", "openvino", "directml", "coreml"]

# Model source type
ModelSource = Literal["huggingface", "github"]
DEFAULT_MODEL_SOURCE: ModelSource = "huggingface"

# Model variant type (for GitHub and HuggingFace sources)
ModelVariant = Literal["v1.0", "v1.1-zh", "v1.1-de"]
DEFAULT_MODEL_VARIANT: ModelVariant = "v1.0"

# Quality to filename mapping (Hugging Face)
MODEL_QUALITY_FILES_HF: dict[str, str] = {
    "fp32": "model.onnx",
    "fp16": "model_fp16.onnx",
    "q8": "model_quantized.onnx",
    "q8f16": "model_q8f16.onnx",
    "q4": "model_q4.onnx",
    "q4f16": "model_q4f16.onnx",
    "uint8": "model_uint8.onnx",
    "uint8f16": "model_uint8f16.onnx",
}

# Quality to filename mapping (GitHub v1.0 - English)
MODEL_QUALITY_FILES_GITHUB_V1_0: dict[str, str] = {
    "fp32": "kokoro-v1.0.onnx",
    "fp16": "kokoro-v1.0.fp16.onnx",
    "fp16-gpu": "kokoro-v1.0.fp16-gpu.onnx",
    "q8": "kokoro-v1.0.int8.onnx",
}

# Quality to filename mapping (GitHub v1.1-zh - Chinese)
MODEL_QUALITY_FILES_GITHUB_V1_1_ZH: dict[str, str] = {
    "fp32": "kokoro-v1.1-zh.onnx",
}

MODEL_QUALITY_FILES_GITHUB_V1_1_DE: dict[str, str] = {
    "fp32": "kokoro-german-v1.1.onnx",
    "q8": "kokoro-german-v1.1.int8.onnx",
}


# Note: Both HF v1.0 and v1.1-zh use the same filename convention
# (MODEL_QUALITY_FILES_HF)

# Backward compatibility
MODEL_QUALITY_FILES = MODEL_QUALITY_FILES_HF

# HuggingFace repositories for models and voices (onnx-community)
HF_REPO_V1_0 = "onnx-community/Kokoro-82M-v1.0-ONNX"
HF_REPO_V1_1_ZH = "onnx-community/Kokoro-82M-v1.1-zh-ONNX"

# HuggingFace repositories for configs (hexgrad)
HF_CONFIG_REPO_V1_0 = "hexgrad/Kokoro-82M"
HF_CONFIG_REPO_V1_1_ZH = "hexgrad/Kokoro-82M-v1.1-zh"
HF_CONFIG_REPO_V1_1_DE = "Tundragoon/Kokoro-German"

# Subfolders and filenames within HuggingFace repos
HF_MODEL_SUBFOLDER = "onnx"
HF_VOICES_SUBFOLDER = "voices"
HF_CONFIG_FILENAME = "config.json"

# URLs for model files (GitHub)
GITHUB_REPO = "thewh1teagle/kokoro-onnx"

# GitHub v1.0 (English)
GITHUB_RELEASE_TAG_V1_0 = "model-files-v1.0"
GITHUB_BASE_URL_V1_0 = (
    f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_TAG_V1_0}"
)
GITHUB_VOICES_FILENAME_V1_0 = "voices-v1.0.bin"

# GitHub v1.1-zh (Chinese)
GITHUB_RELEASE_TAG_V1_1_ZH = "model-files-v1.1"
GITHUB_BASE_URL_V1_1_ZH = (
    f"https://github.com/{GITHUB_REPO}/releases/download/{GITHUB_RELEASE_TAG_V1_1_ZH}"
)
GITHUB_VOICES_FILENAME_V1_1_ZH = "voices-v1.1-zh.bin"

GITHUB_REPO_GERMAN = "holgern/kokoro-onnx-model"
GITHUB_RELEASE_TAG_V1_1_DE = "model-files-german-v1.1"
GITHUB_BASE_URL_V1_1_DE = f"https://github.com/{GITHUB_REPO_GERMAN}/releases/download/{GITHUB_RELEASE_TAG_V1_1_DE}"

GITHUB_VOICES_FILENAME_V1_1_DE = "voices-german-v1.1.bin"
# All available voice names for v1.0 (54 voices - English/multilingual)
# Used by both HuggingFace and GitHub sources
# These are used for downloading individual voice files from HuggingFace
VOICE_NAMES_V1_0 = [
    "af",
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
    "ef_dora",
    "em_alex",
    "em_santa",
    "ff_siwis",
    "hf_alpha",
    "hf_beta",
    "hm_omega",
    "hm_psi",
    "if_sara",
    "im_nicola",
    "jf_alpha",
    "jf_gongitsune",
    "jf_nezumi",
    "jf_tebukuro",
    "jm_kumo",
    "pf_dora",
    "pm_alex",
    "pm_santa",
    "zf_xiaobei",
    "zf_xiaoni",
    "zf_xiaoxiao",
    "zm_yunjian",
    "zm_yunxi",
    "zm_yunxia",
    "zm_yunyang",
]


# Expected voice names for GitHub v1.1-zh (Chinese model)
# Note: These are loaded dynamically from voices.bin, this list is for reference
# The v1.1-zh model contains 103 voices with various Chinese speakers
VOICE_NAMES_ZH = [
    # Sample voices from the v1.1-zh model:
    "af_maple",  # Female voice
    "af_sol",  # Female voice
    "bf_vale",  # British female voice
    # Numbered Chinese female voices (zf_XXX)
    "zf_001",
    "zf_002",
    "zf_003",  # ... many more numbered voices
    # Numbered Chinese male voices (zm_XXX)
    "zm_009",
    "zm_010",
    "zm_011",  # ... many more numbered voices
    # Note: Full list contains 103 voices total
    # Use kokoro.get_voices() to retrieve the complete list at runtime
]

# Complete voice list for v1.1-zh (103 voices - Chinese)
# Used by both HuggingFace and GitHub sources
VOICE_NAMES_V1_1_ZH = [
    "af_maple",
    "af_sol",
    "bf_vale",
    "zf_001",
    "zf_002",
    "zf_003",
    "zf_004",
    "zf_005",
    "zf_006",
    "zf_007",
    "zf_008",
    "zf_017",
    "zf_018",
    "zf_019",
    "zf_021",
    "zf_022",
    "zf_023",
    "zf_024",
    "zf_026",
    "zf_027",
    "zf_028",
    "zf_032",
    "zf_036",
    "zf_038",
    "zf_039",
    "zf_040",
    "zf_042",
    "zf_043",
    "zf_044",
    "zf_046",
    "zf_047",
    "zf_048",
    "zf_049",
    "zf_051",
    "zf_059",
    "zf_060",
    "zf_067",
    "zf_070",
    "zf_071",
    "zf_072",
    "zf_073",
    "zf_074",
    "zf_075",
    "zf_076",
    "zf_077",
    "zf_078",
    "zf_079",
    "zf_083",
    "zf_084",
    "zf_085",
    "zf_086",
    "zf_087",
    "zf_088",
    "zf_090",
    "zf_092",
    "zf_093",
    "zf_094",
    "zf_099",
    "zm_009",
    "zm_010",
    "zm_011",
    "zm_012",
    "zm_013",
    "zm_014",
    "zm_015",
    "zm_016",
    "zm_020",
    "zm_025",
    "zm_029",
    "zm_030",
    "zm_031",
    "zm_033",
    "zm_034",
    "zm_035",
    "zm_037",
    "zm_041",
    "zm_045",
    "zm_050",
    "zm_052",
    "zm_053",
    "zm_054",
    "zm_055",
    "zm_056",
    "zm_057",
    "zm_058",
    "zm_061",
    "zm_062",
    "zm_063",
    "zm_064",
    "zm_065",
    "zm_066",
    "zm_068",
    "zm_069",
    "zm_080",
    "zm_081",
    "zm_082",
    "zm_089",
    "zm_091",
    "zm_095",
    "zm_096",
    "zm_097",
    "zm_098",
    "zm_100",
]

VOICE_NAMES_V1_1_DE = ["df_eva", "dm_bernd"]

# Voice name documentation by language/variant
# These voices are dynamically loaded from the model's voices.bin file
# The actual available voices may vary depending on the model source and variant
VOICE_NAMES_BY_VARIANT = {
    "v1.0": VOICE_NAMES_V1_0,  # Same as HuggingFace (multi-language)
    "v1.1-zh": VOICE_NAMES_V1_1_ZH,  # Chinese-specific voices
    "v1.1-de": VOICE_NAMES_V1_1_DE,  # German-specific voices
}


# Language code mapping for kokoro-onnx
LANG_CODE_TO_ONNX = {
    "a": "en-us",  # American English
    "b": "en-gb",  # British English
    "e": "es",  # Spanish
    "f": "fr",  # French
    "h": "hi",  # Hindi
    "i": "it",  # Italian
    "j": "ja",  # Japanese
    "p": "pt",  # Portuguese
    "z": "zh",  # Chinese
    "d": "de",  # German
}


def get_onnx_lang_code(ttsforge_lang: str) -> str:
    """Convert ttsforge language code to kokoro-onnx language code."""
    return LANG_CODE_TO_ONNX.get(ttsforge_lang, "en-us")


# =============================================================================
# Path helper functions
# =============================================================================


def get_model_dir(
    source: ModelSource = DEFAULT_MODEL_SOURCE,
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
) -> Path:
    """
    Get directory for model files.

    Returns: ~/.cache/pykokoro/models/{source}/{variant}/

    Args:
        source: Model source (huggingface or github)
        variant: Model variant (v1.0, v1.1-de, v1.1-zh)

    Returns:
        Path to model directory
    """
    model_dir = get_user_cache_path("models") / source / variant
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def get_voices_dir(
    source: ModelSource = DEFAULT_MODEL_SOURCE,
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
) -> Path:
    """
    Get directory for voice files.

    Returns: ~/.cache/pykokoro/voices/{source}/{variant}/

    Args:
        source: Model source (huggingface or github)
        variant: Model variant (v1.0 or v1.1-zh, v1.1-de)

    Returns:
        Path to voices directory
    """
    voices_dir = get_user_cache_path("voices") / source / variant
    voices_dir.mkdir(parents=True, exist_ok=True)
    return voices_dir


def get_config_path(variant: ModelVariant = DEFAULT_MODEL_VARIANT) -> Path:
    """
    Get path to config file (shared across sources for same variant).

    Returns: ~/.cache/pykokoro/config/{variant}/config.json

    Config files are downloaded from hexgrad repos and shared between
    HuggingFace and GitHub sources for the same variant.

    Args:
        variant: Model variant (v1.0 or v1.1-zh)

    Returns:
        Path to config file
    """
    config_dir = get_user_cache_path("config") / variant
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / HF_CONFIG_FILENAME


def get_voices_bin_path() -> Path:
    """Get the path to the combined voices.bin.npz file."""
    return get_user_cache_path() / "voices.bin.npz"


def get_model_path(
    quality: ModelQuality = DEFAULT_MODEL_QUALITY,
    source: ModelSource = DEFAULT_MODEL_SOURCE,
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
) -> Path:
    """
    Get full path to a specific model file.

    Args:
        quality: Model quality/quantization level
        source: Model source (huggingface or github)
        variant: Model variant (v1.0 or v1.1-zh)

    Returns:
        Path to model file

    Raises:
        ValueError: If quality is not available for the source/variant combination
    """
    model_dir = get_model_dir(source, variant)

    # Get appropriate filename mapping based on source and variant
    if source == "huggingface":
        # Both v1.0 and v1.1-zh use same filename convention
        quality_files = MODEL_QUALITY_FILES_HF
    elif source == "github":
        if variant == "v1.0":
            quality_files = MODEL_QUALITY_FILES_GITHUB_V1_0
        elif variant == "v1.1-de":
            quality_files = MODEL_QUALITY_FILES_GITHUB_V1_1_DE
        else:  # v1.1-zh
            quality_files = MODEL_QUALITY_FILES_GITHUB_V1_1_ZH
    else:
        raise ValueError(f"Unknown source: {source}")

    # Get filename for quality
    if quality not in quality_files:
        available = ", ".join(quality_files.keys())
        raise ValueError(
            f"Quality '{quality}' not available for {source}/{variant}. "
            f"Available: {available}"
        )

    filename = quality_files[quality]

    # HuggingFace models are stored in onnx/ subdirectory
    if source == "huggingface":
        return model_dir / HF_MODEL_SUBFOLDER / filename

    return model_dir / filename


def get_voice_path(
    voice_name: str,
    source: ModelSource = DEFAULT_MODEL_SOURCE,
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
) -> Path:
    """Get the full path to an individual voice file."""
    return get_voices_dir(source, variant) / f"{voice_name}.bin"


# =============================================================================
# Download check functions
# =============================================================================


def is_config_downloaded(variant: ModelVariant = DEFAULT_MODEL_VARIANT) -> bool:
    """Check if config.json is downloaded for a specific variant.

    Args:
        variant: Model variant (v1.0 or v1.1-zh)

    Returns:
        True if config exists and has content, False otherwise
    """
    config_path = get_config_path(variant)
    return config_path.exists() and config_path.stat().st_size > 0


def is_model_downloaded(quality: ModelQuality = DEFAULT_MODEL_QUALITY) -> bool:
    """Check if a model file is already downloaded for a given quality."""
    model_path = get_model_path(quality)
    return model_path.exists() and model_path.stat().st_size > 0


def is_voice_downloaded(voice_name: str) -> bool:
    """Check if an individual voice file is already downloaded."""
    voice_path = get_voice_path(voice_name)
    return voice_path.exists() and voice_path.stat().st_size > 0


def are_voices_downloaded() -> bool:
    """Check if the combined voices.bin file exists."""
    voices_bin_path = get_voices_bin_path()
    return voices_bin_path.exists() and voices_bin_path.stat().st_size > 0


def are_models_downloaded(quality: ModelQuality = DEFAULT_MODEL_QUALITY) -> bool:
    """Check if model, config, and voices.bin are downloaded."""
    return (
        is_config_downloaded()
        and is_model_downloaded(quality)
        and are_voices_downloaded()
    )


# =============================================================================
# Download functions
# =============================================================================


def _validate_min_size(path: Path, min_size: int) -> None:
    size = path.stat().st_size
    if size < min_size:
        raise RuntimeError(
            f"Downloaded file {path.name} is too small ({size} bytes). "
            f"Expected at least {min_size} bytes."
        )


def _validate_onnx_file(path: Path) -> None:
    _validate_min_size(path, MIN_ONNX_BYTES)

    if "CPUExecutionProvider" not in rt.get_available_providers():
        logger.debug("CPUExecutionProvider unavailable; skipping ONNX validation")
        return

    try:
        rt.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Downloaded ONNX model '{path.name}' is invalid: {exc}"
        ) from exc


def _validate_voice_archive(path: Path) -> None:
    _validate_min_size(path, MIN_VOICE_ARCHIVE_BYTES)

    try:
        with np.load(str(path), allow_pickle=False) as voices:
            if not voices.files:
                raise RuntimeError("Voice archive is empty")
            first_key = voices.files[0]
            normalize_voice_style(
                voices[first_key],
                expected_length=None,
                require_dtype=True,
                voice_name=first_key,
            )
    except Exception as exc:
        raise RuntimeError(
            f"Downloaded voice archive '{path.name}' is invalid: {exc}"
        ) from exc


def _validate_voice_bin(path: Path) -> None:
    _validate_min_size(path, MIN_VOICE_BIN_BYTES)

    size = path.stat().st_size
    if size % 4 != 0:
        raise RuntimeError(
            f"Downloaded voice file '{path.name}' has invalid byte size {size}."
        )


@contextlib.contextmanager
def _download_lock(
    target_path: Path,
    timeout: float = DOWNLOAD_LOCK_TIMEOUT_SECONDS,
) -> Any:
    lock_path = target_path.with_suffix(target_path.suffix + ".lock")
    start = time.monotonic()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError as e:
            if time.monotonic() - start > timeout:
                raise RuntimeError(
                    f"Timed out waiting for download lock on {target_path}"
                ) from e
            time.sleep(0.1)

    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            lock_path.unlink()


def _run_with_retries(
    action: Callable[[], Path],
    *,
    description: str,
    retries: int = DOWNLOAD_RETRIES,
) -> Path:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return action()
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            delay = 2 ** (attempt - 1)
            logger.warning(
                f"{description} failed on attempt {attempt}/{retries}: {exc}. "
                f"Retrying in {delay}s."
            )
            time.sleep(delay)

    raise RuntimeError(f"{description} failed after {retries} attempts: {last_error}")


def _stream_download(
    url: str,
    local_path: Path,
    *,
    timeout: float,
    min_size: int | None = None,
    validator: Callable[[Path], None] | None = None,
) -> Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=local_path.parent,
        prefix=f"{local_path.name}.",
        suffix=".tmp",
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                for chunk in iter(lambda: response.read(DOWNLOAD_CHUNK_SIZE), b""):
                    tmp_file.write(chunk)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()
            raise

    try:
        if min_size is not None:
            _validate_min_size(tmp_path, min_size)
        if validator is not None:
            validator(tmp_path)
        os.replace(tmp_path, local_path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
        raise

    return local_path


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=destination.parent,
        prefix=f"{destination.name}.",
        suffix=".tmp",
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        shutil.copyfile(source, tmp_path)
        os.replace(tmp_path, destination)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
        raise


def _download_from_hf(
    repo_id: str,
    filename: str,
    subfolder: str | None = None,
    local_dir: Path | None = None,
    force: bool = False,
    min_size: int | None = None,
    validator: Callable[[Path], None] | None = None,
    retries: int = DOWNLOAD_RETRIES,
) -> Path:
    """
    Download a file from Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository ID
        filename: File to download
        subfolder: Subfolder in the repository
        local_dir: Local directory to save to
        force: Force re-download even if file exists

    Returns:
        Path to the downloaded file
    """
    # Use hf_hub_download to download the file
    # It handles caching automatically
    local_dir_path = Path(local_dir) if local_dir else None
    target_path: Path | None = None
    if local_dir_path is not None:
        target_path = local_dir_path / filename
        if subfolder:
            target_path = local_dir_path / subfolder / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)

    def _download() -> Path:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            local_dir=str(local_dir_path) if local_dir_path else None,
            force_download=force,
        )
        downloaded = Path(downloaded_path)
        try:
            if min_size is not None:
                _validate_min_size(downloaded, min_size)
            if validator is not None:
                validator(downloaded)
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                downloaded.unlink()
            raise
        return downloaded

    if target_path is not None:
        with _download_lock(target_path):
            return _run_with_retries(
                _download, description=f"HF download of {filename}", retries=retries
            )

    return _run_with_retries(
        _download, description=f"HF download of {filename}", retries=retries
    )


def download_config(
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
    force: bool = False,
) -> Path:
    """
    Download config.json from hexgrad HuggingFace repository.

    Config files are downloaded from hexgrad repos and stored in a shared
    location used by both HuggingFace and GitHub sources.

    Args:
        variant: Model variant (v1.0 or v1.1-zh)
        force: Force re-download even if file exists

    Returns:
        Path to the downloaded config file

    Note:
        - v1.0 config from: hexgrad/Kokoro-82M
        - v1.1-zh config from: hexgrad/Kokoro-82M-v1.1-zh
    """
    config_path = get_config_path(variant)

    if config_path.exists() and not force:
        logger.debug(f"Config already exists: {config_path}")
        return config_path

    # Select hexgrad repo based on variant
    if variant == "v1.0":
        repo_id = HF_CONFIG_REPO_V1_0  # hexgrad/Kokoro-82M
    elif variant == "v1.1-zh":
        repo_id = HF_CONFIG_REPO_V1_1_ZH  # hexgrad/Kokoro-82M-v1.1-zh
    elif variant == "v1.1-de":
        repo_id = HF_CONFIG_REPO_V1_1_DE  # Tundragoon/Kokoro-German
    else:
        raise ValueError(f"Unknown variant: {variant}")

    logger.info(f"Downloading config for {variant} from {repo_id}")

    return _download_from_hf(
        repo_id=repo_id,
        filename=HF_CONFIG_FILENAME,
        local_dir=config_path.parent,
        force=force,
        min_size=MIN_CONFIG_BYTES,
    )


def load_vocab_from_config(
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
) -> dict[str, int]:
    """Load vocabulary from variant-specific config.json.

    Args:
        variant: Model variant (v1.0 or v1.1-zh, or v1.1-de)

    Returns:
        Dictionary mapping phoneme characters to token indices

    Raises:
        FileNotFoundError: If config file doesn't exist after download
        ValueError: If config doesn't contain vocab
    """
    import json

    from kokorog2p import get_kokoro_vocab

    config_path = get_config_path(variant)

    # Download if not exists
    if not config_path.exists():
        logger.info(f"Downloading config for variant '{variant}'...")
        try:
            download_config(variant=variant)
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to download config for variant '{variant}': {e}"
            ) from e

    # Load config
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        logger.error(
            f"Failed to load config from {config_path}: {e}. "
            f"Falling back to default vocabulary."
        )
        return get_kokoro_vocab()

    # Extract vocabulary
    if "vocab" not in config:
        raise ValueError(
            f"Config at {config_path} does not contain 'vocab' key. "
            f"Cannot load variant-specific vocabulary."
        )

    vocab = config["vocab"]
    logger.info(
        f"Loaded vocabulary with {len(vocab)} tokens "
        f"for variant '{variant}' from {config_path.name}"
    )

    return vocab


def download_model(
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
    quality: ModelQuality = DEFAULT_MODEL_QUALITY,
    force: bool = False,
) -> Path:
    """
    Download model from HuggingFace (onnx-community repos).

    Args:
        variant: Model variant (v1.0 or v1.1-zh)
        quality: Model quality/quantization level
        force: Force re-download even if file exists

    Returns:
        Path to the downloaded model file

    Raises:
        ValueError: If quality is not available

    Note:
        - v1.0 from: onnx-community/Kokoro-82M-v1.0-ONNX
        - v1.1-zh from: onnx-community/Kokoro-82M-v1.1-zh-ONNX
    """
    # Select onnx-community repo based on variant
    if variant == "v1.0":
        repo_id = HF_REPO_V1_0
    elif variant == "v1.1-zh":
        repo_id = HF_REPO_V1_1_ZH
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Check if quality is available (both variants use same filenames)
    if quality not in MODEL_QUALITY_FILES_HF:
        available = ", ".join(MODEL_QUALITY_FILES_HF.keys())
        raise ValueError(f"Quality '{quality}' not available. Available: {available}")

    filename = MODEL_QUALITY_FILES_HF[quality]
    # Use new path structure
    model_dir = get_model_dir(source="huggingface", variant=variant)
    local_path = model_dir / HF_MODEL_SUBFOLDER / filename

    if local_path.exists() and not force:
        logger.debug(f"Model already exists: {local_path}")
        return local_path

    logger.info(f"Downloading {variant} model ({quality}) from {repo_id}")

    return _download_from_hf(
        repo_id=repo_id,
        filename=filename,
        subfolder=HF_MODEL_SUBFOLDER,
        local_dir=model_dir,
        force=force,
        validator=_validate_onnx_file,
    )


def download_voice(
    voice_name: str,
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
    force: bool = False,
) -> Path:
    """
    Download a single voice file from HuggingFace.

    Args:
        voice_name: Name of the voice to download
        variant: Model variant (v1.0 or v1.1-zh)
        force: Force re-download even if file exists

    Returns:
        Path to the downloaded voice file
    """
    # Select repo based on variant
    if variant == "v1.0":
        repo_id = HF_REPO_V1_0
    elif variant == "v1.1-zh":
        repo_id = HF_REPO_V1_1_ZH
    else:
        raise ValueError(f"Unknown variant: {variant}")

    filename = f"{voice_name}.bin"
    # Use new path structure
    voices_dir = get_voices_dir(source="huggingface", variant=variant)
    voices_dir.mkdir(parents=True, exist_ok=True)
    local_path = voices_dir / filename

    if local_path.exists() and not force:
        logger.debug(f"Voice already exists: {local_path}")
        return local_path

    logger.info(f"Downloading voice {voice_name} for {variant}")

    download_name = f"{HF_VOICES_SUBFOLDER}/{filename}"
    downloaded_path = _download_from_hf(
        repo_id=repo_id,
        filename=download_name,
        local_dir=None,
        force=force,
        min_size=MIN_VOICE_BIN_BYTES,
        validator=_validate_voice_bin,
    )

    with _download_lock(local_path):
        _atomic_copy(downloaded_path, local_path)

    return local_path


def download_all_voices(
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
    progress_callback: Callable[[str, int, int], None] | None = None,
    force: bool = False,
) -> Path:
    """
    Download all voice files from HuggingFace for a specific variant.

    Downloads individual .bin files and combines them into voices.bin.

    Args:
        variant: Model variant (v1.0 or v1.1-zh)
        progress_callback: Optional callback(filename, current, total)
        force: Force re-download even if files exist

    Returns:
        Path to voices directory

    Note:
        - v1.0: 54 voices from onnx-community/Kokoro-82M-v1.0-ONNX
        - v1.1-zh: 103 voices from onnx-community/Kokoro-82M-v1.1-zh-ONNX
    """
    # Select repo and voice list based on variant
    if variant == "v1.0":
        repo_id = HF_REPO_V1_0
        voice_names = VOICE_NAMES_V1_0
    elif variant == "v1.1-zh":
        repo_id = HF_REPO_V1_1_ZH
        voice_names = VOICE_NAMES_V1_1_ZH
    else:
        raise ValueError(f"Unknown variant: {variant}")

    voices_dir = get_voices_dir(source="huggingface", variant=variant)
    voices_dir.mkdir(parents=True, exist_ok=True)

    voices_bin_path = voices_dir / "voices.bin.npz"
    force_download = force

    if voices_dir.exists():
        for temp_path in voices_dir.glob("voices.bin.npz.*.tmp*"):
            with contextlib.suppress(FileNotFoundError):
                temp_path.unlink()
        if voices_bin_path.exists() and voices_bin_path.stat().st_size == 0:
            logger.warning(
                "voices.bin.npz is empty at %s; re-downloading voices.",
                voices_bin_path,
            )
            force_download = True
        if voices_bin_path.exists() and force_download:
            with contextlib.suppress(FileNotFoundError):
                voices_bin_path.unlink()

    # If voices.bin.npz already exists and not forcing, return early
    if voices_bin_path.exists() and not force_download:
        logger.info(f"voices.bin.npz already exists at {voices_bin_path}")
        return voices_dir

    # Download individual voice files (.bin format from HuggingFace)
    total = len(voice_names)
    downloaded_files = []

    for idx, voice_name in enumerate(voice_names):
        if progress_callback:
            progress_callback(voice_name, idx, total)

        voice_path = voices_dir / f"{voice_name}.bin"

        # Download if not exists or force
        if not voice_path.exists() or force:
            try:
                downloaded_path = _download_from_hf(
                    repo_id=repo_id,
                    filename=f"{HF_VOICES_SUBFOLDER}/{voice_name}.bin",
                    local_dir=None,
                    force=force,
                    min_size=MIN_VOICE_BIN_BYTES,
                    validator=_validate_voice_bin,
                )
                with _download_lock(voice_path):
                    _atomic_copy(downloaded_path, voice_path)
                logger.info(f"Downloaded {voice_name}.bin")
                downloaded_files.append(voice_name)
            except Exception as e:
                logger.warning(f"Failed to download {voice_name}.bin: {e}")
                continue
        else:
            downloaded_files.append(voice_name)

    # Load and combine all voices into a single .npz file (voices.bin.npz)
    if downloaded_files:
        logger.info(f"Combining {len(downloaded_files)} voices into voices.bin.npz")
        voices_data: dict[str, np.ndarray] = {}

        lengths: set[int] = set()
        for voice_name in downloaded_files:
            voice_path = voices_dir / f"{voice_name}.bin"
            try:
                # HuggingFace .bin files are raw float32 arrays
                voice_data = np.fromfile(str(voice_path), dtype=np.float32)
                if voice_data.size % 256 != 0:
                    raise ValueError(
                        f"Voice file length {voice_data.size} not divisible by 256"
                    )
                voice_data = voice_data.reshape(-1, 256)
                voice_array = normalize_voice_style(
                    voice_data,
                    expected_length=None,
                    require_dtype=True,
                    voice_name=voice_name,
                )
                lengths.add(voice_array.shape[0])
                voices_data[voice_name] = voice_array
            except Exception as e:
                logger.warning(f"Failed to load {voice_name}.bin: {e}")

        if lengths and len(lengths) > 1:
            logger.debug(
                "Downloaded voices have mixed lengths: %s. "
                "Voices will be normalized to a common length on load.",
                ", ".join(str(length) for length in sorted(lengths)),
            )

        if voices_data:
            np_savez = cast(Any, np.savez)
            with tempfile.NamedTemporaryFile(
                delete=False,
                dir=voices_bin_path.parent,
                prefix=f"{voices_bin_path.name}.",
                suffix=".tmp.npz",
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
            try:
                np_savez(str(tmp_path), **voices_data)
                os.replace(tmp_path, voices_bin_path)
            except Exception:
                with contextlib.suppress(FileNotFoundError):
                    tmp_path.unlink()
                raise
            logger.info(
                f"Created combined voices.bin.npz with {len(voices_data)} voices"
            )
        else:
            raise RuntimeError(
                "No valid voice files could be loaded. "
                "Check your network connection or clear the cache and retry."
            )

    return voices_dir


def download_all_models(
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
    quality: ModelQuality = DEFAULT_MODEL_QUALITY,
    progress_callback: Callable[[str, int, int], None] | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """
    Download config, model, and all voice files for HuggingFace source.

    Args:
        variant: Model variant (v1.0 or v1.1-zh)
        quality: Model quality/quantization level
        progress_callback: Optional callback (filename, current, total)
        force: Force re-download even if files exist

    Returns:
        Dict mapping filename to path
    """
    paths: dict[str, Path] = {}

    # Download config
    if progress_callback:
        progress_callback("config.json", 0, 3)
    paths["config.json"] = download_config(variant=variant, force=force)

    # Download model
    if progress_callback:
        progress_callback("model", 1, 3)
    model_path = download_model(variant=variant, quality=quality, force=force)
    paths[model_path.name] = model_path

    # Download all voices
    if progress_callback:
        progress_callback("voices", 2, 3)
    voices_dir = download_all_voices(
        variant=variant, progress_callback=None, force=force
    )
    paths["voices"] = voices_dir

    if progress_callback:
        progress_callback("complete", 3, 3)

    return paths


# ============================================================================
# GitHub Download Functions
# ============================================================================


def _download_from_github(
    url: str,
    local_path: Path,
    force: bool = False,
    min_size: int | None = None,
    validator: Callable[[Path], None] | None = None,
    timeout: float = DOWNLOAD_TIMEOUT_SECONDS,
    retries: int = DOWNLOAD_RETRIES,
    lock_timeout: float = DOWNLOAD_LOCK_TIMEOUT_SECONDS,
) -> Path:
    """
    Download a file from GitHub releases using urllib.

    Args:
        url: Full URL to the file
        local_path: Local path to save the file
        force: Force re-download even if file exists

    Returns:
        Path to the downloaded file
    """
    # Check if file already exists
    if local_path.exists() and not force:
        logger.debug(f"File already exists: {local_path}")
        return local_path

    logger.info(f"Downloading from {url} to {local_path}")

    def _download() -> Path:
        return _stream_download(
            url,
            local_path,
            timeout=timeout,
            min_size=min_size,
            validator=validator,
        )

    with _download_lock(local_path, timeout=lock_timeout):
        if local_path.exists() and not force:
            logger.debug(f"File already exists: {local_path}")
            return local_path

        return _run_with_retries(
            _download,
            description=f"Download {local_path.name}",
            retries=retries,
        )


def download_model_github(
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
    quality: ModelQuality = DEFAULT_MODEL_QUALITY,
    force: bool = False,
) -> Path:
    """
    Download a model file from GitHub releases.

    Args:
        variant: Model variant (v1.0 for English, v1.1-zh for Chinese)
        quality: Model quality/quantization level
        force: Force re-download even if file exists

    Returns:
        Path to the downloaded model file

    Raises:
        ValueError: If quality is not available for the variant
    """
    # Get the appropriate quality mapping and base URL
    if variant == "v1.0":
        quality_files = MODEL_QUALITY_FILES_GITHUB_V1_0
        base_url = GITHUB_BASE_URL_V1_0
    elif variant == "v1.1-zh":
        quality_files = MODEL_QUALITY_FILES_GITHUB_V1_1_ZH
        base_url = GITHUB_BASE_URL_V1_1_ZH
    elif variant == "v1.1-de":
        quality_files = MODEL_QUALITY_FILES_GITHUB_V1_1_DE
        base_url = GITHUB_BASE_URL_V1_1_DE
    else:
        raise ValueError(f"Unknown model variant: {variant}")

    # Check if quality is available for this variant
    if quality not in quality_files:
        available = ", ".join(quality_files.keys())
        raise ValueError(
            f"Quality '{quality}' not available for variant '{variant}'. "
            f"Available qualities: {available}"
        )

    # Get filename and construct URL
    filename = quality_files[quality]
    url = f"{base_url}/{filename}"

    # Use new path structure
    model_dir = get_model_dir(source="github", variant=variant)
    local_path = model_dir / filename

    # Download
    return _download_from_github(
        url,
        local_path,
        force,
        validator=_validate_onnx_file,
    )


def download_voices_github(
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
    force: bool = False,
) -> Path:
    """
    Download voices.bin file from GitHub releases.

    Args:
        variant: Model variant (v1.0 for English, v1.1-zh for Chinese)
        force: Force re-download even if file exists

    Returns:
        Path to the downloaded voices.bin file
    """
    # Get the appropriate filename and base URL
    if variant == "v1.0":
        filename = GITHUB_VOICES_FILENAME_V1_0
        base_url = GITHUB_BASE_URL_V1_0
    elif variant == "v1.1-zh":
        filename = GITHUB_VOICES_FILENAME_V1_1_ZH
        base_url = GITHUB_BASE_URL_V1_1_ZH
    elif variant == "v1.1-de":
        filename = GITHUB_VOICES_FILENAME_V1_1_DE
        base_url = GITHUB_BASE_URL_V1_1_DE
    else:
        raise ValueError(f"Unknown model variant: {variant}")

    # Construct URL
    url = f"{base_url}/{filename}"

    # Use new path structure
    voices_dir = get_voices_dir(source="github", variant=variant)
    local_path = voices_dir / filename

    # Download
    return _download_from_github(
        url,
        local_path,
        force,
        validator=_validate_voice_archive,
    )


def download_all_models_github(
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
    quality: ModelQuality = DEFAULT_MODEL_QUALITY,
    progress_callback: Callable[[str, int, int], None] | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """
    Download model and voices files from GitHub.

    Args:
        variant: Model variant (v1.0 for English, v1.1-zh for Chinese)
        quality: Model quality/quantization level
        progress_callback: Optional callback (filename, current, total)
        force: Force re-download even if files exist

    Returns:
        Dict mapping filename to path
    """
    paths: dict[str, Path] = {}

    # Download model
    if progress_callback:
        progress_callback("model", 0, 2)
    model_path = download_model_github(variant, quality, force)
    paths[model_path.name] = model_path

    # Download voices
    if progress_callback:
        progress_callback("voices", 1, 2)
    voices_path = download_voices_github(variant, force)
    paths[voices_path.name] = voices_path

    if progress_callback:
        progress_callback("complete", 2, 2)

    return paths


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
        use_gpu: bool = False,
        provider: ProviderType | None = None,
        session_options: rt.SessionOptions | None = None,
        provider_options: dict[str, Any] | None = None,
        vocab_version: str = "v1.0",
        espeak_config: EspeakConfig | None = None,
        tokenizer_config: "TokenizerConfig | None" = None,
        model_quality: ModelQuality | None = None,
        model_source: ModelSource = DEFAULT_MODEL_SOURCE,
        model_variant: ModelVariant = DEFAULT_MODEL_VARIANT,
        short_sentence_config: "ShortSentenceConfig | None" = None,
    ) -> None:
        """
        Initialize the Kokoro ONNX backend.

        Args:
            model_path: Path to the ONNX model file (auto-downloaded if None)
            voices_path: Path to the voices.bin file (auto-downloaded if None)
            use_gpu: Deprecated. Use provider parameter instead.
                Legacy GPU flag for backward compatibility.
            provider: Execution provider for ONNX Runtime. Options:
                "auto" (auto-select best), "cpu", "cuda" (NVIDIA),
                "openvino" (Intel), "directml" (Windows), "coreml" (macOS)
            session_options: Pre-configured ONNX Runtime SessionOptions object.
                If provided, this takes precedence over provider_options.
                For advanced users who need full control over session configuration.
            provider_options: Dictionary of provider and session options.
                Supports both SessionOptions attributes and provider-specific options.

                Common SessionOptions attributes:
                - intra_op_num_threads: Parallelism within operations (default: auto)
                - inter_op_num_threads: Parallelism across operations (default: 1)
                - graph_optimization_level: 0-3 or GraphOptimizationLevel enum
                - execution_mode: Sequential or parallel
                - enable_profiling: Enable ONNX profiling

                Provider-specific options:

                OpenVINO:
                - device_type: "CPU_FP32", "GPU", etc.
                - precision: "FP32", "FP16", "BF16" (auto-set from model_quality)
                - num_of_threads: Number of threads (default: auto)
                - cache_dir: Model cache directory
                  (default: ~/.cache/pykokoro/openvino_cache)
                - enable_opencl_throttling: "true"/"false" for iGPU

                CUDA:
                - device_id: GPU device ID (default: 0)
                - gpu_mem_limit: Memory limit in bytes
                - arena_extend_strategy: "kNextPowerOfTwo", "kSameAsRequested"
                - cudnn_conv_algo_search: "EXHAUSTIVE", "HEURISTIC", "DEFAULT"

                DirectML:
                - device_id: GPU device ID
                - disable_metacommands: "true"/"false"

                CoreML:
                - MLComputeUnits: "ALL", "CPU_ONLY", "CPU_AND_GPU"
                - EnableOnSubgraphs: "true"/"false"

                Example:
                    provider_options={
                        "precision": "FP16",
                        "num_of_threads": 8,
                        "intra_op_num_threads": 4
                    }
            vocab_version: Vocabulary version for tokenizer
            espeak_config: Optional espeak-ng configuration
                (deprecated, use tokenizer_config)
            tokenizer_config: Optional tokenizer configuration
                (for mixed-language support)
            model_quality: Model quality/quantization level (default from config)
            model_source: Model source ("huggingface" or "github")
            model_variant: Model variant ("v1.0", "v1.1-zh")
            short_sentence_config: Configuration for short sentence handling using
                phoneme pretext. This improves audio quality for short sentences
                (like "Why?" or "Go!") by surrounding phonemes with context.
                If None, uses default thresholds (min_phoneme_length=30).
                Set enabled=False to disable.
                Example:
                    from pykokoro.short_sentence_handler import ShortSentenceConfig
                    config = ShortSentenceConfig(
                        min_phoneme_length=20,  # Treat < 20 phonemes as short
                        enabled=True,
                        phoneme_pretext="—"
                    )
                    tts = Kokoro(short_sentence_config=config)
        """
        self._session: rt.InferenceSession | None = None
        self._voice_manager: VoiceManager | None = None
        self._audio_generator: AudioGenerator | None = None
        self._np = np
        self._voices_path_provided = voices_path is not None

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
        self._auto_switched_variant = False  # Track if we auto-switched

        # Load config for defaults
        from .utils import load_config

        cfg = load_config()

        # Resolve provider_options from config if not specified
        if provider_options is None and "provider_options" in cfg:
            provider_options = cfg.get("provider_options")
            logger.info(f"Loaded provider_options from config: {provider_options}")

        self._provider_options = provider_options

        # Resolve model quality from config if not specified
        resolved_quality: ModelQuality = DEFAULT_MODEL_QUALITY
        if model_quality is not None:
            resolved_quality = model_quality
        else:
            quality_from_cfg = cfg.get("model_quality", DEFAULT_MODEL_QUALITY)
            # Validate it's a valid quality option and cast to ModelQuality
            if quality_from_cfg in MODEL_QUALITY_FILES:
                resolved_quality = quality_from_cfg

        # Validate quality is available for the selected source/variant
        if model_source == "github":
            if model_variant == "v1.0":
                available_qualities = MODEL_QUALITY_FILES_GITHUB_V1_0
            elif model_variant == "v1.1-zh":
                available_qualities = MODEL_QUALITY_FILES_GITHUB_V1_1_ZH
            elif model_variant == "v1.1-de":
                available_qualities = MODEL_QUALITY_FILES_GITHUB_V1_1_DE
            else:
                raise ValueError(f"Unknown model variant: {model_variant}")

            if resolved_quality not in available_qualities:
                available = ", ".join(available_qualities.keys())
                raise ValueError(
                    f"Quality '{resolved_quality}' not available for "
                    f"GitHub {model_variant}. Available qualities: {available}"
                )
        elif model_source == "huggingface":
            # Both v1.0 and v1.1-zh use same filename convention for HuggingFace
            if resolved_quality not in MODEL_QUALITY_FILES_HF:
                available = ", ".join(MODEL_QUALITY_FILES_HF.keys())
                raise ValueError(
                    f"Quality '{resolved_quality}' not available for "
                    f"HuggingFace {model_variant}. Available qualities: {available}"
                )

        self._model_quality: ModelQuality = resolved_quality

        # Resolve paths
        if model_path is None:
            model_path = get_model_path(
                quality=self._model_quality, source=model_source, variant=model_variant
            )

        if voices_path is None:
            if model_source == "huggingface":
                # HuggingFace uses voices.bin.npz for both variants
                voices_path = (
                    get_voices_dir("huggingface", model_variant) / "voices.bin.npz"
                )
            elif model_source == "github":
                # GitHub uses variant-specific filenames
                if model_variant == "v1.0":
                    filename = GITHUB_VOICES_FILENAME_V1_0
                elif model_variant == "v1.1-de":
                    filename = GITHUB_VOICES_FILENAME_V1_1_DE
                else:  # v1.1-zh
                    filename = GITHUB_VOICES_FILENAME_V1_1_ZH
                voices_path = get_voices_dir("github", model_variant) / filename

        self._model_path = model_path
        self._voices_path = voices_path

        # Voice database connection (for kokovoicelab integration)
        self._voice_db: sqlite3.Connection | None = None

        # Tokenizer for phoneme-based generation
        self._tokenizer: Tokenizer | None = None
        # Use model variant as vocab version for proper filtering
        self._vocab_version = self._model_variant
        self._espeak_config = espeak_config
        self._tokenizer_config = tokenizer_config

        # Short sentence handling configuration
        self._short_sentence_config = short_sentence_config

    def _get_vocabulary(self) -> dict[str, int]:
        """Get vocabulary for the current model variant.

        Returns:
            Dictionary mapping phoneme characters to token indices
        """
        from kokorog2p import get_kokoro_vocab

        # For GitHub models or v1.1-zh, load variant-specific vocab from config
        if self._model_source == "github" or self._model_variant == "v1.1-zh":
            return load_vocab_from_config(self._model_variant)

        # For HuggingFace v1.0 or default, use standard vocab
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
        # (Check if variant differs from default)
        if self._initial_model_variant != DEFAULT_MODEL_VARIANT:
            return self._model_variant

        # Auto-detect: Switch to v1.1-zh for Chinese
        if is_chinese_language(lang) and self._model_source == "github":
            if not self._auto_switched_variant:
                logger.info(
                    f"Detected Chinese language '{lang}'. "
                    f"Automatically switching to model variant 'v1.1-zh'."
                )
                self._auto_switched_variant = True
            return "v1.1-zh"

        # Otherwise use configured variant
        return self._model_variant

    @property
    def tokenizer(self) -> Tokenizer:
        """Get the tokenizer instance (lazily initialized).

        Uses variant-specific vocabulary for proper phoneme filtering.
        """
        if self._tokenizer is None:
            # Get variant-specific vocabulary
            vocab = self._get_vocabulary()

            logger.debug(
                f"Initializing tokenizer with {len(vocab)} tokens "
                f"for variant '{self._model_variant}'"
            )

            self._tokenizer = Tokenizer(
                config=self._tokenizer_config,
                espeak_config=self._espeak_config,
                vocab_version=self._vocab_version,
                vocab=vocab,  # Pass variant-specific vocabulary
            )
        return self._tokenizer

    def _ensure_models(self) -> None:
        """Ensure model, voice, and config files are downloaded for current variant."""
        # Download model if needed
        if not self._model_path.exists():
            if self._model_source == "github":
                download_model_github(
                    variant=self._model_variant, quality=self._model_quality
                )
            else:  # huggingface
                download_model(variant=self._model_variant, quality=self._model_quality)

        # Download voices if needed
        if not self._voices_path.exists() or self._voices_path.stat().st_size == 0:
            if self._model_source == "github":
                download_voices_github(variant=self._model_variant)
            else:  # huggingface
                download_all_voices(variant=self._model_variant)

        # Download variant-specific config if needed
        if self._model_source == "github":
            if not is_config_downloaded(variant=self._model_variant):
                logger.info(
                    f"Downloading config for variant '{self._model_variant}'..."
                )
                download_config(variant=self._model_variant)
        else:  # huggingface - default v1.0
            if not is_config_downloaded():
                download_config()

    def _redownload_voices(self, force: bool = False) -> None:
        if self._model_source == "github":
            download_voices_github(variant=self._model_variant, force=force)
            return

        download_all_voices(variant=self._model_variant, force=force)

    def _get_default_provider_options(self, provider: str) -> dict[str, str]:
        """
        Get sensible default options for a provider.

        Uses PyKokoro cache path and model quality for smart defaults.

        Args:
            provider: Provider name (e.g., "OpenVINOExecutionProvider")

        Returns:
            Dictionary of default provider options (string values)
        """
        cache_path = get_user_cache_path()
        return ProviderConfigManager.get_default_provider_options(
            provider=provider,
            model_quality=self._model_quality,
            cache_path=cache_path,
        )

    def _get_provider_specific_options(
        self,
        provider: str,
        all_options: dict[str, Any],
    ) -> dict[str, str]:
        """
        Extract provider-specific options for the given provider.

        Filters out SessionOptions attributes and converts values to strings
        as required by ONNX Runtime.

        Args:
            provider: Provider name (e.g., "OpenVINOExecutionProvider")
            all_options: Dictionary of all options (mixed session and provider options)

        Returns:
            Dictionary of provider-specific options with string values
        """
        return ProviderConfigManager.get_provider_specific_options(
            provider=provider,
            all_options=all_options,
        )

    def _apply_provider_options(
        self,
        sess_opt: rt.SessionOptions,
        options: dict[str, Any],
    ) -> None:
        """
        Apply provider options to SessionOptions.

        Handles both SessionOptions attributes and provider-specific configs.

        Args:
            sess_opt: SessionOptions to modify
            options: Dictionary of options to apply
        """
        # Map of common option names to SessionOptions attributes
        session_option_attrs: dict[str, str] = {
            "intra_op_num_threads": "intra_op_num_threads",
            "inter_op_num_threads": "inter_op_num_threads",
            "num_threads": "intra_op_num_threads",  # Alias
            "threads": "intra_op_num_threads",  # Alias
            "graph_optimization_level": "graph_optimization_level",
            "execution_mode": "execution_mode",
            "enable_profiling": "enable_profiling",
            "enable_mem_pattern": "enable_mem_pattern",
            "enable_cpu_mem_arena": "enable_cpu_mem_arena",
            "enable_mem_reuse": "enable_mem_reuse",
            "log_severity_level": "log_severity_level",
            "log_verbosity_level": "log_verbosity_level",
        }

        # Apply SessionOptions attributes
        for opt_name, value in options.items():
            if opt_name in session_option_attrs:
                attr_name = session_option_attrs[opt_name]
                setattr(sess_opt, attr_name, value)
                logger.debug(f"Set SessionOptions.{attr_name} = {value}")

    def _init_kokoro(self) -> None:
        """Initialize the ONNX session and load voices."""
        if self._session is not None:
            return

        self._ensure_models()

        # Use OnnxSessionManager to create session
        session_manager = OnnxSessionManager(
            provider=self._provider,
            use_gpu=self._use_gpu,
            session_options=self._session_options,
            provider_options=self._provider_options,
        )
        self._session = session_manager.create_session(model_path=self._model_path)

        # Use VoiceManager to load voices
        voice_manager = VoiceManager(model_source=self._model_source)
        try:
            voice_manager.load_voices(voices_path=self._voices_path)
        except ConfigurationError as exc:
            if self._voices_path_provided:
                raise
            logger.warning(
                "Voice archive invalid at %s: %s. Re-downloading...",
                self._voices_path,
                exc,
            )
            self._redownload_voices(force=True)
            voice_manager.load_voices(voices_path=self._voices_path)
        self._voice_manager = voice_manager

        # Create AudioGenerator
        self._audio_generator = AudioGenerator(
            session=self._session,
            tokenizer=self.tokenizer,
            model_source=self._model_source,
            short_sentence_config=self._short_sentence_config,
        )

    def get_voices(self) -> list[str]:
        """Get list of available voice names."""
        self._init_kokoro()
        assert self._voice_manager is not None
        return self._voice_manager.get_voices()

    def get_voice_style(self, voice_name: str) -> np.ndarray:
        """Get the style vector for a voice."""
        self._init_kokoro()
        assert self._voice_manager is not None
        return self._voice_manager.get_voice_style(voice_name)

    def create_blended_voice(self, blend: VoiceBlend) -> np.ndarray:
        """Create a blended voice style vector from a VoiceBlend."""
        self._init_kokoro()
        assert self._voice_manager is not None
        return self._voice_manager.create_blended_voice(blend)

    def _resolve_voice_style(self, voice: str | np.ndarray | VoiceBlend) -> np.ndarray:
        """Resolve voice parameter to a voice style array."""
        self._init_kokoro()
        assert self._voice_manager is not None
        return self._voice_manager.resolve_voice(
            voice,
            voice_db_lookup=self.get_voice_from_database,
        )

    def resolve_voice_style(self, voice: str | np.ndarray | VoiceBlend) -> np.ndarray:
        """Resolve voice parameter to a voice style array."""
        return self._resolve_voice_style(voice)

    def preprocess_segments(
        self,
        segments: list["PhonemeSegment"],
        enable_short_sentence_override: bool | None,
    ) -> list["PhonemeSegment"]:
        """Preprocess phoneme segments for short sentence handling."""
        self._init_kokoro()
        assert self._audio_generator is not None
        return self._audio_generator._preprocess_segments(
            segments, enable_short_sentence_override
        )

    def generate_raw_audio_segments(
        self,
        segments: list["PhonemeSegment"],
        voice_style: np.ndarray,
        speed: float,
        voice_resolver: Callable[[str], np.ndarray] | None,
    ) -> list["PhonemeSegment"]:
        """Generate raw audio for each phoneme segment."""
        self._init_kokoro()
        assert self._audio_generator is not None
        return self._audio_generator._generate_raw_audio_segments(
            segments, voice_style, speed, voice_resolver
        )

    def postprocess_audio_segments(
        self, segments: list["PhonemeSegment"], trim_silence: bool
    ) -> list["PhonemeSegment"]:
        """Trim/prosody-process raw audio segments."""
        self._init_kokoro()
        assert self._audio_generator is not None
        return self._audio_generator._postprocess_audio_segments(segments, trim_silence)

    def concatenate_audio_segments(
        self, segments: list["PhonemeSegment"]
    ) -> np.ndarray:
        """Concatenate processed segments into a single waveform."""
        self._init_kokoro()
        assert self._audio_generator is not None
        return self._audio_generator._concatenate_audio_segments(segments)

    # Voice Database Integration (from kokovoicelab)

    def load_voice_database(self, db_path: Path) -> None:
        """
        Load a voice database for custom/synthetic voices.

        Args:
            db_path: Path to the SQLite voice database
        """
        if self._voice_db is not None:
            self._voice_db.close()

        # Register numpy array converter
        sqlite3.register_converter("array", self._convert_array)
        self._voice_db = sqlite3.connect(
            str(db_path), detect_types=sqlite3.PARSE_DECLTYPES
        )

    def _convert_array(self, blob: bytes) -> np.ndarray:
        """Convert binary blob back to numpy array."""
        out = io.BytesIO(blob)
        return np.load(out)

    def get_voice_from_database(self, voice_name: str) -> np.ndarray | None:
        """
        Get a voice style vector from the database.

        Args:
            voice_name: Name of the voice in the database

        Returns:
            Voice style vector or None if not found
        """
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
        """
        List all voices in the database.

        Returns:
            List of voice metadata dictionaries
        """
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
        """
        Interpolate between two voices.

        This uses the interpolation method from kokovoicelab to create
        voices that lie on the line between two source voices.

        Args:
            voice1: First voice (name or style vector)
            voice2: Second voice (name or style vector)
            factor: Interpolation factor (0.0 = voice1, 1.0 = voice2)

        Returns:
            Interpolated voice style vector
        """
        self._init_kokoro()

        self._init_kokoro()
        assert self._voice_manager is not None

        style1 = self._voice_manager.resolve_voice(
            voice1, voice_db_lookup=self.get_voice_from_database
        )
        style2 = self._voice_manager.resolve_voice(
            voice2, voice_db_lookup=self.get_voice_from_database
        )

        # Use kokovoicelab's interpolation method
        diff_vector = style2 - style1
        midpoint = (style1 + style2) / 2
        return midpoint + (diff_vector * factor / 2)

    def _generate_from_segments(
        self,
        segments: list["PhonemeSegment"],
        voice_style: np.ndarray,
        speed: float,
        trim_silence: bool,
        enable_short_sentence_override: bool | None = None,
    ) -> np.ndarray:
        """Delegate to AudioGenerator with voice resolution support.

        This wrapper provides voice resolution for per-segment voice switching
        via SSMD voice annotations.
        """
        self._init_kokoro()
        assert self._audio_generator is not None
        audio_generator = self._audio_generator

        # Create voice resolver callback
        def voice_resolver(voice_name: str) -> np.ndarray:
            """Resolve voice name to style vector."""
            assert self._voice_manager is not None
            return self._voice_manager.resolve_voice(
                voice_name, voice_db_lookup=self.get_voice_from_database
            )

        return audio_generator.generate_from_segments(
            segments,
            voice_style,
            speed,
            trim_silence,
            voice_resolver=voice_resolver,
            enable_short_sentence_override=enable_short_sentence_override,
        )

    def close(self) -> None:
        """Clean up resources."""
        if self._voice_db is not None:
            self._voice_db.close()
            self._voice_db = None


def is_chinese_language(lang: str) -> bool:
    """Check if language code is Chinese.

    Args:
        lang: Language code (e.g., 'zh', 'cmn', 'zh-cn')

    Returns:
        True if language is Chinese, False otherwise
    """
    lang_lower = lang.lower().strip()
    return lang_lower in ["zh", "cmn", "zh-cn", "zh-tw", "zh-hans", "zh-hant"]
