"""ONNX backend for pykokoro - native ONNX TTS without external dependencies."""

import contextlib
import hashlib
import io
import json
import logging
import os
import shutil
import sqlite3
import tempfile
import time
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import onnxruntime as rt

from .artifact_manifest import (
    hf_config_spec,
    hf_model_spec,  # noqa: F401
    hf_voice_spec,
)
from .asset_constants import (
    HF_CONFIG_FILENAME,
    HF_MODEL_SUBFOLDER,
    MODEL_QUALITY_CACHE_FILES_HF_V1_0,  # noqa: F401
    MODEL_QUALITY_FILES,
    MODEL_QUALITY_FILES_HF,
)
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
from .constants import SAMPLE_RATE
from .exceptions import ConfigurationError
from .model_assets import (
    _is_nonempty_file,
    get_voices_archive_path,
    installed_manifest_path,
    installed_sidecar_path,
    release_asset_path,
)
from .model_assets import (
    are_models_downloaded as _are_models_downloaded,
)
from .model_assets import (
    are_voices_downloaded as _are_voices_downloaded,
)
from .model_assets import (
    is_model_downloaded as _is_model_downloaded,
)
from .model_profiles import VOICE_ALIASES, get_model_profile
from .model_registry import ModelRegistryError
from .onnx_session import OnnxSessionManager
from .provider_config import ProviderConfigManager
from .release_catalog import (
    MODEL_REPOSITORY,
    ReleaseAsset,
    RemoteModelRelease,
    resolve_model_release,  # noqa: F401
)
from .runtime.dispatcher import create_runtime
from .runtime.model_assets import (
    ResolvedRuntimeAssets,
    resolve_runtime_assets,
)
from .tokenizer import EspeakConfig, Tokenizer, TokenizerConfig
from .utils import get_user_cache_path
from .voice_manager import VoiceBlend, VoiceManager, normalize_voice_style

if TYPE_CHECKING:
    from .prosody_config import ProsodyConfig
    from .short_sentence_handler import ShortSentenceConfig
    from .types import PhonemeSegment, Trace

# Logger for debugging
logger = logging.getLogger(__name__)
_DEFAULT_HF_MODEL_SPEC = hf_model_spec
_DEFAULT_RESOLVE_MODEL_RELEASE = resolve_model_release

DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_RETRIES = 3
DOWNLOAD_TIMEOUT_SECONDS = 30
DOWNLOAD_LOCK_TIMEOUT_SECONDS = 30
MIN_ONNX_BYTES = 1_000_000
MIN_VOICE_ARCHIVE_BYTES = 64 * 1024
MIN_VOICE_BIN_BYTES = 100_000
MIN_CONFIG_BYTES = 100


def _hf_hub_download(**kwargs: Any) -> str:
    """Load and call the Hugging Face client only for Hugging Face downloads."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(**kwargs)


class ArtifactValidationError(RuntimeError):
    """Raised when a cached or downloaded artifact fails integrity checks."""


# GitHub release discovery is centralized in release_catalog.py.
# Direct Hugging Face compatibility downloads use the same pinned sources as the registry.
HF_REPO_V1_0 = "onnx-community/Kokoro-82M-v1.0-ONNX"
HF_REPO_V1_1_ZH = "onnx-community/Kokoro-82M-v1.1-zh-ONNX"
HF_CONFIG_REPO_V1_0 = "hexgrad/Kokoro-82M"
HF_CONFIG_REPO_V1_1_ZH = "hexgrad/Kokoro-82M-v1.1-zh"
HF_VOICES_SUBFOLDER = "voices"
# All available voice names for v1.0 (54 voices - English/multilingual)
# Used by both HuggingFace and GitHub sources
# These are used for downloading individual voice files from HuggingFace
VOICE_NAMES_V1_0 = [
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


# Voice name documentation by language/variant
# These voices are dynamically loaded from the model's voices.bin file
# The actual available voices may vary depending on the model source and variant
VOICE_NAMES_BY_VARIANT = {
    "v1.0": VOICE_NAMES_V1_0,  # Same as HuggingFace (multi-language)
    "v1.1-zh": VOICE_NAMES_V1_1_ZH,  # Chinese-specific voices
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
        variant: Model variant (v1.0, v1.1-zh)

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
        variant: Model variant (v1.0 or v1.1-zh)

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


def get_vocabulary_path(variant: ModelVariant) -> Path:
    profile = get_model_profile(variant, "github")
    filename = (
        HF_CONFIG_FILENAME
        if profile.vocabulary_source != "downloaded-release"
        else "vocabulary.json"
    )
    path = get_user_cache_path("config") / variant / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_voices_bin_path() -> Path:
    """Get the legacy global voice archive path.

    New code should use :func:`get_voices_archive_path` with an explicit source
    and variant instead.
    """
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
    if source == "github":
        return model_dir / f"{quality}.onnx"
    profile = get_model_profile(variant, source)
    try:
        filename = profile.quality_files[quality]
    except KeyError as exc:
        available = ", ".join(profile.quality_files.keys())
        raise ValueError(
            f"Quality '{quality}' not available for {source}/{variant}. Available: {available}"
        ) from exc
    return model_dir / HF_MODEL_SUBFOLDER / filename


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


def is_model_downloaded(
    quality: ModelQuality = DEFAULT_MODEL_QUALITY,
    source: ModelSource = DEFAULT_MODEL_SOURCE,
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
) -> bool:
    """Check if the requested source, variant, and quality model is downloaded."""
    return _is_model_downloaded(quality, source, variant)


def is_voice_downloaded(voice_name: str) -> bool:
    """Check if an individual voice file is already downloaded."""
    voice_path = get_voice_path(voice_name)
    return voice_path.exists() and voice_path.stat().st_size > 0


def are_voices_downloaded(
    source: ModelSource = DEFAULT_MODEL_SOURCE,
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
) -> bool:
    """Check if the requested source and variant voice archive is downloaded."""
    return _are_voices_downloaded(source, variant)


def are_models_downloaded(
    quality: ModelQuality = DEFAULT_MODEL_QUALITY,
    source: ModelSource = DEFAULT_MODEL_SOURCE,
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
) -> bool:
    """Check if the requested config, model, and voice archive are downloaded."""
    return _are_models_downloaded(quality, source, variant)


# =============================================================================
# Download functions
# =============================================================================


def _huggingface_repo_for_variant(variant: ModelVariant) -> str:
    repositories = {
        "v1.0": HF_REPO_V1_0,
        "v1.1-zh": HF_REPO_V1_1_ZH,
    }
    try:
        return repositories[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown variant: {variant}") from exc


def _validate_min_size(path: Path, min_size: int) -> None:
    size = path.stat().st_size
    if size < min_size:
        raise ArtifactValidationError(
            f"Downloaded file {path.name} is too small ({size} bytes). "
            f"Expected at least {min_size} bytes."
        )


def _validate_exact_size(path: Path, expected_size: int | None) -> None:
    if expected_size is None:
        return
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ArtifactValidationError(
            f"Downloaded file {path.name} has size {actual_size} bytes; "
            f"expected exactly {expected_size} bytes."
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
        raise ArtifactValidationError(
            f"Downloaded ONNX model '{path.name}' is invalid: {exc}"
        ) from exc


def _validate_voice_archive(
    path: Path,
    *,
    expected_voice_names: tuple[str, ...] | None = None,
) -> None:
    _validate_min_size(path, MIN_VOICE_ARCHIVE_BYTES)

    try:
        with np.load(str(path), allow_pickle=False) as voices:
            if not voices.files:
                raise RuntimeError("Voice archive is empty")
            if expected_voice_names is not None:
                missing = sorted(set(expected_voice_names) - set(voices.files))
                if missing:
                    raise RuntimeError(
                        "Voice archive is missing expected voices: " + ", ".join(missing)
                    )
            for voice_name in expected_voice_names or (voices.files[0],):
                normalize_voice_style(
                    voices[voice_name],
                    expected_length=None,
                    require_dtype=True,
                    voice_name=voice_name,
                )
    except Exception as exc:
        raise ArtifactValidationError(
            f"Downloaded voice archive '{path.name}' is invalid: {exc}"
        ) from exc


def _validate_voice_bin(path: Path) -> None:
    _validate_min_size(path, MIN_VOICE_BIN_BYTES)

    size = path.stat().st_size
    if size % 4 != 0:
        raise ArtifactValidationError(
            f"Downloaded voice file '{path.name}' has invalid byte size {size}."
        )


def _validate_sha256(path: Path, expected_sha256: str | None) -> None:
    if expected_sha256 is None:
        return
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(DOWNLOAD_CHUNK_SIZE), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual.lower() != expected_sha256.lower():
        raise ArtifactValidationError(
            f"Artifact '{path.name}' has checksum {actual}; expected {expected_sha256}."
        )


def _validate_artifact(
    path: Path,
    *,
    min_size: int | None = None,
    validator: Callable[[Path], None] | None = None,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> None:
    if min_size is not None:
        _validate_min_size(path, min_size)
    _validate_exact_size(path, expected_size)
    # A pinned digest rejects known-wrong large files before expensive ONNX parsing.
    _validate_sha256(path, expected_sha256)
    if validator is not None:
        validator(path)


def _remove_invalid_cached_file(
    path: Path,
    *,
    min_size: int | None = None,
    validator: Callable[[Path], None] | None = None,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        _validate_artifact(
            path,
            min_size=min_size,
            validator=validator,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        )
    except (OSError, ArtifactValidationError) as exc:
        logger.warning("Removing invalid cached artifact %s: %s", path, exc)
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        return True
    return False


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
            with os.fdopen(fd, "w", encoding="utf-8") as lock_file:
                json.dump({"pid": os.getpid(), "created_at": time.time()}, lock_file)
            break
        except FileExistsError as e:
            if _is_stale_download_lock(lock_path, timeout):
                logger.warning("Recovering stale download lock %s", lock_path)
                with contextlib.suppress(FileNotFoundError):
                    lock_path.unlink()
                continue
            if time.monotonic() - start > timeout:
                raise RuntimeError(f"Timed out waiting for download lock on {target_path}") from e
            time.sleep(0.1)

    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            lock_path.unlink()


def _is_stale_download_lock(lock_path: Path, timeout: float) -> bool:
    try:
        metadata = json.loads(lock_path.read_text(encoding="utf-8"))
        pid = int(metadata["pid"])
        created_at = float(metadata["created_at"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    if time.time() - created_at <= timeout:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    except OSError as exc:
        # On Windows, probing a nonexistent PID with ``os.kill(pid, 0)``
        # raises ERROR_INVALID_PARAMETER instead of ProcessLookupError.
        return getattr(exc, "winerror", None) == 87
    return False


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
        except (OSError, ArtifactValidationError) as exc:
            last_error = exc
            if attempt == retries:
                break
            delay = 2 ** (attempt - 1)
            logger.warning(
                f"{description} failed on attempt {attempt}/{retries}: {exc}. Retrying in {delay}s."
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
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)

    part_path = local_path.with_suffix(local_path.suffix + ".part")
    resume_from = part_path.stat().st_size if part_path.is_file() else 0
    request: urllib.request.Request | str = url
    if resume_from:
        request = urllib.request.Request(url, headers={"Range": f"bytes={resume_from}-"})

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", None)
            content_range = (
                response.headers.get("Content-Range", "") if hasattr(response, "headers") else ""
            )
            can_resume = bool(
                resume_from and status == 206 and content_range.startswith(f"bytes {resume_from}-")
            )
            mode = "ab" if can_resume else "wb"
            with part_path.open(mode) as part_file:
                for chunk in iter(lambda: response.read(DOWNLOAD_CHUNK_SIZE), b""):
                    part_file.write(chunk)
                part_file.flush()
                os.fsync(part_file.fileno())
    except Exception:
        # Preserve a partial transfer so a retry can use HTTP Range when supported.
        raise

    try:
        _validate_artifact(
            part_path,
            min_size=min_size,
            validator=validator,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        )
        os.replace(part_path, local_path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            part_path.unlink()
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
    local_filename: str | None = None,
    force: bool = False,
    min_size: int | None = None,
    validator: Callable[[Path], None] | None = None,
    retries: int = DOWNLOAD_RETRIES,
    revision: str | None = None,
    expected_sha256: str | None = None,
    offline: bool = False,
) -> Path:
    """
    Download a file from Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository ID
        filename: File to download
        subfolder: Subfolder in the repository
        local_dir: Local directory to save to
        local_filename: Optional filename to use under local_dir
        force: Force re-download even if file exists

    Returns:
        Path to the downloaded file
    """
    # Use hf_hub_download to download the file
    # It handles caching automatically
    local_dir_path = Path(local_dir) if local_dir else None
    target_path: Path | None = None
    if local_dir_path is not None:
        target_filename = local_filename or filename
        target_path = local_dir_path / target_filename
        if subfolder:
            target_path = local_dir_path / subfolder / target_filename
        target_path.parent.mkdir(parents=True, exist_ok=True)

    force_download = force

    def _download() -> Path:
        nonlocal force_download
        use_local_dir = local_dir_path is not None and local_filename is None
        if offline:
            raise RuntimeError(
                f"Offline mode is enabled and {filename} is not available in the cache."
            )
        downloaded_path = _hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            local_dir=str(local_dir_path) if use_local_dir else None,
            force_download=force_download,
            revision=revision,
        )
        downloaded = Path(downloaded_path)
        delete_on_error = use_local_dir
        try:
            _validate_artifact(
                downloaded,
                min_size=min_size,
                validator=validator,
                expected_sha256=expected_sha256,
            )
            if target_path is not None and downloaded != target_path:
                _atomic_copy(downloaded, target_path)
                downloaded = target_path
                delete_on_error = True
        except (OSError, ArtifactValidationError):
            force_download = True
            if delete_on_error:
                with contextlib.suppress(FileNotFoundError):
                    downloaded.unlink()
            raise
        return downloaded

    if target_path is not None:
        with _download_lock(target_path):
            if (
                target_path.exists()
                and not force
                and not _remove_invalid_cached_file(
                    target_path,
                    min_size=min_size,
                    validator=validator,
                    expected_sha256=expected_sha256,
                )
            ):
                return target_path
            if offline:
                raise RuntimeError(f"Offline mode is enabled and {target_path.name} is not cached.")
            return _run_with_retries(
                _download, description=f"HF download of {filename}", retries=retries
            )

    if offline:
        raise RuntimeError(f"Offline mode is enabled and {filename} is not cached.")
    return _run_with_retries(_download, description=f"HF download of {filename}", retries=retries)


def download_config(
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
    force: bool = False,
    *,
    revision: str | None = None,
    sha256: str | None = None,
    offline: bool = False,
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

    # Select hexgrad repo based on variant
    config_repositories = {
        "v1.0": HF_CONFIG_REPO_V1_0,
        "v1.1-zh": HF_CONFIG_REPO_V1_1_ZH,
    }
    try:
        repo_id = config_repositories[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown variant: {variant}") from exc

    if revision is None:
        try:
            spec = hf_config_spec(variant)
        except KeyError as exc:
            raise ValueError(
                f"No immutable artifact manifest is available for config variant {variant!r}; "
                "pass an explicit revision and sha256."
            ) from exc
        revision = spec.revision
        if sha256 is None:
            sha256 = spec.sha256

    logger.info(f"Downloading config for {variant} from {repo_id} at revision {revision}")

    return _download_from_hf(
        repo_id=repo_id,
        filename=HF_CONFIG_FILENAME,
        local_dir=config_path.parent,
        force=force,
        min_size=MIN_CONFIG_BYTES,
        revision=revision,
        expected_sha256=sha256,
        offline=offline,
    )


def _validate_vocabulary(path: Path) -> None:
    _validate_min_size(path, MIN_CONFIG_BYTES)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactValidationError(f"Vocabulary file {path.name} is invalid: {exc}") from exc
    if not isinstance(data, dict) or not all(
        isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool)
        for key, value in data.items()
    ):
        raise ArtifactValidationError(
            f"Vocabulary file {path.name} must map string phonemes to integer IDs"
        )


def _record_installed_release(release: RemoteModelRelease, quality: str) -> None:
    manifest_path = installed_manifest_path(release)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_data = json.dumps(release.manifest, indent=2, ensure_ascii=False).encode() + b"\n"
    manifest_path.write_bytes(manifest_data)
    model_asset = release.model_asset(quality)
    voice_asset = release.voice_asset()
    auxiliary = [asset.name for asset in release.assets if asset.role not in {"model", "voices"}]
    sidecar = {
        "install_format": 1,
        "repository": MODEL_REPOSITORY,
        "profile": release.profile,
        "model_version": release.model_version,
        "release_tag": release.release_tag,
        "release_published_at": release.release_published_at,
        "manifest_schema": release.manifest_schema,
        "runtime_contract": release.runtime_contract,
        "quality": quality,
        "model_asset": model_asset.name,
        "voices_asset": voice_asset.name,
        "auxiliary_assets": auxiliary,
        "installed_at": datetime.now(UTC).isoformat(),
        "manifest_sha256": hashlib.sha256(manifest_data).hexdigest(),
    }
    installed_sidecar_path(release).write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _download_release_asset(
    release: RemoteModelRelease,
    asset: ReleaseAsset,
    *,
    quality: str,
    force: bool,
    validator: Callable[[Path], None] | None = None,
    offline: bool = False,
) -> Path:
    path = release_asset_path(release, asset)
    result = _download_from_github(
        asset.download_url,
        path,
        force,
        validator=validator,
        expected_sha256=asset.sha256,
        expected_size=asset.size,
        offline=offline,
    )
    _record_installed_release(release, quality)
    return result


def download_vocabulary_github(
    variant: ModelVariant,
    force: bool = False,
    offline: bool = False,
    *,
    tag: str | None = None,
) -> Path:
    del tag
    return _download_registry_artifact(
        variant,
        "vocab",
        force=force,
        offline=offline,
        preference="github",
        validator=_validate_vocabulary,
    )


def _validate_json_asset(path: Path) -> None:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactValidationError(f"JSON asset {path.name} is invalid: {exc}") from exc


def download_release_auxiliary(
    variant: ModelVariant,
    role: str,
    force: bool = False,
    offline: bool = False,
    *,
    tag: str | None = None,
) -> Path:
    del tag
    validator = _validate_vocabulary if role == "vocab" else _validate_json_asset
    return _download_registry_artifact(
        variant,
        role,
        force=force,
        offline=offline,
        preference="github",
        validator=validator,
    )


def download_config_github(
    variant: ModelVariant,
    force: bool = False,
    offline: bool = False,
    *,
    tag: str | None = None,
) -> Path:
    return download_release_auxiliary(variant, "config", force, offline, tag=tag)


def download_bundle_github(
    variant: ModelVariant,
    force: bool = False,
    offline: bool = False,
    *,
    tag: str | None = None,
) -> Path:
    return download_release_auxiliary(variant, "bundle", force, offline, tag=tag)


def load_vocab_from_config(
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
    config_path: Path | None = None,
) -> dict[str, int]:
    """Load vocabulary from variant-specific config.json.

    Args:
        variant: Model variant (v1.0 or v1.1-zh)

    Returns:
        Dictionary mapping phoneme characters to token indices

    Raises:
        FileNotFoundError: If config file doesn't exist after download
        ValueError: If config doesn't contain vocab
    """
    import json

    from kokorog2p import get_kokoro_vocab

    explicit_config_path = config_path is not None
    config_path = config_path or get_config_path(variant)

    # Download if not exists
    if not config_path.exists():
        if explicit_config_path:
            raise FileNotFoundError(f"Explicit model config does not exist: {config_path}")
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
    except (OSError, ValueError) as e:
        logger.error(
            f"Failed to load config from {config_path}: {e}. Falling back to default vocabulary."
        )
        return get_kokoro_vocab()

    # Extract vocabulary
    # Release vocabularies are direct JSON maps; legacy configs nest the map under vocab.
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
        f"Loaded vocabulary with {len(vocab)} tokens "
        f"for variant '{variant}' from {config_path.name}"
    )
    return vocab


def _download_legacy_hf_model(
    variant: ModelVariant,
    quality: ModelQuality,
    force: bool,
    revision: str | None,
    sha256: str | None,
    offline: bool,
) -> Path:
    repositories = {"v1.0": HF_REPO_V1_0, "v1.1-zh": HF_REPO_V1_1_ZH}
    filenames = dict(MODEL_QUALITY_FILES_HF)
    if variant == "v1.1-zh":
        filenames = {
            key: value for key, value in filenames.items() if key not in {"q8f16", "uint8f16"}
        }
        filenames.update({"int8": "model_int8.onnx", "bnb4": "model_bnb4.onnx"})
    if quality not in filenames:
        raise ValueError(f"Quality {quality!r} is not available for {variant}")
    filename = filenames[quality]
    if revision is None:
        spec = hf_model_spec(variant, filename)
        revision = spec.revision
        if sha256 is None:
            sha256 = spec.sha256
    return _download_from_hf(
        repo_id=repositories[variant],
        filename=filename,
        subfolder=HF_MODEL_SUBFOLDER,
        force=force,
        validator=_validate_onnx_file,
        revision=revision,
        expected_sha256=sha256,
        offline=offline,
    )


def download_model(
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
    quality: ModelQuality = DEFAULT_MODEL_QUALITY,
    force: bool = False,
    *,
    revision: str | None = None,
    sha256: str | None = None,
    offline: bool = False,
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
    if hf_model_spec is not _DEFAULT_HF_MODEL_SPEC:
        return _download_legacy_hf_model(variant, quality, force, revision, sha256, offline)
    del revision, sha256
    return _download_registry_artifact(
        variant,
        "model",
        quality=quality,
        force=force,
        offline=offline,
        preference="huggingface",
        validator=_validate_onnx_file,
    )


def download_voice(
    voice_name: str,
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
    force: bool = False,
    *,
    revision: str | None = None,
    sha256: str | None = None,
    offline: bool = False,
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
    repo_id = _huggingface_repo_for_variant(variant)

    filename = f"{voice_name}.bin"
    if revision is None:
        spec = hf_voice_spec(variant, filename)
        revision = spec.revision
        if sha256 is None:
            sha256 = spec.sha256
    # Use new path structure
    voices_dir = get_voices_dir(source="huggingface", variant=variant)
    voices_dir.mkdir(parents=True, exist_ok=True)
    local_path = voices_dir / filename

    if (
        local_path.exists()
        and not force
        and not _remove_invalid_cached_file(
            local_path,
            min_size=MIN_VOICE_BIN_BYTES,
            validator=_validate_voice_bin,
            expected_sha256=sha256,
        )
    ):
        logger.debug(f"Voice already exists: {local_path}")
        return local_path
    if offline:
        raise RuntimeError(f"Offline mode is enabled and {local_path.name} is not cached.")

    logger.info(f"Downloading voice {voice_name} for {variant}")

    download_name = f"{HF_VOICES_SUBFOLDER}/{filename}"
    downloaded_path = _download_from_hf(
        repo_id=repo_id,
        filename=download_name,
        local_dir=None,
        force=force,
        min_size=MIN_VOICE_BIN_BYTES,
        validator=_validate_voice_bin,
        revision=revision,
        expected_sha256=sha256,
        offline=offline,
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
    repo_id = _huggingface_repo_for_variant(variant)
    if variant == "v1.0":
        voice_names = VOICE_NAMES_V1_0
    elif variant == "v1.1-zh":
        voice_names = VOICE_NAMES_V1_1_ZH
    else:
        raise ValueError(f"Unknown variant: {variant}")

    voices_dir = get_voices_dir(source="huggingface", variant=variant)
    voices_dir.mkdir(parents=True, exist_ok=True)

    voices_bin_path = get_voices_archive_path("huggingface", variant)
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
            except (RuntimeError, OSError) as e:
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
                    raise ValueError(f"Voice file length {voice_data.size} not divisible by 256")
                voice_data = voice_data.reshape(-1, 256)
                voice_array = normalize_voice_style(
                    voice_data,
                    expected_length=None,
                    require_dtype=True,
                    voice_name=voice_name,
                )
                lengths.add(voice_array.shape[0])
                voices_data[voice_name] = voice_array
            except (RuntimeError, OSError, ValueError) as e:
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
            logger.info(f"Created combined voices.bin.npz with {len(voices_data)} voices")
        else:
            raise RuntimeError(
                "No valid voice files could be loaded. "
                "Check your network connection or clear the cache and retry."
            )

    return voices_dir


def _download_hf_voice_archive(
    variant: ModelVariant,
    *,
    force: bool = False,
) -> Path:
    """Download Hugging Face voices and return the canonical archive file."""
    download_all_voices(variant=variant, force=force)
    archive_path = get_voices_archive_path("huggingface", variant)
    if not _is_nonempty_file(archive_path):
        raise RuntimeError(f"Hugging Face voice download completed without creating {archive_path}")
    return archive_path


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
    voices_dir = download_all_voices(variant=variant, progress_callback=None, force=force)
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
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    offline: bool = False,
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
    if (
        local_path.exists()
        and not force
        and not _remove_invalid_cached_file(
            local_path,
            min_size=min_size,
            validator=validator,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        )
    ):
        logger.debug(f"File already exists: {local_path}")
        return local_path
    if offline:
        raise RuntimeError(f"Offline mode is enabled and {local_path.name} is not cached.")

    logger.info(f"Downloading from {url} to {local_path}")

    if force:
        with contextlib.suppress(FileNotFoundError):
            local_path.with_suffix(local_path.suffix + ".part").unlink()

    def _download() -> Path:
        return _stream_download(
            url,
            local_path,
            timeout=timeout,
            min_size=min_size,
            validator=validator,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        )

    with _download_lock(local_path, timeout=lock_timeout):
        if (
            local_path.exists()
            and not force
            and not _remove_invalid_cached_file(
                local_path,
                min_size=min_size,
                validator=validator,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
            )
        ):
            logger.debug(f"File already exists: {local_path}")
            return local_path
        if offline:
            raise RuntimeError(f"Offline mode is enabled and {local_path.name} is not cached.")

        return _run_with_retries(
            _download,
            description=f"Download {local_path.name}",
            retries=retries,
        )


def _download_registry_artifact(
    variant: ModelVariant,
    role: str,
    *,
    quality: ModelQuality | None = None,
    force: bool = False,
    offline: bool = False,
    validator: Callable[[Path], None] | None = None,
    preference: Literal["auto", "github", "huggingface", "upstream"] = "github",
) -> Path:
    try:
        resolved = resolve_runtime_assets(
            model_id=variant,
            quality=quality,
            preference=preference,
            force=force,
            offline=offline,
        )
        path = resolved.artifact_for_role(role, quality=quality)
        if validator is not None:
            validator(path)
        return path
    except (ModelRegistryError, OSError, ArtifactValidationError) as exc:
        raise ArtifactValidationError(str(exc)) from exc


def _download_legacy_github_model(
    variant: ModelVariant,
    quality: ModelQuality,
    force: bool,
    offline: bool,
    tag: str | None,
) -> Path:
    release = resolve_model_release(variant, tag=tag, quality=quality, offline=offline)
    asset = release.model_asset(quality)
    return _download_release_asset(
        release, asset, quality=quality, force=force, validator=_validate_onnx_file, offline=offline
    )


def _download_legacy_github_voices(
    variant: ModelVariant,
    force: bool,
    offline: bool,
    tag: str | None,
) -> Path:
    release = resolve_model_release(
        variant, tag=tag, quality=DEFAULT_MODEL_QUALITY, offline=offline
    )
    asset = release.voice_asset()
    validator = (
        (lambda path: _validate_voice_archive(path, expected_voice_names=release.voices))
        if asset.format == "numpy-npz"
        else _validate_voice_bin
    )
    return _download_release_asset(
        release,
        asset,
        quality=DEFAULT_MODEL_QUALITY,
        force=force,
        validator=validator,
        offline=offline,
    )


def download_model_github(
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
    quality: ModelQuality = DEFAULT_MODEL_QUALITY,
    force: bool = False,
    offline: bool = False,
    *,
    tag: str | None = None,
) -> Path:
    if resolve_model_release is not _DEFAULT_RESOLVE_MODEL_RELEASE:
        return _download_legacy_github_model(variant, quality, force, offline, tag)
    del tag
    return _download_registry_artifact(
        variant,
        "model",
        quality=quality,
        force=force,
        offline=offline,
        validator=_validate_onnx_file,
    )


def download_voices_github(
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
    force: bool = False,
    offline: bool = False,
    *,
    tag: str | None = None,
) -> Path:
    if resolve_model_release is not _DEFAULT_RESOLVE_MODEL_RELEASE:
        return _download_legacy_github_voices(variant, force, offline, tag)
    del tag
    try:
        resolved = resolve_runtime_assets(
            model_id=variant,
            preference="github",
            force=force,
            offline=offline,
        )
        role = "voices" if resolved.artifacts_for_role("voices") else "voice"
        path = resolved.artifact_for_role(role)
        if role == "voices":
            _validate_voice_archive(path, expected_voice_names=resolved.model.voices)
        else:
            _validate_voice_bin(path)
        return path
    except (ModelRegistryError, OSError, ArtifactValidationError) as exc:
        raise ArtifactValidationError(str(exc)) from exc


def download_all_models_github(
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
    quality: ModelQuality = DEFAULT_MODEL_QUALITY,
    progress_callback: Callable[[str, int, int], None] | None = None,
    force: bool = False,
    offline: bool = False,
    *,
    tag: str | None = None,
) -> dict[str, Path]:
    del tag
    try:
        resolved = resolve_runtime_assets(
            model_id=variant,
            quality=quality,
            preference="github",
            force=force,
            offline=offline,
        )
        model_path = resolved.artifact_for_role("model", quality=quality)
        voice_role = "voices" if resolved.artifacts_for_role("voices") else "voice"
        voices_path = resolved.artifact_for_role(voice_role)
        if progress_callback:
            progress_callback("model", 0, 2)
        if progress_callback:
            progress_callback("voices", 1, 2)
        if progress_callback:
            progress_callback("complete", 2, 2)
        return {model_path.name: model_path, voices_path.name: voices_path}
    except (ModelRegistryError, OSError, ArtifactValidationError) as exc:
        raise ArtifactValidationError(str(exc)) from exc


_DEFAULT_DOWNLOAD_MODEL_GITHUB = download_model_github
_DEFAULT_DOWNLOAD_VOICES_GITHUB = download_voices_github
_DEFAULT_DOWNLOAD_MODEL = download_model


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
            short_sentence_config: Configuration for short sentence handling.
                This improves audio quality for short sentences (like "Why?" or
                "Go!") by adding context.
                If None, uses default thresholds (min_phoneme_length=30).
                Set enabled=False to disable.
                Example:
                    from pykokoro.short_sentence_handler import ShortSentenceConfig
                    config = ShortSentenceConfig(
                        min_phoneme_length=20,  # Treat < 20 phonemes as short
                        resolve_mode="randomized-phrase",
                        enabled=True,
                        phoneme_pretext="—"
                    )
                    tts = Kokoro(short_sentence_config=config)
        """
        self._session: rt.InferenceSession | None = None
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
        # GitHub qualities are discovered from the selected release manifest.
        if model_source == "huggingface" and resolved_quality not in MODEL_QUALITY_FILES_HF:
            available = ", ".join(MODEL_QUALITY_FILES_HF.keys())
            raise ValueError(
                f"Quality '{resolved_quality}' not available for HuggingFace {model_variant}. "
                f"Available qualities: {available}"
            )
        self._model_quality: ModelQuality = resolved_quality

        # Registry assets are resolved lazily as one atomic distribution.
        self._resolved_runtime_assets: ResolvedRuntimeAssets | None = None
        self._model_path = model_path
        self._voices_path = voices_path

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

    def _get_vocabulary(self) -> dict[str, int]:
        """Get vocabulary for the current model variant.

        Returns:
            Dictionary mapping phoneme characters to token indices
        """
        from kokorog2p import get_kokoro_vocab

        profile = get_model_profile(self._model_variant, self._model_source)
        vocabulary_path = getattr(self, "_model_config_path", None)
        if profile.vocabulary_source == "downloaded-release" and vocabulary_path is None:
            vocabulary_path = download_vocabulary_github(self._model_variant)
        if profile.vocabulary_source in {"downloaded-config", "downloaded-release"}:
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
        """Ensure all runtime assets come from one selected registry distribution."""
        if not self._model_path_provided and not self._voices_path_provided:
            if self._model_source == "github" and (
                download_model_github is not _DEFAULT_DOWNLOAD_MODEL_GITHUB
                or download_voices_github is not _DEFAULT_DOWNLOAD_VOICES_GITHUB
            ):
                self._model_path = download_model_github(
                    variant=self._model_variant, quality=self._model_quality
                )
                self._voices_path = download_voices_github(variant=self._model_variant)
                return
            if (
                self._model_source == "huggingface"
                and download_model is not _DEFAULT_DOWNLOAD_MODEL
            ):
                self._model_path = download_model(
                    variant=self._model_variant, quality=self._model_quality
                )
                download_all_voices()
                self._voices_path = get_voices_archive_path("huggingface", self._model_variant)
                return
        if not self._model_path_provided or not self._voices_path_provided:
            preference: Literal["github", "huggingface"] = (
                "github" if self._model_source == "github" else "huggingface"
            )
            try:
                self._resolved_runtime_assets = resolve_runtime_assets(
                    model_id=self._model_variant,
                    quality=self._model_quality,
                    preference=preference,
                )
            except (ModelRegistryError, OSError, ArtifactValidationError) as exc:
                raise ConfigurationError(
                    f"Unable to resolve runtime assets for {self._model_variant!r}: {exc}"
                ) from exc
            resolved = self._resolved_runtime_assets
            if self._model_path is None:
                self._model_path = resolved.artifact_for_role("model", quality=self._model_quality)
            if self._voices_path is None:
                voice_role = "voices" if resolved.artifacts_for_role("voices") else "voice"
                voice_paths = resolved.artifacts_for_role(voice_role)
                if len(voice_paths) != 1:
                    self._voices_path = resolved.materialize_raw_voices()
                else:
                    self._voices_path = next(iter(voice_paths.values()))
            if self._model_config_path is None:
                for role in ("vocab", "config"):
                    if resolved.artifacts_for_role(role):
                        self._model_config_path = resolved.artifact_for_role(role)
                        break

        if self._model_path is None or not _is_nonempty_file(self._model_path):
            label = "Explicit model_path" if self._model_path_provided else "Model path"
            raise ConfigurationError(
                f"{label} does not point to a non-empty file: {self._model_path}"
            )
        if self._model_path_provided:
            try:
                _validate_onnx_file(self._model_path)
            except (OSError, ArtifactValidationError) as exc:
                raise ConfigurationError(
                    f"Explicit model_path is not a valid ONNX model: {exc}"
                ) from exc

        if self._voices_path is None or not _is_nonempty_file(self._voices_path):
            raise ConfigurationError(
                f"Voices path does not point to a non-empty file: {self._voices_path}"
            )
        if self._voices_path_provided:
            try:
                _validate_voice_archive(self._voices_path)
            except (OSError, ArtifactValidationError) as exc:
                raise ConfigurationError(
                    f"Explicit voices_path is not a valid voice archive: {exc}"
                ) from exc

        if (
            self._model_config_path_provided
            and self._model_config_path is not None
            and not _is_nonempty_file(self._model_config_path)
        ):
            raise ConfigurationError(
                f"Explicit model_config_path does not point to a non-empty file: "
                f"{self._model_config_path}"
            )

    def _redownload_voices(self, force: bool = False) -> None:
        if self._model_source == "github":
            self._voices_path = download_voices_github(variant=self._model_variant, force=force)
            return

        self._voices_path = _download_hf_voice_archive(
            self._model_variant,
            force=force,
        )

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
        if self._session is not None or self._runtime is not None:
            return

        self._ensure_models()
        assert self._model_path is not None
        assert self._voices_path is not None
        if self._resolved_runtime_assets is not None:
            self._runtime = create_runtime(self._resolved_runtime_assets)
            if self._runtime is not None:
                return

        # Use OnnxSessionManager to create session
        session_manager = OnnxSessionManager(
            provider=self._provider,
            use_gpu=self._use_gpu,
            session_options=self._session_options,
            provider_options=self._provider_options,
            model_quality=self._model_quality,
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
            assert self._voices_path is not None
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
        self._init_kokoro()
        if self._runtime is not None:
            return sorted(self._runtime.voices)
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
        if self._runtime is not None:
            try:
                return np.asarray(self._runtime.voices[voice_name], dtype=np.float32)[:, None, :]
            except KeyError as exc:
                raise KeyError(f"Voice {voice_name!r} not found") from exc
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
        if self._runtime is not None:
            if not isinstance(voice, str):
                raise ConfigurationError(
                    "Voice blending/arrays are not supported by this runtime layout"
                )
            return self.get_voice_style(voice)
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
        segments: list["PhonemeSegment"],
        enable_short_sentence_override: bool | None,
        random_seed: int | None = None,
    ) -> list["PhonemeSegment"]:
        """Preprocess phoneme segments for short sentence handling."""
        self._init_kokoro()
        if self._runtime is not None:
            return segments
        assert self._audio_generator is not None
        return self._audio_generator._preprocess_segments(
            segments, enable_short_sentence_override, random_seed
        )

    def generate_raw_audio_segments(
        self,
        segments: list["PhonemeSegment"],
        voice_style: np.ndarray,
        speed: float,
        voice_resolver: Callable[[str], np.ndarray] | None,
        *,
        default_voice_name: str | None = None,
        trace: "Trace | None" = None,
    ) -> list["PhonemeSegment"]:
        """Generate raw audio for each phoneme segment."""
        self._init_kokoro()
        if self._runtime is not None:
            default_voice = default_voice_name or next(iter(self._runtime.voices))
            for segment in segments:
                voice_name = segment.voice_name or default_voice
                segment.raw_audio = self._runtime.synthesize(segment.text, voice_name, speed=speed)
            return segments
        assert self._audio_generator is not None
        return self._audio_generator._generate_raw_audio_segments(
            segments, voice_style, speed, voice_resolver, trace
        )

    def postprocess_audio_segments(
        self,
        segments: list["PhonemeSegment"],
        trim_silence: bool,
        prosody_config: "ProsodyConfig | None" = None,
        trace: "Trace | None" = None,
    ) -> list["PhonemeSegment"]:
        """Trim/prosody-process raw audio segments."""
        self._init_kokoro()
        if self._runtime is not None:
            return segments
        assert self._audio_generator is not None
        return self._audio_generator._postprocess_audio_segments(
            segments,
            trim_silence,
            prosody_config,
            trace,
        )

    def concatenate_audio_segments(
        self,
        segments: list["PhonemeSegment"],
        prosody_config: "ProsodyConfig | None" = None,
        trace: "Trace | None" = None,
    ) -> np.ndarray:
        """Concatenate processed segments into a single waveform."""
        self._init_kokoro()
        if self._runtime is not None:
            pieces: list[np.ndarray] = []
            for segment in segments:
                if segment.pause_before > 0:
                    pieces.append(
                        np.zeros(round(SAMPLE_RATE * segment.pause_before), dtype=np.float32)
                    )
                audio = (
                    segment.processed_audio
                    if segment.processed_audio is not None
                    else segment.raw_audio
                )
                if audio is not None:
                    pieces.append(np.asarray(audio, dtype=np.float32).reshape(-1))
                if segment.pause_after > 0:
                    pieces.append(
                        np.zeros(round(SAMPLE_RATE * segment.pause_after), dtype=np.float32)
                    )
            return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float32)
        assert self._audio_generator is not None
        return self._audio_generator._concatenate_audio_segments(
            segments,
            prosody_config,
            trace,
        )

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
        self._voice_db = sqlite3.connect(str(db_path), detect_types=sqlite3.PARSE_DECLTYPES)

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
        random_seed: int | None = None,
        prosody_config: "ProsodyConfig | None" = None,
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
            random_seed=random_seed,
            prosody_config=prosody_config,
        )

    def close(self) -> None:
        """Release database, tokenizer, voice, generator, and ONNX resources."""
        voice_db, self._voice_db = getattr(self, "_voice_db", None), None

        self._audio_generator = None
        self._tokenizer = None
        self._voice_manager = None
        self._session = None

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
