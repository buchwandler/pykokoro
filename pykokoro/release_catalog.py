"""Discovery and validation of published Kokoro model releases."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .utils import get_user_cache_path

MODEL_REPOSITORY = "buchwandler/kokoro-onnx-models"
GITHUB_API = "https://api.github.com"
MANIFEST_NAME = "release-manifest.json"
SUPPORTED_MANIFEST_SCHEMAS = frozenset({2})
SUPPORTED_RUNTIME_CONTRACTS = frozenset({1})
SUPPORTED_FRONTENDS = frozenset(
    {
        "pykokoro-native-v1",
        "pykokoro-native",
        "vig2p",
        "vig2p-v1",
        "german-ipa-v1",
        "German IPA",
        "Arabic diacritizer + espeak-ng + Nabra cleanup",
        "nabra-arabic-v1",
        "Hebrew-specific G2P",
    }
)
SUPPORTED_VOICE_FORMATS = frozenset({"numpy-npz", "raw-float32-le"})


class ReleaseCatalogError(RuntimeError):
    """Raised when a published release cannot be used."""


class IncompatibleReleaseError(ReleaseCatalogError):
    """Raised when an exact release exists but is not compatible."""

    def __init__(self, tag: str, reason: str) -> None:
        self.tag = tag
        self.reason = reason
        super().__init__(f"Release {tag!r} is incompatible: {reason}")


@dataclass(frozen=True, slots=True)
class ReleaseAsset:
    name: str
    role: str
    format: str
    size: int
    sha256: str
    download_url: str
    quality: str | None = None


@dataclass(frozen=True, slots=True)
class RemoteModelRelease:
    profile: str
    model_version: str
    release_tag: str
    release_published_at: str
    manifest_schema: int
    runtime_contract: int
    language_codes: tuple[str, ...]
    frontend: str
    sample_rate: int
    default_voice: str
    voices: tuple[str, ...]
    assets: tuple[ReleaseAsset, ...]
    onnx_contract: Mapping[str, object]
    publication_enabled: bool
    manifest: Mapping[str, object]
    compatible: bool = True
    incompatibility_reason: str | None = None

    def asset(
        self, role: str, *, quality: str | None = None, format: str | None = None
    ) -> ReleaseAsset:
        matches = [
            asset
            for asset in self.assets
            if asset.role == role
            and (quality is None or asset.quality == quality)
            and (format is None or asset.format == format)
        ]
        if not matches:
            detail = ", ".join(f"{item.role}/{item.quality or item.format}" for item in self.assets)
            raise ReleaseCatalogError(
                f"Release {self.release_tag!r} has no {role} asset ({detail})"
            )
        return matches[0]

    def model_asset(self, quality: str = "fp32") -> ReleaseAsset:
        return self.asset("model", quality=quality)

    def voice_asset(
        self, preferred_formats: tuple[str, ...] = ("numpy-npz", "raw-float32-le")
    ) -> ReleaseAsset:
        for asset_format in preferred_formats:
            try:
                return self.asset("voices", format=asset_format)
            except ReleaseCatalogError:
                continue
        return self.asset("voices")

    def assets_for_role(self, role: str) -> tuple[ReleaseAsset, ...]:
        return tuple(asset for asset in self.assets if asset.role == role)


@dataclass(frozen=True, slots=True)
class InstalledModelRelease:
    """Paths and provenance for a verified local model release."""

    release: RemoteModelRelease
    quality: str
    model_path: Path
    voices_path: Path
    auxiliary_paths: tuple[Path, ...] = ()


class GitHubReleaseClient:
    """Small urllib-based GitHub API client, easy to replace in tests."""

    def __init__(self, *, token: str | None = None) -> None:
        self.token = token or os.environ.get("PYKOKORO_GITHUB_TOKEN")

    def _request(self, url: str) -> bytes:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "pykokoro-release-catalog/1",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            raise ReleaseCatalogError(f"GitHub request failed with HTTP {exc.code}: {url}") from exc
        except urllib.error.URLError as exc:
            raise ReleaseCatalogError(f"GitHub request failed: {url}: {exc.reason}") from exc

    def json(self, url: str) -> Any:
        try:
            return json.loads(self._request(url))
        except json.JSONDecodeError as exc:
            raise ReleaseCatalogError(f"GitHub returned invalid JSON: {url}") from exc

    def bytes(self, url: str) -> bytes:
        return self._request(url)

    def list_releases(self) -> list[dict[str, Any]]:
        data = self.json(f"{GITHUB_API}/repos/{MODEL_REPOSITORY}/releases?per_page=100")
        if not isinstance(data, list):
            raise ReleaseCatalogError("GitHub releases response is not a list")
        return [item for item in data if isinstance(item, dict)]

    def get_release(self, tag: str) -> dict[str, Any]:
        data = self.json(f"{GITHUB_API}/repos/{MODEL_REPOSITORY}/releases/tags/{tag}")
        if not isinstance(data, dict):
            raise ReleaseCatalogError(f"GitHub release response for {tag!r} is not an object")
        return data


_DEFAULT_CLIENT = GitHubReleaseClient()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest_asset(release: Mapping[str, Any]) -> Mapping[str, Any]:
    for asset in release.get("assets", []):
        if isinstance(asset, Mapping) and asset.get("name") == MANIFEST_NAME:
            return asset
    raise ReleaseCatalogError(
        f"Release {release.get('tag_name', '<unknown>')!r} has no {MANIFEST_NAME}"
    )


def _verify_manifest_digest(data: bytes, asset: Mapping[str, Any]) -> None:
    digest = asset.get("digest") or asset.get("sha256")
    if isinstance(digest, str) and digest.startswith("sha256:"):
        digest = digest[7:]
    if digest is not None and _sha256(data) != str(digest).lower():
        raise ReleaseCatalogError(
            f"Manifest digest mismatch for {asset.get('name', MANIFEST_NAME)}"
        )


def _parse_asset(
    item: Mapping[str, Any], github_assets: Mapping[str, Mapping[str, Any]]
) -> ReleaseAsset:
    name = item.get("name")
    _require(isinstance(name, str) and name, "Manifest asset name is invalid")
    github_asset = github_assets.get(name)
    _require(github_asset is not None, f"Manifest asset {name!r} is not present in GitHub Release")
    role = item.get("role")
    asset_format = item.get("format")
    _require(isinstance(role, str) and role, f"Manifest asset {name!r} has no role")
    _require(
        isinstance(asset_format, str) and asset_format, f"Manifest asset {name!r} has no format"
    )
    size = item.get("size")
    sha256 = item.get("sha256")
    _require(isinstance(size, int) and size > 0, f"Manifest asset {name!r} has invalid size")
    _require(
        isinstance(sha256, str)
        and len(sha256) == 64
        and all(c in "0123456789abcdef" for c in sha256),
        f"Manifest asset {name!r} has invalid SHA-256",
    )
    url = github_asset.get("browser_download_url")
    _require(isinstance(url, str) and url, f"GitHub asset {name!r} has no browser download URL")
    quality = item.get("quality")
    return ReleaseAsset(
        name, role, asset_format, size, sha256, url, quality if isinstance(quality, str) else None
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseCatalogError(message)


def _parse_release(
    release: Mapping[str, Any],
    manifest_data: bytes,
    manifest_asset: Mapping[str, Any],
    *,
    allow_prerelease: bool,
    local_base: Path | None = None,
) -> RemoteModelRelease:
    _verify_manifest_digest(manifest_data, manifest_asset)
    try:
        manifest = json.loads(manifest_data)
    except json.JSONDecodeError as exc:
        raise ReleaseCatalogError("Release manifest is not valid JSON") from exc
    _require(isinstance(manifest, dict), "Release manifest must be an object")
    _require(manifest.get("schema") in SUPPORTED_MANIFEST_SCHEMAS, "Unsupported manifest schema")
    _require(
        manifest.get("runtime_contract") in SUPPORTED_RUNTIME_CONTRACTS,
        "Unsupported runtime contract",
    )
    tag = release.get("tag_name") or manifest.get("tag")
    _require(isinstance(tag, str) and tag, "GitHub release has no tag")
    _require(
        manifest.get("repository") == MODEL_REPOSITORY,
        "Manifest repository does not match release repository",
    )
    _require(
        manifest.get("tag") == tag,
        f"Manifest tag {manifest.get('tag')!r} does not match GitHub tag {tag!r}",
    )
    if release.get("draft"):
        raise ReleaseCatalogError(f"Release {tag!r} is a draft")
    if release.get("prerelease") and not allow_prerelease:
        raise ReleaseCatalogError(f"Release {tag!r} is a prerelease")

    runtime = manifest.get("runtime")
    _require(isinstance(runtime, dict), "Manifest runtime metadata is invalid")
    language_codes = runtime.get("language_codes")
    voices = runtime.get("voices")
    frontend = runtime.get("frontend")
    _require(
        isinstance(language_codes, list)
        and all(isinstance(value, str) for value in language_codes),
        "Manifest language codes are invalid",
    )
    _require(
        isinstance(voices, list) and voices and all(isinstance(value, str) for value in voices),
        "Manifest voices are invalid",
    )
    _require(isinstance(frontend, str) and frontend, "Manifest frontend is invalid")
    assets_data = manifest.get("assets")
    _require(isinstance(assets_data, list) and assets_data, "Manifest assets are invalid")
    github_assets = {
        item["name"]: item
        for item in release.get("assets", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    seen_names: set[str] = set()
    assets: list[ReleaseAsset] = []
    for item in assets_data:
        _require(isinstance(item, dict), "Manifest asset is not an object")
        asset = _parse_asset(item, github_assets)
        _require(asset.name not in seen_names, f"Duplicate manifest asset {asset.name!r}")
        seen_names.add(asset.name)
        assets.append(
            ReleaseAsset(
                asset.name,
                asset.role,
                asset.format,
                asset.size,
                asset.sha256,
                (local_base / asset.name).as_uri() if local_base else asset.download_url,
                asset.quality,
            )
        )
    _require(any(asset.role == "model" for asset in assets), "Manifest has no model asset")
    voice_assets = [asset for asset in assets if asset.role == "voices"]
    _require(voice_assets, "Manifest has no voices asset")
    _require(
        any(asset.format in SUPPORTED_VOICE_FORMATS for asset in voice_assets),
        "Manifest has no supported voice format",
    )
    return RemoteModelRelease(
        profile=str(manifest["profile"]),
        model_version=str(manifest["model_version"]),
        release_tag=tag,
        release_published_at=str(
            release.get("published_at") or release.get("created_at") or manifest["generated_at"]
        ),
        manifest_schema=int(manifest["schema"]),
        runtime_contract=int(manifest["runtime_contract"]),
        language_codes=tuple(language_codes),
        frontend=frontend,
        sample_rate=int(runtime["sample_rate"]),
        default_voice=str(runtime["default_voice"]),
        voices=tuple(voices),
        assets=tuple(assets),
        onnx_contract=manifest.get("onnx_contract", {}),
        publication_enabled=bool(manifest.get("publication", {}).get("enabled")),
        manifest=manifest,
    )


def _compatibility_reason(
    release: RemoteModelRelease, *, profile: str | None, language: str | None, quality: str | None
) -> str | None:
    if profile is not None and release.profile != profile:
        return f"profile is {release.profile!r}, requested {profile!r}"
    if language is not None and language.lower() not in {
        value.lower() for value in release.language_codes
    }:
        return f"language {language!r} is not declared"
    if not release.publication_enabled:
        return "publication is disabled"
    if release.frontend not in SUPPORTED_FRONTENDS:
        return f"frontend {release.frontend!r} is not implemented"
    if quality is not None:
        try:
            release.model_asset(quality)
        except ReleaseCatalogError:
            return f"model quality {quality!r} is not published"
    return None


def _with_compatibility(
    release: RemoteModelRelease, *, profile: str | None, language: str | None, quality: str | None
) -> RemoteModelRelease:
    reason = _compatibility_reason(release, profile=profile, language=language, quality=quality)
    if reason is None:
        return release
    return replace(release, compatible=False, incompatibility_reason=reason)


def _remote_release(
    release: Mapping[str, Any], client: GitHubReleaseClient, *, allow_prerelease: bool
) -> RemoteModelRelease:
    manifest_asset = _manifest_asset(release)
    url = manifest_asset.get("browser_download_url")
    _require(isinstance(url, str) and url, "Manifest asset has no download URL")
    data = client.bytes(url)
    return _parse_release(release, data, manifest_asset, allow_prerelease=allow_prerelease)


def _installed_root() -> Path:
    return get_user_cache_path("releases")


def _installed_releases() -> list[RemoteModelRelease]:
    root = _installed_root()
    if not root.exists():
        return []
    result = []
    for manifest_path in root.glob("*/*/" + MANIFEST_NAME):
        base = manifest_path.parent
        try:
            manifest_data = manifest_path.read_bytes()
            manifest = json.loads(manifest_data)
            assets = {
                item["name"]: {
                    "name": item["name"],
                    "browser_download_url": (base / item["name"]).as_uri(),
                }
                for item in manifest["assets"]
            }
            release = {
                "tag_name": manifest["tag"],
                "published_at": manifest.get("generated_at"),
                "assets": list(assets.values()),
            }
            parsed = _parse_release(
                release,
                manifest_data,
                {"name": MANIFEST_NAME, "browser_download_url": manifest_path.as_uri()},
                allow_prerelease=True,
                local_base=base,
            )
            if all((base / asset.name).is_file() for asset in parsed.assets):
                if any(
                    not (base / asset.name).is_file()
                    or (base / asset.name).stat().st_size != asset.size
                    or _sha256((base / asset.name).read_bytes()) != asset.sha256
                    for asset in parsed.assets
                ):
                    continue
                result.append(parsed)
        except (OSError, KeyError, TypeError, ValueError, ReleaseCatalogError):
            continue
    return sorted(result, key=lambda item: item.release_published_at, reverse=True)


def available_model_releases(
    *,
    profile: str | None = None,
    language: str | None = None,
    quality: str | None = None,
    include_incompatible: bool = False,
    offline: bool = False,
    client: GitHubReleaseClient | None = None,
) -> tuple[RemoteModelRelease, ...]:
    """Return published compatible releases, newest first."""
    if offline:
        releases = _installed_releases()
    else:
        api = client or _DEFAULT_CLIENT
        releases = []
        for item in api.list_releases():
            if item.get("draft") or item.get("prerelease"):
                continue
            try:
                releases.append(_remote_release(item, api, allow_prerelease=False))
            except ReleaseCatalogError:
                continue
    filtered = [
        _with_compatibility(item, profile=profile, language=language, quality=quality)
        for item in releases
    ]
    if include_incompatible:
        return tuple(filtered)
    return tuple(item for item in filtered if item.compatible)


def resolve_model_release(
    profile: str,
    *,
    tag: str | None = None,
    quality: str = "fp32",
    offline: bool = False,
    client: GitHubReleaseClient | None = None,
) -> RemoteModelRelease:
    """Resolve the newest compatible release, or one exact tag."""
    if offline:
        candidates = _installed_releases()
        if tag is not None:
            candidates = [item for item in candidates if item.release_tag == tag]
    else:
        api = client or _DEFAULT_CLIENT
        if tag is not None:
            try:
                candidates = [_remote_release(api.get_release(tag), api, allow_prerelease=True)]
            except ReleaseCatalogError as exc:
                raise IncompatibleReleaseError(tag, str(exc)) from exc
        else:
            candidates = []
            for item in api.list_releases():
                if item.get("draft") or item.get("prerelease"):
                    continue
                try:
                    candidates.append(_remote_release(item, api, allow_prerelease=False))
                except ReleaseCatalogError:
                    continue
    for release in candidates:
        reason = _compatibility_reason(release, profile=profile, language=None, quality=quality)
        if reason is None:
            return release
        if tag is not None:
            raise IncompatibleReleaseError(tag, reason)
    if tag is not None:
        raise ReleaseCatalogError(f"Published release tag {tag!r} was not found")
    raise ReleaseCatalogError(
        f"No compatible published release found for profile {profile!r} and quality {quality!r}"
    )


def download_model_release(
    profile: str,
    *,
    tag: str | None = None,
    quality: str = "fp32",
    force: bool = False,
    offline: bool = False,
) -> InstalledModelRelease:
    """Download and verify all assets needed by one resolved release."""
    from .model_assets import release_asset_path
    from .onnx_backend import download_all_models_github, download_release_auxiliary

    release = resolve_model_release(profile, tag=tag, quality=quality, offline=offline)
    paths = download_all_models_github(
        profile, quality, force=force, offline=offline, tag=release.release_tag
    )
    model_asset = release.model_asset(quality)
    voice_asset = release.voice_asset()
    auxiliary_paths = []
    for role in ("config", "vocab", "bundle"):
        if release.assets_for_role(role):
            auxiliary_paths.append(
                download_release_auxiliary(
                    profile, role, force=force, offline=offline, tag=release.release_tag
                )
            )
    return InstalledModelRelease(
        release=release,
        quality=quality,
        model_path=paths.get(model_asset.name, release_asset_path(release, model_asset)),
        voices_path=paths.get(voice_asset.name, release_asset_path(release, voice_asset)),
        auxiliary_paths=tuple(auxiliary_paths),
    )
