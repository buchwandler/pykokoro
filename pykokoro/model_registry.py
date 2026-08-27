"""Validated, provider-neutral client for the pykokoro model registry."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .utils import get_user_cache_path

MODEL_REGISTRY_URL = (
    "https://raw.githubusercontent.com/buchwandler/kokoro-onnx-models/main/catalog/models.json"
)
DownloadPreference = Literal["auto", "github", "huggingface", "upstream"]


class ModelRegistryError(RuntimeError):
    """Raised when the model registry or an artifact is unusable."""


@dataclass(frozen=True, slots=True)
class RuntimeArtifact:
    id: str
    role: str
    format: str
    url: str
    local_name: str
    size: int
    sha256: str
    quality: str | None = None
    component: str | None = None
    voice: str | None = None
    handling: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class RuntimeDistribution:
    id: str
    provider: str
    transport: str
    runtime_ready: bool
    artifacts: tuple[RuntimeArtifact, ...]
    repository: str | None = None
    revision: str | None = None
    release_key: str | None = None
    release_tag: str | None = None
    provenance: Mapping[str, Any] | None = None

    def artifact(self, role: str, *, quality: str | None = None) -> RuntimeArtifact:
        matches = [
            item
            for item in self.artifacts
            if item.role == role and (quality is None or item.quality == quality)
        ]
        if not matches:
            available = ", ".join(
                f"{item.role}/{item.quality or item.format}" for item in self.artifacts
            )
            detail = f" ({available})" if available else ""
            quality_detail = f" quality {quality!r}" if quality is not None else ""
            raise ModelRegistryError(
                f"Distribution {self.id!r} has no {role}{quality_detail}{detail}"
            )
        return matches[0]

    @property
    def qualities(self) -> tuple[str, ...]:
        return tuple(
            item.quality
            for item in self.artifacts
            if item.role == "model" and item.quality is not None
        )


@dataclass(frozen=True, slots=True)
class RuntimeModel:
    model_id: str
    data: Mapping[str, Any]
    distributions: tuple[RuntimeDistribution, ...]

    @property
    def runtime(self) -> Mapping[str, Any]:
        return self.data["runtime"]

    @property
    def voices(self) -> tuple[str, ...]:
        return tuple(self.runtime["voices"])

    @property
    def default_voice(self) -> str:
        return str(self.runtime["default_voice"])

    @property
    def runtime_available(self) -> bool:
        return bool(self.data.get("runtime_available", True))

    @property
    def language_codes(self) -> tuple[str, ...]:
        values = self.data.get("language_codes", self.runtime.get("language_codes", []))
        return tuple(str(value) for value in values)

    @property
    def frontend(self) -> str:
        return str(self.data.get("frontend", self.runtime.get("frontend", "")))

    @property
    def layout(self) -> str:
        return str(self.runtime.get("layout", ""))

    @property
    def sample_rate(self) -> int:
        return int(self.data.get("sample_rate", self.runtime.get("sample_rate", 24000)))

    @property
    def max_tokens(self) -> int:
        return int(self.runtime.get("max_tokens", self.data.get("max_tokens", 510)))

    @property
    def onnx_contract(self) -> Mapping[str, Any]:
        contract = self.data.get("onnx_contract", self.runtime.get("onnx", self.data.get("onnx", {})))
        return contract if isinstance(contract, Mapping) else {}

    @property
    def redistribution_allowed(self) -> bool:
        license_data = self.data.get("license")
        if isinstance(license_data, Mapping):
            redistribution = str(license_data.get("redistribution", "allowed"))
            if redistribution.lower().startswith("restrict") or redistribution.lower() == "forbidden":
                return False
        policy = self.data.get("redistribution", self.data.get("redistribution_allowed", True))
        if isinstance(policy, Mapping):
            return bool(policy.get("allowed", True))
        return bool(policy)


    def distribution(self, preference: DownloadPreference = "auto") -> RuntimeDistribution:
        return select_distribution(self.distributions, preference)


@dataclass(frozen=True, slots=True)
class ModelRegistry:
    data: Mapping[str, Any]
    source: str

    @property
    def models(self) -> Mapping[str, RuntimeModel]:
        return {
            model_id: RuntimeModel(
                model_id,
                model,
                tuple(_distribution(item) for item in model["distributions"]),
            )
            for model_id, model in self.data["models"].items()
        }

    def model(self, model_id: str) -> RuntimeModel:
        try:
            return self.models[model_id]
        except KeyError as exc:
            raise ModelRegistryError(f"Unknown model profile: {model_id}") from exc


class RegistryClient:
    """Load, cache, validate, and resolve the central model registry."""

    def __init__(
        self,
        *,
        url: str | None = None,
        path: Path | None = None,
        cache_path: Path | None = None,
    ) -> None:
        self.url = url or os.environ.get("PYKOKORO_MODEL_REGISTRY_URL", MODEL_REGISTRY_URL)
        configured_path = os.environ.get("PYKOKORO_MODEL_REGISTRY_PATH")
        self.path = path or (Path(configured_path) if configured_path else None)
        self.cache_path = cache_path or get_user_cache_path("registry") / "models.json"

    def load(self, *, offline: bool = False) -> ModelRegistry:
        if self.path is not None:
            data = _read_json(self.path)
            _validate_registry(data)
            return ModelRegistry(data, str(self.path))
        if offline:
            data = _read_json(self.cache_path)
            _validate_registry(data)
            return ModelRegistry(data, str(self.cache_path))
        try:
            with urllib.request.urlopen(
                urllib.request.Request(self.url, headers={"User-Agent": "pykokoro-model-registry/1"}),
                timeout=60,
            ) as response:
                data = json.loads(response.read())
            _validate_registry(data)
        except (OSError, urllib.error.URLError, json.JSONDecodeError, ModelRegistryError) as exc:
            if not self.cache_path.is_file():
                raise ModelRegistryError(f"Cannot load model registry: {exc}") from exc
            try:
                data = _read_json(self.cache_path)
                _validate_registry(data)
            except (OSError, json.JSONDecodeError, ModelRegistryError) as cache_exc:
                raise ModelRegistryError(f"Cannot load model registry or valid cache: {cache_exc}") from exc
            return ModelRegistry(data, str(self.cache_path))
        _write_json_atomically(self.cache_path, data)
        return ModelRegistry(data, self.url)

    def select_distribution(
        self, model_id: str, preference: DownloadPreference = "auto", *, offline: bool = False
    ) -> RuntimeDistribution:
        return self.load(offline=offline).model(model_id).distribution(preference)

    def select_artifact(
        self,
        model_id: str,
        role: str,
        *,
        quality: str | None = None,
        preference: DownloadPreference = "auto",
        offline: bool = False,
    ) -> RuntimeArtifact:
        distribution = self.select_distribution(model_id, preference, offline=offline)
        return distribution.artifact(role, quality=quality)


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ModelRegistryError(f"Registry {path} must contain an object")
    return data


def _write_json_atomically(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")
    temporary.replace(path)


def _validate_registry(data: Mapping[str, Any]) -> None:
    if data.get("schema") != 1 or data.get("runtime_contract") != 1:
        raise ModelRegistryError("Unsupported model registry schema or runtime contract")
    models = data.get("models")
    if not isinstance(models, dict) or not models:
        raise ModelRegistryError("Model registry has no models")
    distribution_ids: set[str] = set()
    for model_id, model in models.items():
        if not isinstance(model_id, str) or not isinstance(model, dict):
            raise ModelRegistryError("Model registry contains an invalid model entry")
        runtime = model.get("runtime")
        if not isinstance(runtime, dict) or not isinstance(runtime.get("voices"), list):
            raise ModelRegistryError(f"Model {model_id} has invalid runtime metadata")
        voices = runtime["voices"]
        if not voices or len(voices) != len(set(voices)):
            raise ModelRegistryError(f"Model {model_id} has an invalid voice roster")
        if runtime.get("default_voice") not in voices:
            raise ModelRegistryError(f"Model {model_id} default voice is not advertised")
        distributions = model.get("distributions")
        if not isinstance(distributions, list):
            raise ModelRegistryError(f"Model {model_id} distributions are invalid")
        if not distributions:
            if model.get("runtime_available", True) is False:
                continue
            raise ModelRegistryError(f"Model {model_id} has no runtime distributions")
        for item in distributions:
            distribution = _distribution(item)
            if distribution.id in distribution_ids:
                raise ModelRegistryError(f"Duplicate distribution id: {distribution.id}")
            distribution_ids.add(distribution.id)
            if not distribution.artifacts:
                raise ModelRegistryError(f"Distribution {distribution.id} has no artifacts")
            if not any(asset.role == "model" for asset in distribution.artifacts):
                raise ModelRegistryError(f"Distribution {distribution.id} has no model artifact")
            if not any(asset.role in {"voice", "voices"} for asset in distribution.artifacts):
                raise ModelRegistryError(f"Distribution {distribution.id} has no voice artifact")


def _distribution(data: Mapping[str, Any]) -> RuntimeDistribution:
    if not isinstance(data, Mapping):
        raise ModelRegistryError("Distribution is not an object")
    required = ("id", "provider", "transport", "runtime_ready", "artifacts")
    if any(not data.get(field) for field in required):
        raise ModelRegistryError("Distribution is missing required fields")
    artifacts_data = data["artifacts"]
    if not isinstance(artifacts_data, list):
        raise ModelRegistryError(f"Distribution {data.get('id')} artifacts are invalid")
    artifacts = tuple(_artifact(item) for item in artifacts_data)
    return RuntimeDistribution(
        id=str(data["id"]),
        provider=str(data["provider"]),
        transport=str(data["transport"]),
        runtime_ready=bool(data["runtime_ready"]),
        artifacts=artifacts,
        repository=str(data["repository"]) if data.get("repository") else None,
        revision=str(data["revision"]) if data.get("revision") else None,
        release_key=str(data["release_key"]) if data.get("release_key") else None,
        release_tag=str(data["release_tag"]) if data.get("release_tag") else None,
        provenance=data.get("provenance"),
    )


def _artifact(data: Mapping[str, Any]) -> RuntimeArtifact:
    if not isinstance(data, Mapping):
        raise ModelRegistryError("Artifact is not an object")
    required = ("id", "role", "format", "url", "local_name", "size", "sha256")
    if any(field not in data for field in required):
        raise ModelRegistryError("Artifact is missing required fields")
    url = str(data["url"])
    if not url.startswith("https://"):
        raise ModelRegistryError(f"Artifact URL is not HTTPS: {url}")
    size = data["size"]
    digest = str(data["sha256"])
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ModelRegistryError(f"Artifact {data['id']} has invalid size")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ModelRegistryError(f"Artifact {data['id']} has invalid SHA-256")
    return RuntimeArtifact(
        id=str(data["id"]),
        role=str(data["role"]),
        format=str(data["format"]),
        url=url,
        local_name=str(data["local_name"]),
        size=size,
        sha256=digest,
        quality=str(data["quality"]) if data.get("quality") is not None else None,
        component=str(data["component"]) if data.get("component") is not None else None,
        voice=str(data["voice"]) if data.get("voice") is not None else None,
        handling=data.get("handling"),
    )


def select_distribution(
    distributions: tuple[RuntimeDistribution, ...], preference: DownloadPreference = "auto"
) -> RuntimeDistribution:
    if preference not in {"auto", "github", "huggingface", "upstream"}:
        raise ModelRegistryError(f"Unknown download preference: {preference}")
    providers = {
        "github": ("github-release",),
        "huggingface": ("huggingface",),
        "upstream": ("huggingface",),
        "auto": ("github-release", "huggingface"),
    }[preference]
    for provider in providers:
        for distribution in distributions:
            if distribution.runtime_ready and distribution.provider == provider:
                return distribution
    raise ModelRegistryError(f"No runtime distribution matches download preference {preference!r}")


def verify_artifact(path: Path, artifact: RuntimeArtifact) -> None:
    if not path.is_file() or path.stat().st_size != artifact.size:
        raise ModelRegistryError(f"Size mismatch for artifact {artifact.id}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != artifact.sha256:
        raise ModelRegistryError(f"SHA-256 mismatch for artifact {artifact.id}")


def download_artifact(artifact: RuntimeArtifact, target: Path) -> Path:
    """Download one artifact and replace the target only after verification."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=target.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        request = urllib.request.Request(artifact.url, headers={"User-Agent": "pykokoro/registry"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        verify_artifact(temporary, artifact)
        temporary.replace(target)
    except (OSError, urllib.error.URLError, ModelRegistryError):
        temporary.unlink(missing_ok=True)
        raise
    return target


ModelRegistryClient = RegistryClient


def load_registry(
    *,
    url: str | None = None,
    path: Path | None = None,
    cache_path: Path | None = None,
    offline: bool = False,
) -> ModelRegistry:
    return RegistryClient(url=url, path=path, cache_path=cache_path).load(offline=offline)
