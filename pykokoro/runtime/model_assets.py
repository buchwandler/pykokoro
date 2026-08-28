"""Atomic runtime asset resolution from the canonical model registry."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..model_registry import (
    DownloadPreference,
    ModelRegistry,
    ModelRegistryError,
    RegistryClient,
    RuntimeArtifact,
    RuntimeDistribution,
    RuntimeModel,
    download_artifact,
    verify_artifact,
)
from ..utils import get_user_cache_path


@dataclass(frozen=True, slots=True)
class ResolvedRuntimeAssets:
    """All files selected for one model and one registry distribution."""

    model_id: str
    distribution_id: str
    provider: str
    layout: str
    artifacts: Mapping[str, Path]
    model: RuntimeModel
    distribution: RuntimeDistribution

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.model.data

    def artifact(self, artifact_id: str) -> Path:
        try:
            return self.artifacts[artifact_id]
        except KeyError as exc:
            raise ModelRegistryError(
                f"Distribution {self.distribution_id!r} has no materialized artifact {artifact_id!r}"
            ) from exc

    def artifacts_for_role(self, role: str) -> Mapping[str, Path]:
        ids = {artifact.id for artifact in self.distribution.artifacts if artifact.role == role}
        return {
            artifact_id: path for artifact_id, path in self.artifacts.items() if artifact_id in ids
        }

    def artifact_for_role(
        self, role: str, *, quality: str | None = None, component: str | None = None
    ) -> Path:
        candidates = [
            artifact
            for artifact in self.distribution.artifacts
            if artifact.role == role
            and (quality is None or artifact.quality == quality)
            and (component is None or artifact.component == component)
        ]
        if not candidates:
            raise ModelRegistryError(
                f"Distribution {self.distribution_id!r} has no matching {role} artifact"
            )
        return self.artifacts[candidates[0].id]

    def materialize_raw_voices(self, *, force: bool = False) -> Path:
        """Convert validated individual raw voice artifacts into a cache-only NPZ."""
        voice_artifacts = [
            artifact for artifact in self.distribution.artifacts if artifact.role == "voice"
        ]
        if not voice_artifacts:
            return self.artifact_for_role("voices")
        parent = next(iter(self.artifacts.values())).parent
        target = parent / "voices-materialized.npz"
        provenance = target.with_suffix(".json")
        identity = {
            "model_id": self.model_id,
            "distribution_id": self.distribution_id,
            "provider": self.provider,
            "revision": self.distribution.revision,
            "artifacts": [
                {"id": artifact.id, "sha256": artifact.sha256} for artifact in voice_artifacts
            ],
        }
        if target.is_file() and provenance.is_file() and not force:
            try:
                if json.loads(provenance.read_text(encoding="utf-8")) == identity:
                    return target
            except (OSError, json.JSONDecodeError):
                pass
        voices: dict[str, np.ndarray] = {}
        for artifact in voice_artifacts:
            handling = artifact.handling or {}
            shape = handling.get("shape")
            if handling.get("dtype") != "float32" or not isinstance(shape, list) or len(shape) != 2:
                raise ModelRegistryError(
                    f"Raw voice artifact {artifact.id!r} has unsupported handling metadata"
                )
            endianness = handling.get("endianness", "little")
            if endianness not in {"little", "big"}:
                raise ModelRegistryError(
                    f"Raw voice artifact {artifact.id!r} has invalid endianness"
                )
            dtype = np.dtype((">" if endianness == "big" else "<") + "f4")
            values = np.fromfile(self.artifacts[artifact.id], dtype=dtype)
            expected = int(shape[0]) * int(shape[1])
            if values.size != expected:
                raise ModelRegistryError(
                    f"Raw voice artifact {artifact.id!r} has {values.size} values; expected {expected}"
                )
            name = artifact.voice or Path(artifact.local_name).stem
            voices[name] = values.reshape(int(shape[0]), 1, int(shape[1])).astype(np.float32)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".npz", dir=target.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            np.savez(temporary_path, **voices)  # type: ignore[arg-type]
            temporary_path.replace(target)
            provenance.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        return target


def _cache_path(
    artifact: RuntimeArtifact,
    *,
    model_id: str,
    distribution_id: str,
    cache_dir: Path | None,
) -> Path:
    root = cache_dir or get_user_cache_path("registry")
    return root / model_id / distribution_id / artifact.local_name


def _materialize_artifact(
    artifact: RuntimeArtifact,
    target: Path,
    *,
    force: bool,
    offline: bool,
) -> Path:
    if target.is_file() and not force:
        try:
            verify_artifact(target, artifact)
            return target
        except ModelRegistryError:
            target.unlink(missing_ok=True)

    if offline:
        raise ModelRegistryError(
            f"Offline mode is enabled and {artifact.local_name!r} is not cached for "
            f"distribution {target.parent.name!r}"
        )

    return download_artifact(artifact, target)


def resolve_runtime_assets(
    *,
    model_id: str,
    quality: str | None = None,
    preference: DownloadPreference = "auto",
    offline: bool = False,
    force: bool = False,
    registry: ModelRegistry | None = None,
    registry_client: RegistryClient | None = None,
    cache_dir: Path | None = None,
) -> ResolvedRuntimeAssets:
    """Select and materialize every runtime artifact from one distribution.

    ``registry`` and ``registry_client`` are injectable for deterministic callers and
    tests. A normal caller only needs to provide ``model_id``.
    """
    if registry is None:
        client = registry_client or RegistryClient()
        registry = client.load(offline=offline)

    model = registry.model(model_id)
    if model.data.get("runtime_available", True) is False:
        raise ModelRegistryError(
            f"Model profile {model_id!r} is present in the registry but has no runtime-ready distribution"
        )
    if not model.redistribution_allowed:
        raise ModelRegistryError(
            f"Model profile {model_id!r} is restricted by its redistribution policy"
        )

    distribution = model.distribution(preference)
    artifacts = list(distribution.artifacts)
    if quality is not None:
        model_artifacts = [
            artifact
            for artifact in artifacts
            if artifact.role == "model" and artifact.quality == quality
        ]
        if not model_artifacts:
            raise ModelRegistryError(
                f"Distribution {distribution.id!r} has no model artifact for quality {quality!r}"
            )

    materialized: dict[str, Path] = {}
    for artifact in artifacts:
        if artifact.id in materialized:
            raise ModelRegistryError(f"Duplicate artifact id in distribution {distribution.id!r}")
        target = _cache_path(
            artifact,
            model_id=model_id,
            distribution_id=distribution.id,
            cache_dir=cache_dir,
        )
        materialized[artifact.id] = _materialize_artifact(
            artifact,
            target,
            force=force,
            offline=offline,
        )

    runtime = model.runtime
    layout = runtime.get("layout")
    if not isinstance(layout, str) or not layout:
        raise ModelRegistryError(f"Model {model_id!r} has no supported runtime layout metadata")
    return ResolvedRuntimeAssets(
        model_id=model_id,
        distribution_id=distribution.id,
        provider=distribution.provider,
        layout=layout,
        artifacts=materialized,
        model=model,
        distribution=distribution,
    )
