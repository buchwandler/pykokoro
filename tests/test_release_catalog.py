from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pykokoro import release_catalog


def _manifest(
    tag: str = "model-files-test", *, frontend: str = "pykokoro-native-v1", quality: str = "fp32"
) -> bytes:
    digest = hashlib.sha256(b"12345").hexdigest()
    assets = [
        {
            "name": "model.onnx",
            "role": "model",
            "quality": quality,
            "format": "onnx",
            "size": 5,
            "sha256": digest,
        },
        {
            "name": "voices.npz",
            "role": "voices",
            "format": "numpy-npz",
            "size": 5,
            "sha256": digest,
        },
    ]
    return json.dumps(
        {
            "schema": 2,
            "runtime_contract": 1,
            "repository": release_catalog.MODEL_REPOSITORY,
            "tag": tag,
            "profile": "test-profile",
            "model_version": "1.0",
            "generated_at": "2026-01-01T00:00:00+00:00",
            "source": {"type": "test", "repository": "source/repo", "revision": "rev"},
            "license": "Apache-2.0",
            "publication": {"enabled": True},
            "runtime": {
                "language_codes": ["en"],
                "sample_rate": 24000,
                "frontend": frontend,
                "frontend_experimental": False,
                "max_tokens": 510,
                "default_voice": "af",
                "voices": ["af"],
            },
            "onnx_contract": {"inputs": {}, "outputs": {}, "max_tokens": 510},
            "assets": assets,
        }
    ).encode()


class FakeClient:
    def __init__(self, releases: list[dict], manifests: dict[str, bytes]):
        self.releases = releases
        self.manifests = manifests
        self.urls: list[str] = []

    def list_releases(self):
        return self.releases

    def get_release(self, tag: str):
        return next(item for item in self.releases if item["tag_name"] == tag)

    def bytes(self, url: str):
        self.urls.append(url)
        return self.manifests[url]


def _release(tag: str, url: str, **flags: bool) -> dict:
    return {
        "tag_name": tag,
        "published_at": "2026-01-01T00:00:00Z",
        "assets": [
            {"name": "release-manifest.json", "browser_download_url": url},
            {"name": "model.onnx", "browser_download_url": "https://download/model"},
            {"name": "voices.npz", "browser_download_url": "https://download/voices"},
        ],
        **flags,
    }


def test_lists_published_release_and_uses_github_asset_url() -> None:
    data = _manifest()
    client = FakeClient(
        [_release("model-files-test", "https://manifest/test")], {"https://manifest/test": data}
    )
    releases = release_catalog.available_model_releases(client=client)
    assert releases[0].profile == "test-profile"
    assert releases[0].model_asset().download_url == "https://download/model"
    assert client.urls == ["https://manifest/test"]


def test_drafts_and_prereleases_are_ignored() -> None:
    data = _manifest()
    releases = [
        _release("draft", "https://manifest/draft", draft=True),
        _release("pre", "https://manifest/pre", prerelease=True),
        _release("model-files-test", "https://manifest/live"),
    ]
    client = FakeClient(releases, {"https://manifest/live": data})
    assert [
        item.release_tag for item in release_catalog.available_model_releases(client=client)
    ] == ["model-files-test"]


def test_exact_tag_rejects_manifest_tag_mismatch() -> None:
    data = _manifest("different-tag")
    client = FakeClient(
        [_release("model-files-test", "https://manifest/test")], {"https://manifest/test": data}
    )
    with pytest.raises(release_catalog.IncompatibleReleaseError, match="does not match"):
        release_catalog.resolve_model_release("test-profile", tag="model-files-test", client=client)


def test_digest_is_verified() -> None:
    data = _manifest()
    client = FakeClient(
        [_release("model-files-test", "https://manifest/test")],
        {"https://manifest/test": data},
    )
    client.releases[0]["assets"][0]["digest"] = "sha256:" + hashlib.sha256(data).hexdigest()
    assert (
        release_catalog.resolve_model_release("test-profile", client=client).release_tag
        == "model-files-test"
    )
    client.releases[0]["assets"][0]["digest"] = "sha256:" + "0" * 64
    assert release_catalog.available_model_releases(client=client) == ()


def test_unknown_frontend_is_reported_as_incompatible() -> None:
    data = _manifest(frontend="new-frontend-v1")
    client = FakeClient(
        [_release("model-files-test", "https://manifest/test")], {"https://manifest/test": data}
    )
    releases = release_catalog.available_model_releases(include_incompatible=True, client=client)
    assert releases[0].compatible is False
    assert "not implemented" in (releases[0].incompatibility_reason or "")


def test_offline_release_reads_installed_manifest_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "releases" / "test-profile" / "model-files-test"
    root.mkdir(parents=True)
    manifest = json.loads(_manifest().decode())
    for asset in manifest["assets"]:
        path = root / asset["name"]
        path.write_bytes(b"12345")
    (root / "release-manifest.json").write_bytes(_manifest())
    monkeypatch.setattr(release_catalog, "_installed_root", lambda: tmp_path / "releases")
    releases = release_catalog.available_model_releases(offline=True)
    assert releases[0].release_tag == "model-files-test"
    assert releases[0].model_asset().download_url.startswith("file:")


def test_remote_profile_ids_are_open_strings():
    from pykokoro.config_types import ModelVariant

    profile_id: ModelVariant = "new-compatible-profile"
    assert isinstance(profile_id, str)
