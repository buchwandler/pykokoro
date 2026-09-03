"""Compatibility import for examples executed through importlib or runpy."""

from examples._output import ARTIFACT_DIR, PROJECT_ROOT, artifact_dir, artifact_path

__all__ = ["ARTIFACT_DIR", "PROJECT_ROOT", "artifact_dir", "artifact_path"]
