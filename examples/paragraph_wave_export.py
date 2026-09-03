#!/usr/bin/env python3
"""Export one WAV per prepared paragraph with an atomic resume manifest.

This example keeps only one paragraph waveform live at a time. Preparation still
runs document parsing, G2P, and phoneme preprocessing globally so SSMD spans,
paragraph boundaries, voices, pauses, and markers remain deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import soundfile as sf

try:
    from ._output import artifact_dir
except ImportError:
    from _output import artifact_dir

from pykokoro import GenerationConfig, KokoroPipeline, PipelineConfig

MANIFEST_SCHEMA = "pykokoro-paragraph-export-v1"
DEFAULT_TEXT = """# A Short Demonstration

The first paragraph is rendered and persisted independently.

The second paragraph can be skipped on a resumed run when its identity hash and WAV
file still match the manifest.

The third paragraph proves that preparation is global while audio generation remains
bounded to one output unit at a time.
"""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    """Convert metadata and diagnostics into JSON-compatible values."""
    return json.loads(json.dumps(value, default=str))


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_write_wav(path: Path, audio: Any, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.stem}.{uuid4().hex}.tmp.wav")
    try:
        sf.write(temp_path, audio, sample_rate)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _load_manifest(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to read resume manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Resume manifest {path} must contain a JSON object")
    return value


def _read_source(path: Path | None) -> str:
    return path.read_text(encoding="utf-8") if path is not None else DEFAULT_TEXT


def _entry_filename(index: int, text_hash: str) -> str:
    return f"paragraph-{index + 1:08d}-{text_hash[:12]}.wav"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="UTF-8 SSMD/Markdown input")
    parser.add_argument("--output-dir", type=Path, default=artifact_dir() / "paragraph-waves")
    parser.add_argument("--voice", default="af_sarah")
    parser.add_argument("--lang", default="en-us")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip manifest entries whose unit hash and WAV file still match",
    )
    args = parser.parse_args()

    text = _read_source(args.input)
    source_hash = _sha256_text(text)
    output_dir: Path = args.output_dir
    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    previous = _load_manifest(manifest_path) if args.resume else None
    previous_entries = {
        int(entry["index"]): entry
        for entry in (previous or {}).get("units", [])
        if isinstance(entry, dict) and isinstance(entry.get("index"), int)
    }
    resume_compatible = bool(
        previous
        and previous.get("schema") == MANIFEST_SCHEMA
        and previous.get("source_sha256") == source_hash
        and previous.get("voice") == args.voice
        and previous.get("language") == args.lang
    )
    if args.resume and previous and not resume_compatible:
        print("Existing manifest is incompatible; all paragraphs will be rendered again.")

    config = PipelineConfig(
        voice=args.voice,
        generation=GenerationConfig(lang=args.lang),
        retain_segment_audio=False,
    )

    with (
        KokoroPipeline(config) as pipeline,
        pipeline.prepare_units(text, unit="paragraph") as prepared,
    ):
        units: list[dict[str, Any]] = []
        completed: set[int] = set()
        for descriptor in prepared.units:
            filename = _entry_filename(descriptor.index, descriptor.text_hash)
            entry: dict[str, Any] = {
                "index": descriptor.index,
                "paragraph_idx": descriptor.paragraph_idx,
                "char_start": descriptor.char_start,
                "char_end": descriptor.char_end,
                "text": descriptor.text,
                "text_hash": descriptor.text_hash,
                "segment_ids": list(descriptor.segment_ids),
                "phoneme_segment_ids": list(descriptor.phoneme_segment_ids),
                "marker_names": list(descriptor.marker_names),
                "filename": filename,
                "status": "pending",
                "sample_rate": None,
                "sample_count": None,
            }
            old = previous_entries.get(descriptor.index) if resume_compatible else None
            output_path = output_dir / filename
            if (
                old
                and old.get("text_hash") == descriptor.text_hash
                and old.get("filename") == filename
                and old.get("status") == "complete"
                and output_path.is_file()
            ):
                entry["status"] = "complete"
                entry["sample_rate"] = old.get("sample_rate")
                entry["sample_count"] = old.get("sample_count")
                completed.add(descriptor.index)
            units.append(entry)

        manifest: dict[str, Any] = {
            "schema": MANIFEST_SCHEMA,
            "source": str(args.input) if args.input else "<built-in>",
            "source_sha256": source_hash,
            "voice": args.voice,
            "language": args.lang,
            "document_metadata": _json_safe(dict(prepared.document_metadata)),
            "diagnostics": _json_safe(list(prepared.diagnostics)),
            "units": units,
        }
        _atomic_write_json(manifest_path, manifest)

        print(f"Prepared {len(prepared.units)} paragraph units")
        print(f"Skipping {len(completed)} completed units")

        for result in prepared.render(skip_indices=completed):
            entry = units[result.descriptor.index]
            output_path = output_dir / str(entry["filename"])
            try:
                _atomic_write_wav(output_path, result.audio, result.sample_rate)
                entry["status"] = "complete"
                entry["sample_rate"] = result.sample_rate
                entry["sample_count"] = len(result.audio)
                entry["markers"] = _json_safe(result.markers)
                _atomic_write_json(manifest_path, manifest)
                print(f"Wrote {output_path}")
            finally:
                # Destructive and idempotent. The iterator also releases the
                # previous result before advancing, so persist/copy inside the loop.
                result.release_audio()

    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
