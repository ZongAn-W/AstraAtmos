"""Canonical Mars directory/manifest identity, shared by training and inference."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def _natural_key(value: Any) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"([0-9]+)", str(value))]


def file_manifest_for_directory(directory: Any) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(Path(directory).glob("*.nc"), key=_natural_key):
        stat = path.stat()
        entries.append({"name": path.name, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    return entries


def directory_fingerprint(directory: Any, manifest=None) -> str:
    """Preserve the existing single-directory fingerprint serialization."""
    root = Path(directory).expanduser().resolve()
    payload = {"path": str(root), "files": file_manifest_for_directory(root) if manifest is None else list(manifest)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def canonical_dataset_identity(
    *, training_dataset: Any, openmars_dir=None, mcd_dir=None, raw_dir=None,
    data_directories=None, file_manifest=None,
) -> dict[str, Any]:
    """Read live metadata, or reproduce a captured identity without rescanning.

    Captured directories and manifest let checkpoints retain the identity of
    the data actually loaded for training, even if files later change.
    """
    dataset = str(training_dataset or "openmars_mcd").strip().lower()
    if dataset not in ("mcd_overview", "openmars_mcd"):
        raise ValueError(f"Unsupported Mars training dataset: {training_dataset}")
    if data_directories is None:
        data_directories = [raw_dir] if dataset == "mcd_overview" else [openmars_dir, mcd_dir]
    count = 1 if dataset == "mcd_overview" else 2
    if not isinstance(data_directories, (list, tuple)) or len(data_directories) != count or any(not value for value in data_directories):
        raise ValueError("Mars dataset identity requires all bound data directories")
    directories = [str(Path(value).expanduser().resolve()) for value in data_directories]
    if file_manifest is None:
        if dataset == "mcd_overview":
            manifest = file_manifest_for_directory(directories[0])
        else:
            manifest = [
                {"root": root, **entry}
                for root, directory in zip(("openmars", "mcd"), directories)
                for entry in file_manifest_for_directory(directory)
            ]
    else:
        if not isinstance(file_manifest, (list, tuple)):
            raise ValueError("Mars dataset file manifest must be an array")
        manifest = [dict(entry) for entry in file_manifest]
    for entry in manifest:
        if (
            not isinstance(entry.get("name"), str)
            or Path(entry["name"]).name != entry["name"]
            or any(type(entry.get(key)) is not int or entry[key] < 0 for key in ("size", "mtime_ns"))
            or (dataset == "openmars_mcd" and entry.get("root") not in ("openmars", "mcd"))
        ):
            raise ValueError("Mars dataset file manifest is invalid")
    if dataset == "mcd_overview":
        fingerprint = directory_fingerprint(directories[0], manifest)
    else:
        # This is the existing two-directory snapshot algorithm, including root tags.
        payload = {"openmars_dir": directories[0], "mcd_dir": directories[1], "files": manifest}
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "dataset_id": dataset, "planet": "mars",
        "source_type": "mcd_raw_3h" if dataset == "mcd_overview" else "openmars_mcd",
        "data_directories": directories, "file_manifest": manifest,
        "dataset_fingerprint": fingerprint,
        # Compatibility only: the primary directory cannot replace the full binding.
        "data_dir": directories[-1],
    }


def has_mars_file_identity(value: Any) -> bool:
    return isinstance(value, dict) and any(
        key in value and value[key] is not None
        for key in ("data_dir", "data_directories", "manifest", "file_manifest", "dataset_fingerprint", "file_fingerprint")
    )


def normalize_mars_dataset_identity(value: Any, *, dataset_id=None) -> dict[str, Any]:
    """Accept explicit historical aliases, and reject partial or conflicting evidence."""
    if not isinstance(value, dict) or value.get("planet", "mars") != "mars":
        raise ValueError("Mars dataset identity metadata is missing or invalid")
    dataset = value.get("dataset_id") or dataset_id
    if dataset is None or (dataset_id is not None and dataset != dataset_id):
        raise ValueError("Mars dataset identity does not match the training dataset")
    directories = value.get("data_directories")
    if not directories and dataset == "mcd_overview" and value.get("data_dir"):
        directories = [value["data_dir"]]
    manifest = value.get("file_manifest", value.get("manifest"))
    fingerprint = value.get("dataset_fingerprint") or value.get("file_fingerprint")
    if manifest is None or not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("Mars dataset identity requires a file manifest and fingerprint")
    identity = canonical_dataset_identity(training_dataset=dataset, data_directories=directories, file_manifest=manifest)
    if fingerprint != identity["dataset_fingerprint"]:
        raise ValueError("Mars dataset identity fingerprint does not match its manifest")
    for key in ("dataset_fingerprint", "file_fingerprint"):
        if value.get(key) is not None and value[key] != fingerprint:
            raise ValueError("Mars dataset identity fingerprint aliases disagree")
    for key in ("source_type", "data_source_type"):
        if value.get(key) is not None and value[key] != identity["source_type"]:
            raise ValueError("Mars dataset identity source type is invalid")
    if not value.get("source_type") and not value.get("data_source_type"):
        raise ValueError("Mars dataset identity source type is missing")
    if "manifest" in value and value["manifest"] != identity["file_manifest"]:
        raise ValueError("Mars dataset identity manifest aliases disagree")
    if value.get("data_dir") and str(Path(value["data_dir"]).expanduser().resolve()) != identity["data_dir"]:
        raise ValueError("Mars dataset identity primary directory disagrees with its binding")
    return identity


def volume_dataset_identity(volume: Any) -> dict[str, Any]:
    directories = getattr(volume, "data_directories", ())
    if not directories and volume.dataset_id == "mcd_overview":
        directories = [volume.data_dir]
    return normalize_mars_dataset_identity({
        "dataset_id": volume.dataset_id, "source_type": volume.source_type,
        "data_directories": list(directories), "file_manifest": list(volume.manifest),
        "dataset_fingerprint": getattr(volume, "fingerprint", getattr(volume, "dataset_fingerprint", None)),
        "data_dir": str(volume.data_dir),
    })
