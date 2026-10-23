from pathlib import Path


def resolve_legacy_data_dirs(base_dir):
    base = Path(base_dir)
    return base / "data" / "openmars", base / "data" / "MCD"
