from __future__ import annotations

from pathlib import Path


def is_valid_model_weight_file(path: str | Path | None) -> bool:
    if not path:
        return False

    try:
        weight_path = Path(path)
        return weight_path.is_file() and weight_path.stat().st_size > 0
    except OSError:
        return False
