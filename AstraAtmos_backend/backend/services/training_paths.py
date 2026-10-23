from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

from config import TRAINING_RESULTS_DIR


MAX_MODEL_NAME_LENGTH = 120


def sanitize_model_name(value: str | None) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    safe_value = re.sub(r"[^A-Za-z0-9_-]+", "_", ascii_value).strip("_-")
    safe_value = safe_value or "model"
    if len(safe_value) <= MAX_MODEL_NAME_LENGTH:
        return safe_value

    digest = hashlib.sha256(safe_value.encode("ascii")).hexdigest()[:12]
    prefix_length = MAX_MODEL_NAME_LENGTH - len(digest) - 1
    return f"{safe_value[:prefix_length]}-{digest}"


def build_task_output_path(
    task_id: int,
    custom_model_name: str | None,
    results_dir: Path = TRAINING_RESULTS_DIR,
) -> Path:
    filename = f"task_{int(task_id)}_{sanitize_model_name(custom_model_name)}.pth"
    return Path(results_dir) / filename
