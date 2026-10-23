from __future__ import annotations

from pathlib import Path


CUDA_OOM_ERROR_CODE = "cuda_out_of_memory"


def classify_training_failure(log_text: str | None) -> str | None:
    normalized = str(log_text or "").lower()
    if "cuda out of memory" in normalized:
        return CUDA_OOM_ERROR_CODE
    if "cuda error: out of memory" in normalized:
        return CUDA_OOM_ERROR_CODE
    if "torch.outofmemoryerror" in normalized and "cuda" in normalized:
        return CUDA_OOM_ERROR_CODE
    return None


def classify_training_log(log_file: Path) -> str | None:
    try:
        content = log_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return classify_training_failure(content)
