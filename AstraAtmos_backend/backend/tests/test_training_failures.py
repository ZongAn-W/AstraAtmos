from pathlib import Path

import pytest

from services.training_failures import (
    CUDA_OOM_ERROR_CODE,
    classify_training_failure,
    classify_training_log,
)


@pytest.mark.parametrize(
    "log_text",
    [
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 22.00 MiB",
        "RuntimeError: CUDA error: out of memory",
        "TORCH.OUTOFMEMORYERROR while allocating on CUDA device 0",
    ],
)
def test_explicit_cuda_oom_signatures_are_classified(log_text):
    assert classify_training_failure(log_text) == CUDA_OOM_ERROR_CODE


@pytest.mark.parametrize(
    "log_text",
    [
        "MemoryError: out of memory",
        "torch.OutOfMemoryError: CPU allocator failed",
        "RuntimeError: training process exited with code 1",
        "",
    ],
)
def test_non_cuda_failures_are_not_classified_as_cuda_oom(log_text):
    assert classify_training_failure(log_text) is None


def test_completed_training_log_is_read_for_classification(tmp_path: Path):
    log_file = tmp_path / "task.log"
    log_file.write_text("CUDA out of memory", encoding="utf-8")

    assert classify_training_log(log_file) == CUDA_OOM_ERROR_CODE


def test_unreadable_training_log_is_not_classified(tmp_path: Path):
    assert classify_training_log(tmp_path / "missing.log") is None
