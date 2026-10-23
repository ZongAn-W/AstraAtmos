"""Loss histories come from epoch summaries, including Earth metric labels."""

import pytest


@pytest.fixture
def training_service(monkeypatch):
    import services.training_service as module

    def forbidden_io(*_args, **_kwargs):
        pytest.fail("progress parsing must not access the database or launch training")

    monkeypatch.setattr(module, "async_session_maker", forbidden_io)
    monkeypatch.setattr(module.subprocess, "Popen", forbidden_io)
    monkeypatch.setattr(module.time, "time", lambda: 130.0)
    return module.TrainingService()


@pytest.mark.parametrize("label", [
    " (area-weighted normalized MSE)",
    " (normalized MSE)",
    "",
], ids=["earth-v3-area-weighted", "earth-v2-uniform", "mars-unlabelled"])
def test_epoch_summary_keeps_full_precision_validation_history(training_service, label):
    history = {"train": [], "val": []}
    progress = training_service._parse_progress_from_log(
        f"Epoch 1/3 Loss=0.131582{label} Val Loss=0.067518{label}",
        3, 100.0, history,
    )

    assert progress["current_epoch"] == 1
    assert progress["total_epochs"] == 3
    assert progress["progress"] == pytest.approx(33.33)
    assert progress["current_loss"] == pytest.approx(0.1316)
    assert progress["val_loss"] == pytest.approx(0.0675)
    assert progress["loss_history"] is history
    assert history == {"train": [0.131582], "val": [0.067518]}


def test_area_weighted_epoch_sequence_survives_batch_lines_and_duplicate_summaries(training_service):
    history = {"train": [], "val": []}
    values = [(0.131582, 0.067518), (0.101234, 0.061234), (0.091234, 0.051234)]
    for epoch, (train, val) in enumerate(values, 1):
        batch = training_service._parse_progress_from_log(
            f"Epoch {epoch}/3 Batch 2/4 Loss=0.500000 (area-weighted normalized MSE)",
            3, 100.0, history,
        )
        assert "loss_history" not in batch
        assert len(history["train"]) == epoch - 1

        line = (
            f"Epoch {epoch}/3 Loss={train:.6f} (area-weighted normalized MSE) "
            f"Val Loss={val:.6f} (area-weighted normalized MSE)"
        )
        progress = training_service._parse_progress_from_log(line, 3, 100.0, history)
        training_service._parse_progress_from_log(line, 3, 100.0, history)
        assert progress["val_loss"] == round(val, 4)
        assert len(history["train"]) == epoch

    assert progress["progress"] == 100.0
    assert history == {
        "train": [train for train, _ in values],
        "val": [val for _, val in values],
    }


def test_replayed_summary_repairs_previously_missing_validation_value(training_service):
    history = {"train": [0.131582], "val": [None]}
    progress = training_service._parse_progress_from_log(
        "Epoch 1/3 Loss=0.131582 (area-weighted normalized MSE) "
        "Val Loss=0.067518 (area-weighted normalized MSE)",
        3, 100.0, history,
    )

    assert progress["val_loss"] == pytest.approx(0.0675)
    assert history == {"train": [0.131582], "val": [0.067518]}


@pytest.mark.parametrize("label", ["", " (normalized MSE)", " (area-weighted normalized MSE)"])
def test_train_only_summary_preserves_missing_validation_as_none(training_service, label):
    history = {"train": [], "val": []}
    progress = training_service._parse_progress_from_log(
        f"Epoch 1/3 Loss=0.131582{label}", 3, 100.0, history,
    )

    assert progress["val_loss"] is None
    assert history == {"train": [0.131582], "val": [None]}


@pytest.mark.parametrize("train_token,val_token", [
    ("1.31582e-1", "6.7518E-2"),
    ("+1.31582E-01", "+6.7518e-02"),
    (".131582", ".067518"),
])
def test_epoch_summary_accepts_complete_numeric_tokens(training_service, train_token, val_token):
    history = {"train": [], "val": []}
    progress = training_service._parse_progress_from_log(
        f"Epoch 1/3 Loss={train_token} (area-weighted normalized MSE) "
        f"Val Loss={val_token} (area-weighted normalized MSE)",
        3, 100.0, history,
    )

    assert progress["current_loss"] == pytest.approx(0.1316)
    assert progress["val_loss"] == pytest.approx(0.0675)
    assert history["train"] == pytest.approx([0.131582])
    assert history["val"] == pytest.approx([0.067518])


@pytest.mark.parametrize("loss_token", ["0.131582", "1.31582e-1"])
def test_batch_update_does_not_create_an_epoch_loss_point(training_service, loss_token):
    history = {"train": [0.2], "val": [0.1]}
    progress = training_service._parse_progress_from_log(
        f"Epoch 2/3 Batch 2/4 Loss={loss_token} (area-weighted normalized MSE)",
        3, 100.0, history,
    )

    assert progress["current_epoch"] == 2
    assert progress["progress"] == 50.0
    assert progress["current_loss"] == pytest.approx(0.1316)
    assert "val_loss" not in progress
    assert "loss_history" not in progress
    assert history == {"train": [0.2], "val": [0.1]}


def test_non_epoch_validation_diagnostic_is_not_loss_history(training_service):
    history = {"train": [], "val": []}
    assert training_service._parse_progress_from_log(
        "Validation RMSE=14.334705 DU MAE=8.664098 DU", 3, 100.0, history,
    ) is None
    assert history == {"train": [], "val": []}
