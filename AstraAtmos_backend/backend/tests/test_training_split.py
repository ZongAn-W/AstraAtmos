import pytest

from services.training_split import (
    TrainingSplitError,
    normalize_split_ratios,
    split_sample_ranges,
    split_time_boundary,
    strict_window_split_metadata,
)


def test_new_task_defaults_to_seven_two_one():
    assert normalize_split_ratios({}) == {
        "train_ratio": 0.7,
        "validation_ratio": 0.2,
        "test_ratio": 0.1,
    }


def test_legacy_default_is_explicit_eight_two_zero():
    assert normalize_split_ratios({}, legacy_default=True) == {
        "train_ratio": 0.8,
        "validation_ratio": 0.0,
        "test_ratio": 0.2,
    }


@pytest.mark.parametrize("raw", [
    {"train_ratio": 0.6, "validation_ratio": 0.2, "test_ratio": 0.1},
    {"train_ratio": float("nan"), "validation_ratio": 0.2, "test_ratio": 0.1},
    {"train_ratio": -0.1, "validation_ratio": 0.5, "test_ratio": 0.6},
    {"train_ratio": 1.0, "validation_ratio": 0.0, "test_ratio": 0.0},
])
def test_invalid_ratios_are_rejected(raw):
    with pytest.raises(TrainingSplitError):
        normalize_split_ratios(raw)


def test_largest_remainder_ranges_are_contiguous_and_cover_all_samples():
    ranges = split_sample_ranges(10, {"train_ratio": 0.7, "validation_ratio": 0.2, "test_ratio": 0.1})
    assert ranges == {"train": (0, 7), "validation": (7, 9), "test": (9, 10)}


def test_small_data_requires_one_sample_per_split():
    with pytest.raises(TrainingSplitError):
        split_sample_ranges(2, {"train_ratio": 0.7, "validation_ratio": 0.2, "test_ratio": 0.1})


def test_time_boundary_accounts_for_input_window():
    assert split_time_boundary(10, 3, {"train_ratio": 0.7, "validation_ratio": 0.2, "test_ratio": 0.1}) == {
        "train_start": 0,
        "train_end": 10,
        "validation_start": 7,
        "validation_end": 9,
        "test_start": 9,
        "test_end": 10,
    }


def test_time_boundary_changes_with_window_but_keeps_horizon_out_of_boundary():
    ratios = {"train_ratio": 0.7, "validation_ratio": 0.2, "test_ratio": 0.1}
    short_window = split_time_boundary(20, 2, ratios)
    long_window = split_time_boundary(20, 5, ratios)

    assert isinstance(short_window, dict)
    assert short_window["train_end"] == 16
    assert long_window["train_end"] == 19
    assert long_window["train_end"] - long_window["train_start"] > (
        short_window["train_end"] - short_window["train_start"]
    )


def test_strict_window_partitions_use_requested_window_ratios_and_allow_my_crossing():
    metadata = strict_window_split_metadata(
        time_count=120,
        window=7,
        horizon=3,
        ratios={"train_ratio": 0.7, "validation_ratio": 0.2, "test_ratio": 0.1},
        blocks=[(0, 40), (40, 80), (80, 120)],
    )
    occupied = {}
    counts = {}
    for name in ("train", "validation", "test"):
        starts = [
            start
            for left, right in metadata[name]["window_ranges"]
            for start in range(left, right)
        ]
        assert starts
        counts[name] = len(starts)
        occupied[name] = {
            index
            for start in starts
            for index in range(start, start + 10)
        }
    assert not (occupied["train"] & occupied["validation"])
    assert not (occupied["train"] & occupied["test"])
    assert not (occupied["validation"] & occupied["test"])
    assert counts == {"train": 65, "validation": 19, "test": 9}
    assert any(start < 40 <= start + 9 for start in [
        start
        for left, right in metadata["train"]["window_ranges"]
        for start in range(left, right)
    ])


def test_strict_window_partitions_honor_a_zero_validation_ratio():
    metadata = strict_window_split_metadata(
        time_count=120,
        window=7,
        horizon=3,
        ratios={"train_ratio": 0.8, "validation_ratio": 0.0, "test_ratio": 0.2},
    )
    counts = {
        name: sum(right - left for left, right in metadata[name]["window_ranges"])
        for name in ("train", "validation", "test")
    }
    assert counts == {"train": 82, "validation": 0, "test": 20}
    assert metadata["validation"]["window_ranges"] == []
