"""Earth window construction, channel selection and training-only normalization."""

import numpy as np
import pytest

from services.dataset_registry import DatasetRegistry
from services.earth_dataset import (
    CHANNELS,
    EarthOzoneWindows,
    canonical_input_channels,
    reference_ozone,
    release_date_index,
    release_split_codes,
)


def _registry(fixture):
    return DatasetRegistry(earth_dataset_id="earth_merra2_daily_v2", **fixture)


def test_published_window_counts_and_split_boundaries(earth_global_release):
    registry = _registry(earth_global_release)
    release = registry.get_earth_snapshot("earth_merra2_daily_v2")
    train = EarthOzoneWindows.from_release(release, split="train", window=7, horizon=3, normalization=None)
    validation = EarthOzoneWindows.from_release(
        release, split="validation", window=7, horizon=3, normalization=train.normalization
    )
    test = EarthOzoneWindows.from_release(
        release, split="test", window=7, horizon=3, normalization=train.normalization
    )
    # 366/181/184 published days -> 357/172/175 sample-minus-window-horizon-plus-one.
    assert (len(train), len(validation), len(test)) == (357, 172, 175)
    assert str(train.dates[0]) == "2020-01-01"
    assert str(train.dates[-1]) == "2020-12-31"
    assert str(validation.dates[0]) == "2021-01-01"
    assert str(validation.dates[-1]) == "2021-06-30"
    assert str(test.dates[0]) == "2021-07-01"
    assert str(test.dates[-1]) == "2021-12-31"


def test_first_and_last_window_dates_never_borrow_from_another_split(earth_global_release):
    registry = _registry(earth_global_release)
    release = registry.get_earth_snapshot("earth_merra2_daily_v2")
    train = EarthOzoneWindows.from_release(release, split="train", window=7, horizon=3, normalization=None)
    # First train window: inputs 01-01..01-07, targets 01-08..01-10.
    assert str(train.dates[0]) == "2020-01-01"
    assert str(train.dates[7]) == "2020-01-08"
    assert str(train.dates[9]) == "2020-01-10"
    # The last train sample (index 356) owns targets 357..359 = 12-29..12-31, so no
    # window may reach into January 2021.
    assert str(train.dates[356 + 7]) == "2020-12-29"
    assert str(train.dates[356 + 9]) == "2020-12-31"

    test = EarthOzoneWindows.from_release(
        release, split="test", window=7, horizon=3, normalization=train.normalization
    )
    assert str(test.dates[0]) == "2021-07-01"
    assert str(test.dates[7]) == "2021-07-08"
    assert str(test.dates[9]) == "2021-07-10"
    assert str(test.dates[174 + 7]) == "2021-12-29"
    assert str(test.dates[174 + 9]) == "2021-12-31"


def test_targets_start_exactly_one_day_after_the_last_input(earth_global_release):
    registry = _registry(earth_global_release)
    release = registry.get_earth_snapshot("earth_merra2_daily_v2")
    train = EarthOzoneWindows.from_release(
        release, split="train", window=7, horizon=3, selected_channels=["T2M"], normalization=None
    )
    windows = EarthOzoneWindows.from_release(
        release,
        split="test",
        window=7,
        horizon=3,
        selected_channels=["T2M"],
        normalization=train.normalization,
    )
    for index in (0, 1, len(windows) // 2, len(windows) - 1):
        inputs_end = windows.dates[index + windows.window - 1]
        first_target = windows.dates[index + windows.window]
        assert first_target - inputs_end == np.timedelta64(1, "D")


def test_window_tensor_shapes_and_channel_order(earth_global_release):
    registry = _registry(earth_global_release)
    release = registry.get_earth_snapshot("earth_merra2_daily_v2")
    train = EarthOzoneWindows.from_release(
        release, split="train", window=7, horizon=3, selected_channels=["SWGDN", "U10M"], normalization=None
    )
    windows = EarthOzoneWindows.from_release(
        release,
        split="test",
        window=7,
        horizon=3,
        selected_channels=["SWGDN", "U10M"],
        normalization=train.normalization,
    )
    # Request order never decides model order.
    assert windows.input_channels == ["TO3", "U10M", "SWGDN"]
    assert windows.input_units == ["DU", "m s-1", "W m-2"]
    inputs, targets = windows[0]
    assert inputs.shape == (7, 3, 36, 72)
    assert targets.shape == (3, 1, 36, 72)
    assert inputs.dtype == np.float32 and targets.dtype == np.float32
    assert np.isfinite(inputs).all() and np.isfinite(targets).all()


def test_ozone_only_model_has_a_single_input_channel(earth_global_release):
    registry = _registry(earth_global_release)
    release = registry.get_earth_snapshot("earth_merra2_daily_v2")
    windows = EarthOzoneWindows.from_release(
        release, split="train", window=7, horizon=3, selected_channels=[], normalization=None
    )
    assert windows.input_channels == ["TO3"]
    inputs, targets = windows[0]
    assert inputs.shape == (7, 1, 36, 72)
    assert targets.shape == (3, 1, 36, 72)


def test_reference_targets_round_trip_to_the_published_du_field(earth_global_release):
    registry = _registry(earth_global_release)
    release = registry.get_earth_snapshot("earth_merra2_daily_v2")
    train = EarthOzoneWindows.from_release(
        release, split="train", window=7, horizon=3, selected_channels=["U10M"], normalization=None
    )
    windows = EarthOzoneWindows.from_release(
        release,
        split="test",
        window=7,
        horizon=3,
        selected_channels=["U10M"],
        normalization=train.normalization,
    )
    inputs, targets = windows[0]
    # Sanity: the normalized TO3 input is not a constant placeholder.
    assert float(np.std(inputs[:, 0])) > 0.1
    start = release_date_index(release, str(windows.dates[7]))
    expected = np.asarray(release.fields["TO3"], dtype="float32")[start:start + 3]
    restored = windows.denormalize_ozone(targets[:, 0])
    np.testing.assert_allclose(restored, expected, rtol=1e-6, atol=1e-4)


def test_denormalize_keeps_the_tensor_type(earth_global_release):
    torch = pytest.importorskip("torch")
    registry = _registry(earth_global_release)
    release = registry.get_earth_snapshot("earth_merra2_daily_v2")
    windows = EarthOzoneWindows.from_release(
        release, split="train", window=7, horizon=3, selected_channels=["T2M"], normalization=None
    )
    tensor = torch.zeros((2, 3))
    restored = windows.denormalize_ozone(tensor)
    assert isinstance(restored, torch.Tensor)
    assert float(restored[0, 0]) == pytest.approx(float(windows.normalization["mean"][0]))
    array = windows.denormalize_ozone(np.zeros((2, 3), dtype="float32"))
    assert isinstance(array, np.ndarray)


def test_normalization_is_fitted_on_training_dates_only(earth_global_release):
    registry = _registry(earth_global_release)
    release = registry.get_earth_snapshot("earth_merra2_daily_v2")
    train = EarthOzoneWindows.from_release(release, split="train", window=7, horizon=3, normalization=None)
    block = train.normalization
    assert block["method"] == "per_channel_standard"
    assert block["fit_split"] == "train"
    assert block["fit_date_start"] == "2020-01-01"
    assert block["fit_date_end"] == "2020-12-31"
    assert block["ddof"] == 0
    assert block["epsilon"] == 1e-6
    assert block["target_channel_index"] == 0
    assert block["channel_order"] == list(CHANNELS)

    # Recomputing from the training days reproduces the stored statistics.
    dates = np.asarray(release.dates, dtype="datetime64[D]")
    training = (dates >= np.datetime64("2020-01-01")) & (dates <= np.datetime64("2020-12-31"))
    for index, name in enumerate(CHANNELS):
        values = np.asarray(release.fields[name], dtype="float64")[training]
        assert block["mean"][index] == pytest.approx(float(values.mean()), rel=1e-6, abs=1e-4)
        expected_std = float(values.std())
        assert block["scale"][index] == pytest.approx(
            expected_std if expected_std >= 1e-6 else 1.0, rel=1e-6, abs=1e-4
        )


def test_validation_and_test_values_cannot_move_the_training_statistics(earth_global_release):
    """Prove no leakage by moving every held-out value by a huge offset.

    The fitting entry point only ever sees the training dates, so the statistics
    must be bit-identical even when validation and test are replaced by nonsense.
    """
    from dataclasses import replace

    from services.earth_dataset import fit_normalization, stack_release_cube

    registry = _registry(earth_global_release)
    release = registry.get_earth_snapshot("earth_merra2_daily_v2")
    dates = np.asarray(release.dates, dtype="datetime64[D]")
    split_ids = release_split_codes(dates, release.metadata)
    channels = ["TO3", "U10M"]

    original_cube = stack_release_cube(release)
    baseline = EarthOzoneWindows.from_release(
        release, split="train", window=7, horizon=3, selected_channels=["U10M"], normalization=None
    )

    corrupted_cube = original_cube.copy()
    corrupted_cube[split_ids != 0] += 500.0
    corrupted_release = replace(
        release,
        fields={name: corrupted_cube[:, index] for index, name in enumerate(CHANNELS)},
    )
    # The corrupted cube differs on every held-out day...
    assert not np.allclose(corrupted_cube[split_ids != 0], original_cube[split_ids != 0])
    # ...yet fitting on the training days alone is unchanged.
    refit = fit_normalization(
        corrupted_cube[split_ids == 0], channels, fit_dates=dates[split_ids == 0]
    )
    assert refit["mean"] == baseline.normalization["mean"]
    assert refit["scale"] == baseline.normalization["scale"]

    # The existing windows keep using the stored block rather than refitting.
    reused = EarthOzoneWindows.from_release(
        release, split="test", window=7, horizon=3,
        selected_channels=["U10M"], normalization=baseline.normalization,
    )
    assert reused.normalization == baseline.normalization

    # And the corrupted release can only ever be read through the saved block,
    # which still normalizes with the training statistics.
    corrupted = EarthOzoneWindows.from_release(
        corrupted_release, split="test", window=7, horizon=3,
        selected_channels=["U10M"], normalization=baseline.normalization,
    )
    assert corrupted.normalization == baseline.normalization
    assert not np.allclose(reused.data, corrupted.data)
    assert np.isfinite(corrupted.data).all()


def test_constant_channel_keeps_scale_one(earth_global_release):
    from services.earth_dataset import fit_normalization, stack_release_cube

    registry = _registry(earth_global_release)
    release = registry.get_earth_snapshot("earth_merra2_daily_v2")
    cube = stack_release_cube(release).copy()
    cube[:, CHANNELS.index("U10M")] = 3.5
    dates = np.asarray(release.dates, dtype="datetime64[D]")
    training = dates <= np.datetime64("2020-12-31")
    block = fit_normalization(cube[training], ["TO3", "U10M"], fit_dates=dates[training])
    # A near-constant channel keeps a scale of 1.0 instead of dividing into NaN.
    assert block["scale"][1] == 1.0
    assert block["constant_channel_mask"][1] is True
    assert block["constant_channel_mask"][0] is False


def test_non_finite_values_fail_the_fit(earth_global_release):
    from services.earth_dataset import fit_normalization, stack_release_cube

    release = _registry(earth_global_release).get_earth_snapshot("earth_merra2_daily_v2")
    cube = stack_release_cube(release).copy()
    cube[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_normalization(cube, ["TO3"])


def test_validation_cannot_fit_its_own_statistics(earth_global_release):
    registry = _registry(earth_global_release)
    release = registry.get_earth_snapshot("earth_merra2_daily_v2")
    with pytest.raises(ValueError):
        EarthOzoneWindows.from_release(release, split="test", window=7, horizon=3, normalization=None)


def test_saved_normalization_must_match_the_selected_channels(earth_global_release):
    registry = _registry(earth_global_release)
    release = registry.get_earth_snapshot("earth_merra2_daily_v2")
    train = EarthOzoneWindows.from_release(
        release, split="train", window=7, horizon=3, selected_channels=["U10M"], normalization=None
    )
    with pytest.raises(ValueError):
        EarthOzoneWindows.from_release(
            release,
            split="test",
            window=7,
            horizon=3,
            selected_channels=["T2M"],
            normalization=train.normalization,
        )


def test_tampered_normalization_is_rejected(earth_global_release):
    import copy

    registry = _registry(earth_global_release)
    release = registry.get_earth_snapshot("earth_merra2_daily_v2")
    train = EarthOzoneWindows.from_release(
        release, split="train", window=7, horizon=3, selected_channels=["U10M"], normalization=None
    )
    for mutate in (
        lambda block: block.update({"fit_split": "test"}),
        lambda block: block.update({"scale": [0.0, 1.0]}),
        lambda block: block.update({"mean": [float("nan"), 1.0]}),
        lambda block: block.update({"channel_order": ["TO3", "T2M"]}),
        lambda block: block.update({"scale": [1.0]}),
    ):
        block = copy.deepcopy(train.normalization)
        mutate(block)
        with pytest.raises(ValueError):
            EarthOzoneWindows.from_release(
                release, split="test", window=7, horizon=3, selected_channels=["U10M"], normalization=block
            )


def test_expectation_window_reads_a_plain_date_range(earth_global_release):
    registry = _registry(earth_global_release)
    release = registry.get_earth_snapshot("earth_merra2_daily_v2")
    train = EarthOzoneWindows.from_release(
        release, split="train", window=7, horizon=3, selected_channels=["T2M"], normalization=None
    )
    expectation = EarthOzoneWindows.from_release(
        release,
        split="test",
        window=7,
        horizon=3,
        selected_channels=["T2M"],
        normalization=train.normalization,
        require_split_coverage=False,
    )
    assert len(expectation.dates) == len(release.dates)
    origin = int(np.flatnonzero(expectation.dates == np.datetime64("2021-07-08"))[0])
    assert str(expectation.dates[origin - 6]) == "2021-07-02"
    assert expectation.data[origin - 6:origin + 1].shape == (7, 2, 36, 72)


def test_reference_ozone_and_date_lookup(earth_global_release):
    registry = _registry(earth_global_release)
    release = registry.get_earth_snapshot("earth_merra2_daily_v2")
    reference = reference_ozone(release, "2021-07-09", 3)
    assert reference.shape == (3, 36, 72)
    start = release_date_index(release, "2021-07-09")
    np.testing.assert_allclose(reference, np.asarray(release.fields["TO3"])[start:start + 3])
    with pytest.raises(KeyError):
        release_date_index(release, "2019-01-01")


def test_split_codes_come_from_the_published_manifest(earth_global_release):
    registry = _registry(earth_global_release)
    release = registry.get_earth_snapshot("earth_merra2_daily_v2")
    codes = release_split_codes(np.asarray(release.dates, dtype="datetime64[D]"), release.metadata)
    assert set(np.unique(codes).tolist()) == {0, 1, 2}
    # 181 validation days at code 1 plus 184 test days at code 2.
    assert int(codes.sum()) == 181 + (184 * 2)
    assert int((codes == 0).sum()) == 366
    with pytest.raises(ValueError):
        release_split_codes(np.asarray(release.dates, dtype="datetime64[D]"), {})


def test_canonical_channel_helper_rejects_unknown_and_duplicates():
    assert canonical_input_channels(None) == list(CHANNELS)
    assert canonical_input_channels(["SWGDN", "U10M"]) == ["TO3", "U10M", "SWGDN"]
    for bad in (["U"], ["U10M", "U10M"], ["TO3", "TO3"]):
        with pytest.raises(ValueError):
            canonical_input_channels(bad)


def test_path_based_constructor_keeps_the_original_behaviour(earth_global_release):
    path = earth_global_release["earth_package_dir"] / "earth_merra2_daily.nc"
    windows = EarthOzoneWindows(path, split="train", window=7, horizon=3)
    assert len(windows) == 357
    assert windows.input_channels == list(CHANNELS)
    assert windows[0][0].shape == (7, 5, 36, 72)
