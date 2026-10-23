"""Opt-in, read-only integration of the configured complete production release."""

import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import config
from services.earth_dataset import _threehour_values, threehour_release_split_codes
from services.earth_dataset_metadata import package_signature
from services.earth_training_artifact import threehour_available_origin_range
from services.netcdf_read_lock import netcdf_read_lock
from services.training_service import TrainingService


@pytest.mark.skipif(not os.environ.get("ASTRAATMOS_TEST_EARTH_3HOURLY_PACKAGE"), reason="requires explicit read-only full package")
def test_configured_full_package_descriptor_and_uploaded_window_contract():
    root = Path(os.environ["ASTRAATMOS_TEST_EARTH_3HOURLY_PACKAGE"])
    assert config.EARTH_MERRA2_3HOURLY_DIR.resolve() == root.resolve()
    before = package_signature(root, data_file_name="earth_merra2_3hourly.nc")
    registry = TrainingService._default_dataset_registry()
    descriptor = registry.get_dataset("earth_merra2_3hourly_v1")
    assert descriptor["availability"] == "available", descriptor.get("availability_reason")
    assert descriptor["time"]["count"] == 5848 and descriptor["grid"]["shape"] == [240, 480]
    assert descriptor["training_profile"]["model_sources"] == ["official", "uploaded"]
    assert descriptor["channel_order"] == ["TO3", "U10M", "V10M", "T2M", "SWGDN"]
    assert [v["units"] for v in descriptor["variables"]] == ["DU", "m s-1", "m s-1", "K", "W m-2"]
    release = registry.get_earth_snapshot("earth_merra2_3hourly_v1")
    origins = threehour_available_origin_range(release, strict_split=True)
    assert origins["count"] == 2849 + 1369 + 1393
    codes = threehour_release_split_codes(release.dates, release.metadata)
    with netcdf_read_lock(), xr.open_dataset(release.data_path, engine="netcdf4", mask_and_scale=False) as ds:
        for code in (0, 1, 2):
            start = int(np.flatnonzero(codes == code)[0])
            assert np.all(codes[start:start + 80] == code)
            fields = []
            for channel in descriptor["channel_order"]:
                fields.append(np.concatenate([
                    _threehour_values(ds, channel, i, min(i + 8, start + 80),
                                      lat_slice=slice(0, 24), lon_slice=slice(0, 48))
                    for i in range(start, start + 80, 8)
                ]))
            inputs = np.stack(fields, axis=1)[:56]
            target = fields[0][56:, None]
            assert inputs.shape == (56, 5, 24, 48) and target.shape == (24, 1, 24, 48)
            assert inputs.dtype == target.dtype == np.float32
            assert np.isfinite(inputs).all() and np.isfinite(target).all()
    assert before == package_signature(root, data_file_name="earth_merra2_3hourly.nc")
    print("Read-only full package: descriptor available; 5848 steps; 5611 split-contained origins; source signatures unchanged")
