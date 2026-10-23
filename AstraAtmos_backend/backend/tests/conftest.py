"""Shared fixtures for backend tests.

Fixtures here are opt-in only: nothing in this module runs automatically, so
importing heavyweight data dependencies stays scoped to the tests that ask for
them.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def _relax_private_temp_dirs() -> None:
    """Keep ``tempfile.mkdtemp`` directories writable inside this workspace.

    ``tempfile`` creates its directories with POSIX mode ``0o700``; on Windows
    that mode is applied as a private ACL, and the harness filesystem policy
    denies every later write *inside* such a directory. That would break the
    built-in ``tmp_path`` fixture (and ``--basetemp``) for every test here, so
    the mode is widened to ``0o777`` before ``os.mkdir`` runs. File system
    semantics are otherwise untouched: the directories still live in the
    session temporary root, and only the POSIX mode argument differs.
    """
    if os.name != "nt" or getattr(os, "_astraatmos_tempdir_relaxed", False):
        return
    original_mkdir = os.mkdir

    def mkdir(path, mode=0o777, *args, **kwargs):
        if mode == 0o700:
            mode = 0o777
        return original_mkdir(path, mode, *args, **kwargs)

    os.mkdir = mkdir
    os._astraatmos_tempdir_relaxed = True


_relax_private_temp_dirs()


@pytest.fixture
def earth_release(tmp_path):
    """Build a temporary Earth compact release with the real dimensions.

    Covers the full 731 day / 31 x 49 grid of the production package so registry
    and route contracts can be exercised without the multi-megabyte local file.
    """
    import hashlib
    import json
    import numpy as np
    from scripts.build_earth_ozone_dataset import build_dataset

    dates = np.arange("2020-01-01", "2022-01-01", dtype="datetime64[D]")
    lat = np.arange(-60, 61, 4, dtype=np.float32)
    lon = np.arange(-120, 121, 5, dtype=np.float32)
    cube = np.broadcast_to(
        np.arange(len(dates), dtype=np.float32)[:, None, None],
        (len(dates), len(lat), len(lon)),
    )
    source = tmp_path / "source.npz"
    np.savez_compressed(
        source, **{key: cube + offset for offset, key in enumerate(("O3", "U", "V", "T", "S"))},
        lat=lat, lon=lon, time_iso=dates.astype(str),
        metadata=json.dumps({
            "source": "MERRA2", "temporal": "daily", "lat_stride": 8, "lon_stride": 8,
            "used_variables": {"O3": ["TO3"], "U": ["U10M"], "V": ["V10M"], "T": ["T2M"], "S": ["SWGDN"]},
        }),
    )
    path = build_dataset(source, tmp_path / "earth")
    return {
        "earth_package_dir": path.parent,
        "expected_manifest_sha256": hashlib.sha256(path.with_name("manifest.json").read_bytes()).hexdigest(),
        "expected_data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


V2_DATASET_ID = "earth_merra2_daily_v2"
V2_LATITUDE = np.arange(-87.5, 90.0, 5.0)
V2_LONGITUDE = np.arange(-177.5, 180.0, 5.0)
V2_TRAIN_END = "2020-12-31"
V2_VALIDATION_END = "2021-06-30"
V2_SOURCE_SHA256 = "f" * 64


def _v2_synthetic_fields(dates):
    """Physically plausible fields varying along both spatial axes and time.

    ``TO3`` follows the documented seasonal shape; every other channel is a
    distinct, bounded periodic surface so a transposed, flipped or mis-indexed
    read differs visibly. Offsets are per channel, so cross-channel confusion
    also shows up.
    """
    day = np.arange(len(dates), dtype="float64")[:, None, None]
    lat = np.deg2rad(V2_LATITUDE)[None, :, None]
    lon = np.deg2rad(V2_LONGITUDE)[None, None, :]
    seasonal = np.sin(2.0 * np.pi * day / 366.0)
    return {
        "TO3": 300.0 + 40.0 * seasonal + 10.0 * np.cos(lat) + 5.0 * np.sin(lon),
        "U10M": 2.0 + 8.0 * seasonal + 6.0 * np.cos(lat) * np.sin(lon),
        "V10M": -3.0 + 7.0 * seasonal + 6.0 * np.sin(lat) * np.cos(lon),
        "T2M": 270.0 + 12.0 * seasonal + 10.0 * np.cos(lat) + 4.0 * np.sin(lon),
        "SWGDN": 160.0 + 20.0 * seasonal + 20.0 * np.cos(lat) + 10.0 * np.sin(lon),
    }


@pytest.fixture
def earth_global_release(tmp_path):
    """Temporary contract-valid v2 **global** release written straight to NetCDF.

    Unlike :func:`earth_release` this never goes through
    ``build_earth_ozone_dataset.build_dataset``, which only produces the v1
    regional form. It mirrors what ``scripts/preprocess_earth_reanalysis.py``
    writes: 36 global 5 degree latitude cells, 72 longitude cells, explicit
    ``lat_bounds``/``lon_bounds`` cell edges, a ``split`` flag variable and a
    manifest carrying every field ``earth_dataset_metadata`` cross checks.
    """
    import hashlib
    import json

    import xarray as xr

    from services.earth_dataset import CHANNELS, SCHEMA, SPLITS, UNITS, split_days

    dates = np.arange("2020-01-01", "2022-01-01", dtype="datetime64[D]")
    assert len(dates) == 731
    split = split_days(dates, V2_TRAIN_END, V2_VALIDATION_END)

    lat_edges = np.column_stack((
        np.arange(-90.0, 90.0, 5.0), np.arange(-85.0, 95.0, 5.0),
    ))
    lon_edges = np.column_stack((
        np.arange(-180.0, 180.0, 5.0), np.arange(-175.0, 185.0, 5.0),
    ))
    assert lat_edges.shape == (36, 2) and lon_edges.shape == (72, 2)
    assert np.allclose(lat_edges.mean(axis=1), V2_LATITUDE)
    assert np.allclose(lon_edges.mean(axis=1), V2_LONGITUDE)

    fields = _v2_synthetic_fields(dates)
    for name, values in fields.items():
        assert np.isfinite(values).all()
    assert 200.0 <= fields["TO3"].min() and fields["TO3"].max() <= 450.0
    assert 200.0 <= fields["T2M"].min() and fields["T2M"].max() <= 320.0
    assert 0.0 <= fields["SWGDN"].min() and fields["SWGDN"].max() <= 400.0
    for name in ("U10M", "V10M"):
        assert -30.0 <= fields[name].min() and fields[name].max() <= 30.0

    package_dir = tmp_path / "merra2_daily_v2"
    package_dir.mkdir(parents=True, exist_ok=True)
    data_path = package_dir / "earth_merra2_daily.nc"

    train_mask = split == SPLITS["train"]
    # Published float32 storage, exactly as the writer stores it. The verifier
    # reads these float32 values back and reduces them in float64, so the
    # statistics must be computed from the same stored array - reducing the
    # float64 originals would differ by more than the rtol/atol 1e-12 window.
    stored = {name: fields[name].astype("float32") for name in CHANNELS}
    train_stats = {}
    for name in CHANNELS:
        training = np.asarray(stored[name], dtype="float64")[train_mask]
        std = float(training.std())
        train_stats[name] = {
            "mean": float(training.mean()),
            "std": std,
            "scale_std": std if std >= 1e-6 else 1.0,
        }

    dataset = xr.Dataset(
        {
            name: (
                ("time", "lat", "lon"),
                stored[name],
                {"units": unit, "long_name": name},
            )
            for name, unit in zip(CHANNELS, UNITS)
        },
        coords={
            "time": ("time", dates.astype("datetime64[ns]")),
            "lat": ("lat", V2_LATITUDE.astype("float32"), {
                "units": "degrees_north", "bounds": "lat_bounds",
            }),
            "lon": ("lon", V2_LONGITUDE.astype("float32"), {
                "units": "degrees_east", "bounds": "lon_bounds",
            }),
        },
        attrs={
            "planet": "Earth",
            "schema": SCHEMA,
            "source": "NASA MERRA-2",
            "source_product": "MERRA-2 tavg1_2d_slv_Nx + tavg1_2d_rad_Nx",
            "source_sha256": V2_SOURCE_SHA256,
            "temporal_resolution": "1 day",
            "train_end": V2_TRAIN_END,
            "validation_end": V2_VALIDATION_END,
            "processing": "synthetic test package; no source archive",
            "spatial_method": "spherical_area_weighted_overlap",
            "temporal_method": "mean_of_complete_source_steps",
        },
    )
    dataset["lat_bounds"] = (("lat", "bounds"), lat_edges)
    dataset["lon_bounds"] = (("lon", "bounds"), lon_edges)
    dataset["split"] = ("time", split, {
        "flag_values": np.array([0, 1, 2], dtype="int8"),
        "flag_meanings": "train validation test",
    })
    encoding = {
        name: {"dtype": "float32", "zlib": True, "complevel": 1, "shuffle": True}
        for name in CHANNELS
    }
    dataset.to_netcdf(data_path, engine="netcdf4", encoding=encoding)
    dataset.close()

    manifest = {
        "schema": SCHEMA,
        "planet": "Earth",
        "dataset_id": V2_DATASET_ID,
        "dataset_version": "v2",
        "source": {
            "product": "MERRA-2",
            "root_label": "synthetic",
            "date_start": str(dates[0]),
            "date_end": str(dates[-1]),
            "daily_file_count": int(len(dates)),
            "source_sha256": V2_SOURCE_SHA256,
        },
        "source_sha256": V2_SOURCE_SHA256,
        "source_fingerprint_basis": "synthetic test package; not a real inventory digest",
        "data_file": data_path.name,
        "dimensions": {"time": 731, "lat": 36, "lon": 72, "bounds": 2},
        "time_start": str(dates[0]),
        "time_end": str(dates[-1]),
        "cadence": "daily mean",
        "latitude_range": [float(lat_edges[0, 0]), float(lat_edges[-1, 1])],
        "longitude_range": [float(lon_edges[0, 0]), float(lon_edges[-1, 1])],
        "cell_bounds": {
            "latitude": [float(lat_edges[0, 0]), float(lat_edges[-1, 1])],
            "longitude": [float(lon_edges[0, 0]), float(lon_edges[-1, 1])],
        },
        "channel_order": list(CHANNELS),
        "variables": {},
        "splits": {},
        "normalization": {
            "method": "per_channel_train_mean_std",
            "fit_split": "train",
            "reduction": "unweighted time/latitude/longitude population statistics (ddof=0)",
            "minimum_std": 1e-6,
            "train_stats": train_stats,
        },
        "processing": {
            "temporal_method": "mean_of_complete_source_steps",
            "spatial_method": "spherical_area_weighted_overlap",
            "target_grid": "global 5x5 degree cells",
        },
        "limitations": [
            "Daily means do not retain hourly variation",
            "Synthetic test package written by the test suite",
        ],
    }
    for name, unit in zip(CHANNELS, UNITS):
        manifest["variables"][name] = {
            "units": unit,
            "dtype": "float32",
            "min": float(stored[name].min()),
            "max": float(stored[name].max()),
        }
    for name, code in SPLITS.items():
        selected = dates[split == code]
        manifest["splits"][name] = {
            "start": str(selected[0]), "end": str(selected[-1]), "days": int(len(selected)),
        }
    manifest["data_bytes"] = data_path.stat().st_size
    manifest["data_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    manifest_path = package_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "earth_package_dir": package_dir,
        "expected_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "expected_data_sha256": manifest["data_sha256"],
    }


@pytest.fixture
def earth_spatial_release(tmp_path):
    """Temporary Earth release whose fields vary along both spatial axes.

    Values are ``1000 * t + 10 * row + col`` plus a per-variable offset, so a
    transposed, flipped or mis-indexed read differs visibly instead of happening
    to match. Dimensions, dates, units and splits follow the same protocol as the
    production package (731 days, 31 x 49).
    """
    import hashlib
    import json
    import numpy as np
    from scripts.build_earth_ozone_dataset import build_dataset

    dates = np.arange("2020-01-01", "2022-01-01", dtype="datetime64[D]")
    lat = np.arange(-60, 61, 4, dtype=np.float32)
    lon = np.arange(-120, 121, 5, dtype=np.float32)
    base = (
        1000.0 * np.arange(len(dates), dtype=np.float32)[:, None, None]
        + 10.0 * np.arange(len(lat), dtype=np.float32)[None, :, None]
        + np.arange(len(lon), dtype=np.float32)[None, None, :]
    )
    source = tmp_path / "spatial_source.npz"
    np.savez_compressed(
        source,
        **{key: base + offset for offset, key in enumerate(("O3", "U", "V", "T", "S"))},
        lat=lat, lon=lon, time_iso=dates.astype(str),
        metadata=json.dumps({
            "source": "MERRA2", "temporal": "daily", "lat_stride": 8, "lon_stride": 8,
            "used_variables": {"O3": ["TO3"], "U": ["U10M"], "V": ["V10M"], "T": ["T2M"], "S": ["SWGDN"]},
        }),
    )
    path = build_dataset(source, tmp_path / "earth_spatial")
    return {
        "earth_package_dir": path.parent,
        "expected_manifest_sha256": hashlib.sha256(path.with_name("manifest.json").read_bytes()).hexdigest(),
        "expected_data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
