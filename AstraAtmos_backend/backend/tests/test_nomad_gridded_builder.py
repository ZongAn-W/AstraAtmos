import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scripts.build_nomad_gridded_dataset import (  # noqa: E402
    build_nomad_gridded_datasets,
    grid_nomad_dataframe,
    infer_nomad_mars_years,
)


def test_infer_nomad_mars_years_increments_on_ls_wrap():
    ls_values = np.array([348.0, 355.0, 2.0, 4.0, 120.0, 358.0, 1.0], dtype=np.float32)

    years = infer_nomad_mars_years(ls_values, start_my=34)

    assert years.tolist() == [34, 34, 35, 35, 35, 35, 36]


def test_grid_nomad_dataframe_keeps_sparse_observed_cells_only():
    frame = pd.DataFrame(
        {
            "o3_abund": [10.0, 14.0, 50.0, 90.0],
            "o3_abund_uncty": [1.0, 3.0, 5.0, 9.0],
            "ls": [350.2, 349.7, 2.1, 40.0],
            "longitude": [-179.6, -179.4, -174.9, 20.0],
            "latitude": [1.0, 1.2, -1.0, 60.0],
            "mars_year": [34, 34, 35, 35],
        }
    )
    target_ls = np.array([350.0, 2.0], dtype=np.float32)
    target_lat = np.array([2.5, -2.5], dtype=np.float32)
    target_lon = np.array([-180.0, -175.0], dtype=np.float32)

    gridded = grid_nomad_dataframe(
        frame,
        target_my=34,
        target_ls=target_ls,
        target_lat=target_lat,
        target_lon=target_lon,
        tolerance_ls=2.5,
    )

    assert gridded["o3col"].shape == (2, 2, 2)
    assert gridded["count"][0, 0, 0] == 2
    assert gridded["o3col"][0, 0, 0] == pytest.approx(12.0)
    assert gridded["uncertainty"][0, 0, 0] == pytest.approx(2.0)
    assert int(np.nansum(gridded["count"])) == 2
    assert np.isnan(gridded["o3col"][1, 1, 1])


def test_build_nomad_gridded_datasets_writes_year_files_and_manifest(tmp_path):
    csv_path = tmp_path / "nomad.csv"
    csv_path.write_text(
        "\n".join(
            [
                "o3_abund,o3_abund_uncty,ls,longitude,latitude,lst,sza,psurf,flag_lambertian",
                "10,1,350,-179.5,1,8,50,6,0",
                "20,2,2,-175,-1,9,55,7,1",
            ]
        ),
        encoding="utf-8",
    )
    overview_dir = tmp_path / "mcd_overview"
    overview_dir.mkdir()
    for year, ls_values in [(34, [350.0]), (35, [2.0])]:
        ds = xr.Dataset(
            data_vars={"Ls": (("time",), np.array(ls_values, dtype=np.float32))},
            coords={
                "time": np.arange(len(ls_values), dtype=np.int32),
                "lat": np.array([2.5, -2.5], dtype=np.float32),
                "lon": np.array([-180.0, -175.0], dtype=np.float32),
            },
        )
        ds.to_netcdf(overview_dir / f"MCD_MY{year}_overview.nc")
        ds.close()

    output_dir = tmp_path / "nomad"
    written = build_nomad_gridded_datasets(
        nomad_csv_path=csv_path,
        mcd_overview_dir=overview_dir,
        output_dir=output_dir,
        start_my=34,
        years=[34, 35],
        tolerance_ls=2.5,
    )

    assert [path.name for path in written] == [
        "NOMAD_ozone_MY34_gridded.nc",
        "NOMAD_ozone_MY35_gridded.nc",
    ]
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"] == "NOMAD UVIS ozone column retrieval dataset"
    assert manifest["start_my"] == 34
    assert manifest["years"] == [34, 35]
    assert manifest["files"]["34"] == "NOMAD_ozone_MY34_gridded.nc"
    assert manifest["files"]["35"] == "NOMAD_ozone_MY35_gridded.nc"
