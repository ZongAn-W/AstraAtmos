"""
Build the MCD-only overview dataset used by the data overview page.

The downloaded MCD reference files are the canonical source for the overview.
The legacy runtime builder is retained for explicit rebuilds, but the default
path keeps ozone and environmental variables from the same raw source.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
RAW_MCD_RELATIVE_DIR = Path("Data") / "MCD_Output_global_10m_ls_lst"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import MCD_DIR, MCD_OVERVIEW_DIR, SUPPORTED_MARS_YEARS  # noqa: E402
from core.data_align import unwrap_ls  # noqa: E402
from services.ozone_units import OZONE_COLUMN_UNIT, normalize_ozone_column_units  # noqa: E402

REFERENCE_FIELD_MAP = {
    "o3col": "O3COL",
    "U_Wind": "U",
    "V_Wind": "V",
    "Temperature": "T",
    "Pressure": "PS",
    "Solar_Flux_DN": "FSDS",
    "Ls": "LS",
}

BASE_FIELD_NAMES = [
    "Pressure",
    "Temperature",
    "U_Wind",
    "V_Wind",
    "Dust_Optical_Depth",
    "Solar_Flux_DN",
]

DIRECT_TARGET_LAT = np.arange(87.5, -90.0, -5.0, dtype=np.float32)
DIRECT_TARGET_LON = np.arange(-180.0, 180.0, 5.0, dtype=np.float32)
DIRECT_SAMPLES_PER_SOL = 8

DIRECT_REFERENCE_FIELD_MAP = {
    "o3col": "O3COL",
    "Pressure": "PS",
    "Temperature": "T",
    "U_Wind": "U",
    "V_Wind": "V",
    "Solar_Flux_DN": "FSDS",
}


def _daily_mean(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 4:
        return np.nanmean(arr, axis=1).astype(np.float32)
    if arr.ndim == 3:
        return arr.astype(np.float32)
    raise ValueError(f"Expected 3D or 4D field, got shape {arr.shape}")


def _trim_to_groups(values: np.ndarray, group_size: int) -> np.ndarray:
    arr = np.asarray(values)
    usable = (arr.shape[0] // group_size) * group_size
    if usable == 0:
        raise ValueError("Input field has no complete sample groups")
    return arr[:usable]


def _mean_by_sample_group(values: np.ndarray, group_size: int = DIRECT_SAMPLES_PER_SOL) -> np.ndarray:
    arr = _trim_to_groups(np.asarray(values, dtype=np.float32), group_size)
    grouped = arr.reshape((arr.shape[0] // group_size, group_size, *arr.shape[1:]))
    return np.nanmean(grouped, axis=1).astype(np.float32)


def _circular_mean_degrees(values: np.ndarray, group_size: int = DIRECT_SAMPLES_PER_SOL) -> np.ndarray:
    arr = _trim_to_groups(np.asarray(values, dtype=np.float64), group_size)
    grouped = arr.reshape((arr.shape[0] // group_size, group_size))
    radians = np.deg2rad(grouped)
    mean_sin = np.nanmean(np.sin(radians), axis=1)
    mean_cos = np.nanmean(np.cos(radians), axis=1)
    out = np.rad2deg(np.arctan2(mean_sin, mean_cos))
    return np.mod(out, 360.0).astype(np.float32)


def _sort_by_ls(data_vars: dict[str, np.ndarray], ls: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    order = np.argsort(np.asarray(ls, dtype=np.float32))
    sorted_vars = {name: np.asarray(values)[order] for name, values in data_vars.items()}
    return sorted_vars, np.asarray(ls, dtype=np.float32)[order]


def _regrid_latitude_safe(data: np.ndarray, source_lat: np.ndarray, target_lat: np.ndarray) -> np.ndarray:
    source = np.asarray(source_lat, dtype=np.float64)
    target = np.asarray(target_lat, dtype=np.float64)
    field = np.asarray(data, dtype=np.float64)
    order = np.argsort(source)
    source_sorted = source[order]
    field_sorted = field[:, order, :]
    regridded = np.empty((field.shape[0], target.shape[0], field.shape[2]), dtype=np.float32)
    for lon_idx in range(field.shape[2]):
        interpolator = interp1d(
            source_sorted,
            field_sorted[:, :, lon_idx],
            axis=1,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )
        regridded[:, :, lon_idx] = interpolator(target).astype(np.float32)
    return regridded


def _align_reference_ls(ref_ls: np.ndarray, target_ls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref_unwrapped = unwrap_ls(np.asarray(ref_ls, dtype=np.float64))
    target_unwrapped = unwrap_ls(np.asarray(target_ls, dtype=np.float64))

    while target_unwrapped[0] < ref_unwrapped[0] - 180:
        target_unwrapped = target_unwrapped + 360
    while target_unwrapped[0] > ref_unwrapped[-1] + 180:
        target_unwrapped = target_unwrapped - 360

    order = np.argsort(ref_unwrapped)
    return ref_unwrapped[order], target_unwrapped


def interpolate_reference_ozone(ref_o3: np.ndarray, ref_ls: np.ndarray, target_ls: np.ndarray) -> np.ndarray:
    source_ls, target_unwrapped = _align_reference_ls(ref_ls, target_ls)
    source_o3 = np.asarray(ref_o3, dtype=np.float64)[np.argsort(unwrap_ls(np.asarray(ref_ls, dtype=np.float64)))]
    interpolator = interp1d(
        source_ls,
        source_o3,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    return np.asarray(interpolator(target_unwrapped), dtype=np.float32)


def regrid_latitude(data: np.ndarray, source_lat: np.ndarray, target_lat: np.ndarray) -> np.ndarray:
    source_lat_arr = np.asarray(source_lat, dtype=np.float64)
    target_lat_arr = np.asarray(target_lat, dtype=np.float64)
    field = np.asarray(data, dtype=np.float64)

    if source_lat_arr[0] > source_lat_arr[-1]:
        source_sorted = source_lat_arr
        field_sorted = field
    else:
        source_sorted = source_lat_arr[::-1]
        field_sorted = field[:, ::-1, :]

    regridded = np.empty((field_sorted.shape[0], target_lat_arr.shape[0], field_sorted.shape[2]), dtype=np.float32)
    for lon_idx in range(field_sorted.shape[2]):
        interpolator = interp1d(
            source_sorted,
            field_sorted[:, :, lon_idx],
            axis=1,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=False,
        )
        regridded[:, :, lon_idx] = interpolator(target_lat_arr).astype(np.float32)
    return regridded


def build_overview_dataset(base_mcd_path: Path, ozone_ref_path: Path, output_path: Path) -> Path:
    base_ds = xr.open_dataset(base_mcd_path)
    ref_ds = xr.open_dataset(ozone_ref_path)
    try:
        target_ls = np.asarray(base_ds["Ls"].values, dtype=np.float32)
        target_lat = np.asarray(base_ds["lat"].values, dtype=np.float32)
        target_lon = np.asarray(base_ds["lon"].values, dtype=np.float32)

        ozone_by_ls = interpolate_reference_ozone(
            normalize_ozone_column_units(
                ref_ds["O3COL"].values,
                ref_ds["O3COL"].attrs.get("units"),
                allow_mcd_legacy_heuristic=True,
            ),
            np.asarray(ref_ds["LS"].values, dtype=np.float32),
            target_ls,
        )
        ozone_by_grid = regrid_latitude(ozone_by_ls, np.asarray(ref_ds["lat"].values), target_lat)

        data_vars = {
            "o3col": (("time", "lat", "lon"), ozone_by_grid.astype(np.float32)),
            "Ls": (("time",), target_ls.astype(np.float32)),
        }
        for field_name in BASE_FIELD_NAMES:
            if field_name not in base_ds:
                raise ValueError(f"Missing required MCD field: {field_name}")
            data_vars[field_name] = (("time", "lat", "lon"), _daily_mean(base_ds[field_name].values))

        out = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": np.arange(target_ls.shape[0], dtype=np.int32),
                "lat": target_lat.astype(np.float32),
                "lon": target_lon.astype(np.float32),
            },
            attrs={
                "source": "AstraAtmos derived MCD overview dataset",
                "ozone_source": str(ozone_ref_path),
                "base_mcd_source": str(base_mcd_path),
            },
        )
        out["o3col"].attrs["units"] = OZONE_COLUMN_UNIT

        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_netcdf(output_path)
        return output_path
    finally:
        base_ds.close()
        ref_ds.close()


def build_overview_from_reference_dataset(reference_path: Path, output_path: Path) -> Path:
    ref_ds = xr.open_dataset(reference_path, decode_times=False)
    try:
        if "LS" not in ref_ds:
            raise ValueError(f"{reference_path} missing required field: LS")
        source_lat = np.asarray(ref_ds["lat"].values, dtype=np.float32)
        target_lat = DIRECT_TARGET_LAT
        target_lon = np.asarray(ref_ds["lon"].values, dtype=np.float32)
        if target_lon.shape[0] != DIRECT_TARGET_LON.shape[0]:
            raise ValueError(f"{reference_path} has unsupported lon grid: {target_lon.shape}")

        ls_daily = _circular_mean_degrees(ref_ds["LS"].values)
        daily_vars: dict[str, np.ndarray] = {}
        for output_name, input_name in DIRECT_REFERENCE_FIELD_MAP.items():
            if input_name not in ref_ds:
                raise ValueError(f"{reference_path} missing required field: {input_name}")
            daily_field = _mean_by_sample_group(ref_ds[input_name].values)
            if output_name == "o3col":
                daily_field = normalize_ozone_column_units(
                    daily_field,
                    ref_ds[input_name].attrs.get("units"),
                    allow_mcd_legacy_heuristic=True,
                )
            daily_vars[output_name] = _regrid_latitude_safe(daily_field, source_lat, target_lat)

        # Downloaded MCD reference files do not include a dust field; keep it explicit and sparse.
        daily_vars["Dust_Optical_Depth"] = np.full_like(daily_vars["o3col"], np.nan, dtype=np.float32)
        daily_vars, ls_sorted = _sort_by_ls(daily_vars, ls_daily)

        data_vars = {
            "Ls": (("time",), ls_sorted.astype(np.float32)),
        }
        for name, values in daily_vars.items():
            data_vars[name] = (("time", "lat", "lon"), np.asarray(values, dtype=np.float32))

        out = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": np.arange(ls_sorted.shape[0], dtype=np.int32),
                "lat": target_lat.astype(np.float32),
                "lon": target_lon.astype(np.float32),
            },
            attrs={
                "source": "AstraAtmos direct MCD overview dataset",
                "build_mode": "reference_direct",
                "reference_mcd_source": str(reference_path),
                "dust_note": "Dust_Optical_Depth is NaN because the downloaded reference MCD file does not contain a dust field.",
            },
        )
        out["o3col"].attrs["units"] = OZONE_COLUMN_UNIT
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_netcdf(output_path)
        return output_path
    finally:
        ref_ds.close()


def _default_reference_dir() -> Path:
    candidates = [
        WORKSPACE_ROOT / RAW_MCD_RELATIVE_DIR,
        REPO_ROOT / RAW_MCD_RELATIVE_DIR,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def build_year(
    year: int,
    reference_dir: Path,
    output_dir: Path,
    base_mcd_dir: Path = MCD_DIR,
    mode: str = "auto",
) -> Path:
    base_mcd = base_mcd_dir / f"MCD_MY{year}_Lat-90-90_real.nc"
    ozone_ref = reference_dir / f"MCD_MY{year}_global_3h_5deg_10m_ls_lst.nc"
    output_path = output_dir / f"MCD_MY{year}_overview.nc"

    if not ozone_ref.is_file():
        raise FileNotFoundError(f"Missing reference MCD ozone file: {ozone_ref}")
    if mode not in {"auto", "runtime", "reference"}:
        raise ValueError("mode must be one of: auto, runtime, reference")
    if mode == "runtime":
        if base_mcd.is_file():
            return build_overview_dataset(base_mcd, ozone_ref, output_path)
        raise FileNotFoundError(f"Missing backend MCD file: {base_mcd}")

    return build_overview_from_reference_dataset(ozone_ref, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MCD overview NetCDF files.")
    parser.add_argument("--years", nargs="+", type=int, default=SUPPORTED_MARS_YEARS)
    parser.add_argument("--reference-dir", type=Path, default=_default_reference_dir())
    parser.add_argument("--output-dir", type=Path, default=MCD_OVERVIEW_DIR)
    parser.add_argument("--base-mcd-dir", type=Path, default=MCD_DIR)
    parser.add_argument("--mode", choices=["auto", "runtime", "reference"], default="reference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for year in args.years:
        output_path = build_year(year, args.reference_dir, args.output_dir, args.base_mcd_dir, args.mode)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
