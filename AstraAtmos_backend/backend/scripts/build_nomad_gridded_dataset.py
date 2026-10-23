from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import MCD_OVERVIEW_DIR  # noqa: E402

REQUIRED_NOMAD_COLUMNS = [
    "o3_abund",
    "o3_abund_uncty",
    "ls",
    "longitude",
    "latitude",
]


def _circular_delta(values: np.ndarray, target: np.ndarray | float) -> np.ndarray:
    return np.abs((np.asarray(values, dtype=np.float64) - target + 180.0) % 360.0 - 180.0)


def infer_nomad_mars_years(ls_values: np.ndarray, start_my: int) -> np.ndarray:
    if start_my is None:
        raise ValueError("start_my is required for NOMAD because the source file has no Mars Year column")
    ls_arr = np.asarray(ls_values, dtype=np.float64)
    if ls_arr.size == 0:
        return np.asarray([], dtype=np.int32)

    years = np.empty(ls_arr.shape[0], dtype=np.int32)
    current_year = int(start_my)
    years[0] = current_year
    for idx in range(1, ls_arr.shape[0]):
        if ls_arr[idx] - ls_arr[idx - 1] < -180.0:
            current_year += 1
        years[idx] = current_year
    return years


def nearest_lat_indices(values: np.ndarray, target_lat: np.ndarray) -> np.ndarray:
    lat = np.asarray(values, dtype=np.float64)
    centers = np.asarray(target_lat, dtype=np.float64)
    return np.abs(lat[:, None] - centers[None, :]).argmin(axis=1).astype(np.int64)


def nearest_lon_indices(values: np.ndarray, target_lon: np.ndarray) -> np.ndarray:
    lon = ((np.asarray(values, dtype=np.float64) + 180.0) % 360.0) - 180.0
    centers = np.asarray(target_lon, dtype=np.float64)
    delta = _circular_delta(lon[:, None], centers[None, :])
    return delta.argmin(axis=1).astype(np.int64)


def nearest_time_indices(values: np.ndarray, target_ls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ls = np.asarray(values, dtype=np.float64)
    centers = np.asarray(target_ls, dtype=np.float64)
    order = np.argsort(centers)
    sorted_centers = centers[order]
    insert_at = np.searchsorted(sorted_centers, ls)
    candidate_positions = np.vstack(
        [
            np.clip(insert_at - 1, 0, len(sorted_centers) - 1),
            np.clip(insert_at, 0, len(sorted_centers) - 1),
            np.zeros_like(insert_at),
            np.full_like(insert_at, len(sorted_centers) - 1),
        ]
    )
    candidate_indices = order[candidate_positions]
    candidate_values = centers[candidate_indices]
    deltas = _circular_delta(ls[None, :], candidate_values)
    best = np.argmin(deltas, axis=0)
    row = np.arange(ls.shape[0])
    return candidate_indices[best, row].astype(np.int64), deltas[best, row]


def grid_nomad_dataframe(
    frame: pd.DataFrame,
    target_my: int,
    target_ls: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    tolerance_ls: float,
) -> dict[str, np.ndarray]:
    year_frame = frame.loc[frame["mars_year"] == int(target_my)].copy()
    shape = (len(target_ls), len(target_lat), len(target_lon))
    sum_o3 = np.zeros(shape, dtype=np.float64)
    sum_uncertainty = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.int32)

    if year_frame.empty:
        return {
            "o3col": np.full(shape, np.nan, dtype=np.float32),
            "count": count,
            "uncertainty": np.full(shape, np.nan, dtype=np.float32),
        }

    time_idx, time_distance = nearest_time_indices(year_frame["ls"].to_numpy(), target_ls)
    keep = time_distance <= float(tolerance_ls)
    if not np.any(keep):
        return {
            "o3col": np.full(shape, np.nan, dtype=np.float32),
            "count": count,
            "uncertainty": np.full(shape, np.nan, dtype=np.float32),
        }

    kept = year_frame.loc[keep]
    kept_time = time_idx[keep]
    kept_lat = nearest_lat_indices(kept["latitude"].to_numpy(), target_lat)
    kept_lon = nearest_lon_indices(kept["longitude"].to_numpy(), target_lon)
    o3 = kept["o3_abund"].to_numpy(dtype=np.float64)
    uncertainty = kept["o3_abund_uncty"].to_numpy(dtype=np.float64)

    np.add.at(sum_o3, (kept_time, kept_lat, kept_lon), o3)
    np.add.at(sum_uncertainty, (kept_time, kept_lat, kept_lon), uncertainty)
    np.add.at(count, (kept_time, kept_lat, kept_lon), 1)

    with np.errstate(invalid="ignore", divide="ignore"):
        o3_mean = sum_o3 / count
        uncertainty_mean = sum_uncertainty / count
    o3_mean[count == 0] = np.nan
    uncertainty_mean[count == 0] = np.nan
    return {
        "o3col": o3_mean.astype(np.float32),
        "count": count.astype(np.int32),
        "uncertainty": uncertainty_mean.astype(np.float32),
    }


def read_nomad_csv(nomad_csv_path: Path, start_my: int) -> pd.DataFrame:
    frame = pd.read_csv(nomad_csv_path)
    missing = [name for name in REQUIRED_NOMAD_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"{nomad_csv_path} missing required columns: {missing}")
    frame = frame.dropna(subset=REQUIRED_NOMAD_COLUMNS).copy()
    frame["mars_year"] = infer_nomad_mars_years(frame["ls"].to_numpy(), start_my)
    return frame


def _year_from_overview_path(path: Path) -> int | None:
    match = re.search(r"MY(\d+)", path.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def load_overview_axis(mcd_overview_dir: Path, year: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matches = sorted(mcd_overview_dir.glob(f"*MY{year}*overview*.nc"))
    if not matches:
        raise FileNotFoundError(f"Missing MCD overview file for MY{year}: {mcd_overview_dir}")
    with xr.open_dataset(matches[0], decode_times=False) as ds:
        return (
            np.asarray(ds["Ls"].values, dtype=np.float32),
            np.asarray(ds["lat"].values, dtype=np.float32),
            np.asarray(ds["lon"].values, dtype=np.float32),
        )


def discover_overview_years(mcd_overview_dir: Path) -> list[int]:
    years = []
    for path in sorted(mcd_overview_dir.glob("*overview*.nc")):
        year = _year_from_overview_path(path)
        if year is not None:
            years.append(year)
    return sorted(set(years))


def write_nomad_year_dataset(
    output_path: Path,
    year: int,
    target_ls: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    gridded: dict[str, np.ndarray],
    source_path: Path,
    tolerance_ls: float,
) -> Path:
    ds = xr.Dataset(
        data_vars={
            "Ls": (("time",), np.asarray(target_ls, dtype=np.float32)),
            "o3col": (("time", "lat", "lon"), gridded["o3col"].astype(np.float32)),
            "count": (("time", "lat", "lon"), gridded["count"].astype(np.int32)),
            "uncertainty": (("time", "lat", "lon"), gridded["uncertainty"].astype(np.float32)),
        },
        coords={
            "time": np.arange(len(target_ls), dtype=np.int32),
            "lat": np.asarray(target_lat, dtype=np.float32),
            "lon": np.asarray(target_lon, dtype=np.float32),
        },
        attrs={
            "source": "NOMAD UVIS ozone column retrieval dataset",
            "source_file": str(source_path),
            "mars_year": int(year),
            "tolerance_ls": float(tolerance_ls),
            "aggregation": "Sparse bin mean on MCD overview time/lat/lon grid",
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)
    ds.close()
    return output_path


def build_nomad_gridded_datasets(
    nomad_csv_path: Path,
    mcd_overview_dir: Path,
    output_dir: Path,
    start_my: int,
    years: list[int] | None = None,
    tolerance_ls: float = 2.5,
) -> list[Path]:
    if start_my is None:
        raise ValueError("--nomad-start-my is required")
    frame = read_nomad_csv(nomad_csv_path, start_my)
    available_nomad_years = sorted(int(year) for year in frame["mars_year"].dropna().unique())
    candidate_years = sorted(years if years is not None else discover_overview_years(mcd_overview_dir))
    target_years = [year for year in candidate_years if year in available_nomad_years]
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    manifest_files: dict[str, str] = {}

    for year in target_years:
        target_ls, target_lat, target_lon = load_overview_axis(mcd_overview_dir, year)
        gridded = grid_nomad_dataframe(frame, year, target_ls, target_lat, target_lon, tolerance_ls)
        if not np.any(gridded["count"] > 0):
            continue
        output_path = output_dir / f"NOMAD_ozone_MY{year}_gridded.nc"
        write_nomad_year_dataset(output_path, year, target_ls, target_lat, target_lon, gridded, nomad_csv_path, tolerance_ls)
        written.append(output_path)
        manifest_files[str(year)] = output_path.name

    manifest = {
        "source": "NOMAD UVIS ozone column retrieval dataset",
        "source_file": str(nomad_csv_path),
        "start_my": int(start_my),
        "years": [int(path.stem.split("_MY")[1].split("_")[0]) for path in written],
        "tolerance_ls": float(tolerance_ls),
        "files": manifest_files,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sparse gridded NOMAD ozone NetCDF files.")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "Data" / "NOMAD" / "UVIS_ozone_column_retrieval_dataset_v1.txt",
    )
    parser.add_argument("--mcd-overview-dir", type=Path, default=MCD_OVERVIEW_DIR)
    parser.add_argument("--output-dir", type=Path, default=BACKEND_DIR / "data" / "nomad")
    parser.add_argument("--start-my", type=int, required=True)
    parser.add_argument("--years", nargs="*", type=int, default=None)
    parser.add_argument("--tolerance-ls", type=float, default=2.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = build_nomad_gridded_datasets(
        nomad_csv_path=args.input,
        mcd_overview_dir=args.mcd_overview_dir,
        output_dir=args.output_dir,
        start_my=args.start_my,
        years=args.years,
        tolerance_ls=args.tolerance_ls,
    )
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
