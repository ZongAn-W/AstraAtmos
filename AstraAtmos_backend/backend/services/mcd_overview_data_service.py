"""
MCD-only data adapter for the data overview page.

This class deliberately mimics the subset of DataService consumed by
AnalysisService. Its "openmars" view is the MCD overview ozone field, which
lets existing analysis code treat o3col as the main target without changing
legacy /explore behavior elsewhere.
"""

from __future__ import annotations

import glob
import logging
import re
from pathlib import Path

import numpy as np
import xarray as xr

from config import (
    MCD_OVERVIEW_DIR,
    MCD_RAW_3H_DIR,
    NOMAD_DIR,
    NOMAD_MATCH_TOLERANCE_LS,
    OVERVIEW_OZONE_MATCH_TOLERANCE_LS,
    SUPPORTED_MARS_YEARS,
)
from services.data_service import DataService
from services.ozone_units import normalize_ozone_column_units

logger = logging.getLogger("astraatmos.mcd_overview")
MAX_COVERAGE_GAP_LS = 30.0

OVERVIEW_ENV_FIELDS = [
    "Temperature",
    "U_Wind",
    "V_Wind",
    "Dust_Optical_Depth",
    "Solar_Flux_DN",
]


class McdOverviewDataService:
    def __init__(
        self,
        base_data_service: DataService,
        overview_dir: Path = MCD_OVERVIEW_DIR,
        nomad_dir: Path = NOMAD_DIR,
        raw_3h_dir: Path = MCD_RAW_3H_DIR,
        supported_years: list[int] | None = None,
    ):
        self.base = base_data_service
        self.overview_dir = Path(overview_dir)
        self.nomad_dir = Path(nomad_dir)
        self.raw_3h_dir = Path(raw_3h_dir)
        self.supported_years = supported_years
        self.overview: dict[int, dict] = {}
        self.nomad: dict[int, dict] = {}
        self.raw_hourly_mcd: dict[int, dict | None] = {}
        self._load_all()

    def _load_all(self) -> None:
        years = self.supported_years if self.supported_years is not None else self._discover_overview_years()
        if not years:
            years = SUPPORTED_MARS_YEARS
        for mars_year in years:
            self._load_year(mars_year)
        self._load_nomad_all()
        logger.info("MCD overview data loaded for years: %s", sorted(self.overview.keys()))
        logger.info("NOMAD overview data loaded for years: %s", sorted(self.nomad.keys()))

    @staticmethod
    def _year_from_path(path: Path) -> int | None:
        match = re.search(r"MY(\d+)", path.name, flags=re.IGNORECASE)
        return int(match.group(1)) if match else None

    def _discover_overview_years(self) -> list[int]:
        years = []
        for path in sorted(self.overview_dir.glob("*overview*.nc")):
            year = self._year_from_path(path)
            if year is not None:
                years.append(year)
        return sorted(set(years))

    def _load_year(self, mars_year: int) -> None:
        pattern = str(self.overview_dir / f"*MY{mars_year}*overview*.nc")
        files = sorted(glob.glob(pattern))
        if not files:
            logger.warning("MCD overview file not found for MY%s: %s", mars_year, pattern)
            return

        file_path = Path(files[0])
        with xr.open_dataset(file_path, decode_times=False) as ds:
            required = ["o3col", "Ls", "lat", "lon", *OVERVIEW_ENV_FIELDS]
            missing = [name for name in required if name not in ds]
            if missing:
                raise ValueError(f"{file_path} missing required fields: {missing}")

            data = {
                "o3col": normalize_ozone_column_units(
                    ds["o3col"].values,
                    ds["o3col"].attrs.get("units"),
                    allow_mcd_legacy_heuristic=True,
                ),
                "ls": np.asarray(ds["Ls"].values, dtype=np.float32),
                "lat": np.asarray(ds["lat"].values, dtype=np.float32),
                "lon": np.asarray(ds["lon"].values, dtype=np.float32),
                "source_file": str(file_path),
            }
            for field_name in OVERVIEW_ENV_FIELDS:
                data[field_name] = np.asarray(ds[field_name].values, dtype=np.float32)

        self.overview[mars_year] = data

    def _load_nomad_all(self) -> None:
        if not self.nomad_dir.is_dir():
            logger.info("NOMAD directory not found, skip: %s", self.nomad_dir)
            return
        for file_path in sorted(self.nomad_dir.glob("*MY*_gridded.nc")):
            year = self._year_from_path(file_path)
            if year is None:
                logger.warning("Skip NOMAD file without MY marker: %s", file_path)
                continue
            self._load_nomad_year(year, file_path)

    def _load_nomad_year(self, mars_year: int, file_path: Path) -> None:
        with xr.open_dataset(file_path, decode_times=False) as ds:
            required = ["o3col", "Ls", "lat", "lon", "count"]
            missing = [name for name in required if name not in ds]
            if missing:
                raise ValueError(f"{file_path} missing required fields: {missing}")
            self.nomad[mars_year] = {
                "o3col": np.asarray(ds["o3col"].values, dtype=np.float32),
                "ls": np.asarray(ds["Ls"].values, dtype=np.float32),
                "lat": np.asarray(ds["lat"].values, dtype=np.float32),
                "lon": np.asarray(ds["lon"].values, dtype=np.float32),
                "count": np.asarray(ds["count"].values, dtype=np.int32),
                "source_file": str(file_path),
            }

    def _require_year(self, mars_year: int) -> dict:
        if mars_year not in self.overview:
            raise ValueError(f"MY{mars_year} MCD overview data is not loaded")
        return self.overview[mars_year]

    def get_openmars_data(self, mars_year: int) -> dict:
        year = self._require_year(mars_year)
        return {
            "o3col": year["o3col"],
            "ls": year["ls"],
            "lat": year["lat"],
            "lon": year["lon"],
        }

    def get_aligned_mcd_data(self, mars_year: int) -> dict:
        year = self._require_year(mars_year)
        aligned = {
            "ls": year["ls"],
            "lat": year["lat"],
            "lon": year["lon"],
        }
        for field_name in OVERVIEW_ENV_FIELDS:
            aligned[field_name] = year[field_name]
        return aligned

    def _load_raw_3h_ozone(self, mars_year: int) -> dict | None:
        if mars_year in self.raw_hourly_mcd:
            return self.raw_hourly_mcd[mars_year]
        if not self.raw_3h_dir.is_dir():
            self.raw_hourly_mcd[mars_year] = None
            return None

        files = sorted(self.raw_3h_dir.glob(f"*MY{mars_year}*3h*.nc"))
        if not files:
            files = sorted(self.raw_3h_dir.glob(f"*MY{mars_year}*.nc"))
        if not files:
            self.raw_hourly_mcd[mars_year] = None
            return None

        file_path = files[0]
        try:
            with xr.open_dataset(file_path, decode_times=False) as ds:
                ozone_name = next((name for name in ("O3COL", "o3col") if name in ds), None)
                ls_name = next((name for name in ("LS", "Ls", "ls") if name in ds), None)
                if ozone_name is None or ls_name is None or "lat" not in ds or "lon" not in ds:
                    logger.warning("Raw 3h MCD file missing ozone/LS/lat/lon fields: %s", file_path)
                    self.raw_hourly_mcd[mars_year] = None
                    return None

                ozone = normalize_ozone_column_units(
                    ds[ozone_name].values,
                    ds[ozone_name].attrs.get("units"),
                    allow_mcd_legacy_heuristic=True,
                )
                ls_values = np.asarray(ds[ls_name].values, dtype=np.float32)
                if ozone.ndim != 3 or ls_values.ndim != 1:
                    logger.warning("Raw 3h MCD file has unsupported dimensions: %s", file_path)
                    self.raw_hourly_mcd[mars_year] = None
                    return None

                steps_per_sol = 8
                sol_count = min(ozone.shape[0], ls_values.shape[0]) // steps_per_sol
                if sol_count < 1:
                    self.raw_hourly_mcd[mars_year] = None
                    return None

                trim = sol_count * steps_per_sol
                hourly = ozone[:trim].reshape(sol_count, steps_per_sol, ozone.shape[1], ozone.shape[2])
                ls_sol = np.nanmean(ls_values[:trim].reshape(sol_count, steps_per_sol), axis=1).astype(np.float32)
                loaded = {
                    "O3COL_hourly": hourly,
                    "ls": ls_sol,
                    "lat": np.asarray(ds["lat"].values, dtype=np.float32),
                    "lon": np.asarray(ds["lon"].values, dtype=np.float32),
                    "raw_3h_source_file": str(file_path),
                }
                self.raw_hourly_mcd[mars_year] = loaded
                return loaded
        except Exception as exc:
            logger.warning("Failed to load raw 3h MCD ozone for MY%s from %s: %s", mars_year, file_path, exc)
            self.raw_hourly_mcd[mars_year] = None
            return None

    def get_mcd_data(self, mars_year: int) -> dict:
        try:
            out = dict(self.base.get_mcd_data(mars_year))
        except ValueError:
            year = self._require_year(mars_year)
            out = {
                "ls": year["ls"],
                "lat": year["lat"],
                "lon": year["lon"],
            }
            for field_name in OVERVIEW_ENV_FIELDS:
                if field_name in year:
                    out[field_name] = year[field_name]

        raw_hourly = self._load_raw_3h_ozone(mars_year)
        if raw_hourly is not None:
            out.update(raw_hourly)
        return out

    def get_available_years(self) -> list[int]:
        return sorted(self.overview.keys())

    def get_ls_range(self, mars_year: int) -> tuple[float, float]:
        year = self._require_year(mars_year)
        return float(year["ls"][0]), float(year["ls"][-1])

    @staticmethod
    def get_nearest_ls_index(ls_array: np.ndarray, target_ls: float) -> int:
        return int(np.argmin(np.abs(np.asarray(ls_array, dtype=float) - float(target_ls))))

    def _points_from_field(self, field: np.ndarray, lat_arr: np.ndarray, lon_arr: np.ndarray) -> list[dict]:
        points = []
        for i, lat in enumerate(lat_arr):
            for j, lon in enumerate(lon_arr):
                val = float(field[i, j])
                if np.isfinite(val):
                    lon_value = float(lon) if float(lon) <= 180 else float(lon) - 360
                    points.append({"lat": float(lat), "lng": lon_value, "val": val})
        return points

    def _build_layer_from_source(self, source_data: dict, ls: float, source: str) -> dict:
        idx = self.get_nearest_ls_index(source_data["ls"], ls)
        field = np.asarray(source_data["o3col"][idx], dtype=np.float32)
        valid = field[np.isfinite(field)]
        return {
            "source": source,
            "points": self._points_from_field(field, source_data["lat"], source_data["lon"]),
            "minVal": float(np.nanmin(valid)) if valid.size else 0.0,
            "maxVal": float(np.nanmax(valid)) if valid.size else 1.0,
            "ls": float(source_data["ls"][idx]),
        }

    def _match_openmars_layer(self, mars_year: int, anchor_ls: float) -> dict | None:
        try:
            openmars = self.base.get_openmars_data(mars_year)
        except ValueError:
            return None

        idx = self.get_nearest_ls_index(openmars["ls"], anchor_ls)
        matched_ls = float(openmars["ls"][idx])
        if abs(matched_ls - float(anchor_ls)) > OVERVIEW_OZONE_MATCH_TOLERANCE_LS:
            return None

        return self._build_layer_from_source(openmars, matched_ls, "openmars")

    def _match_nomad_layer(self, mars_year: int, anchor_ls: float) -> dict | None:
        nomad = self.nomad.get(mars_year)
        if nomad is None:
            return None

        idx = self.get_nearest_ls_index(nomad["ls"], anchor_ls)
        matched_ls = float(nomad["ls"][idx])
        if abs(matched_ls - float(anchor_ls)) > NOMAD_MATCH_TOLERANCE_LS:
            return None

        field = np.asarray(nomad["o3col"][idx], dtype=np.float32)
        count = np.asarray(nomad["count"][idx], dtype=np.int32)
        if not np.any((count > 0) & np.isfinite(field)):
            return None

        layer = self._build_layer_from_source(
            {
                "o3col": nomad["o3col"],
                "ls": nomad["ls"],
                "lat": nomad["lat"],
                "lon": nomad["lon"],
            },
            matched_ls,
            "nomad",
        )
        count_by_key = {}
        for i, lat in enumerate(nomad["lat"]):
            for j, lon in enumerate(nomad["lon"]):
                lon_value = float(lon) if float(lon) <= 180 else float(lon) - 360
                count_by_key[(round(float(lat), 3), round(lon_value, 3))] = int(count[i, j])
        for point in layer.get("points", []):
            key = (round(float(point["lat"]), 3), round(float(point["lng"]), 3))
            point["count"] = count_by_key.get(key, 1)
        return layer if layer["points"] else None

    @staticmethod
    def _safe_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
        if a.size < 2 or b.size < 2:
            return None
        if float(np.nanstd(a)) == 0.0 or float(np.nanstd(b)) == 0.0:
            return None
        corr = float(np.corrcoef(a, b)[0, 1])
        return corr if np.isfinite(corr) else None

    def _build_nomad_validation(self, mcd_layer: dict, nomad_layer: dict | None) -> dict | None:
        if not nomad_layer:
            return None

        mcd_by_key = {
            (round(float(point["lat"]), 3), round(float(point["lng"]), 3)): float(point["val"])
            for point in mcd_layer.get("points", [])
            if np.isfinite(float(point.get("val", np.nan)))
        }

        points = []
        mcd_values = []
        nomad_values = []
        diffs = []
        for point in nomad_layer.get("points", []):
            key = (round(float(point["lat"]), 3), round(float(point["lng"]), 3))
            mcd_value = mcd_by_key.get(key)
            nomad_value = float(point.get("val", np.nan))
            if mcd_value is None or not np.isfinite(nomad_value):
                continue
            diff = float(mcd_value - nomad_value)
            points.append({
                "lat": float(point["lat"]),
                "lng": float(point["lng"]),
                "val": diff,
                "mcd_value": float(mcd_value),
                "nomad_value": nomad_value,
                "count": int(point.get("count", 1)),
            })
            mcd_values.append(float(mcd_value))
            nomad_values.append(nomad_value)
            diffs.append(diff)

        if not points:
            return None

        diff_arr = np.asarray(diffs, dtype=np.float64)
        mcd_arr = np.asarray(mcd_values, dtype=np.float64)
        nomad_arr = np.asarray(nomad_values, dtype=np.float64)
        values_abs = np.abs(diff_arr)
        return {
            "source": "nomad",
            "comparison": "MCD-NOMAD",
            "matched_ls": float(nomad_layer.get("ls", mcd_layer.get("ls", 0.0))),
            "sample_count": int(len(points)),
            "bias": float(np.mean(diff_arr)),
            "mae": float(np.mean(values_abs)),
            "rmse": float(np.sqrt(np.mean(diff_arr ** 2))),
            "correlation": self._safe_correlation(mcd_arr, nomad_arr),
            "minDiff": float(np.min(diff_arr)),
            "maxDiff": float(np.max(diff_arr)),
            "points": points,
        }

    @staticmethod
    def _round_ls(value: float) -> float:
        rounded = round(float(value), 3)
        return 0.0 if rounded == -0.0 else rounded

    @classmethod
    def _coverage_intervals_from_source(cls, source_data: dict, require_count: bool = False) -> list[dict]:
        ls_values = np.asarray(source_data.get("ls", []), dtype=np.float64)
        ozone = np.asarray(source_data.get("o3col", []), dtype=np.float32)
        if ls_values.size == 0 or ozone.ndim == 0:
            return []

        time_count = min(int(ls_values.shape[0]), int(ozone.shape[0]))
        if time_count == 0:
            return []

        ozone = ozone[:time_count]
        valid_cells = np.isfinite(ozone)
        if require_count:
            count = np.asarray(source_data.get("count", []), dtype=np.int32)
            if count.shape[:1] != ozone.shape[:1]:
                return []
            valid_cells = valid_cells & (count[:time_count] > 0)

        valid_by_time = valid_cells.reshape(time_count, -1).any(axis=1)
        ls_values = ls_values[:time_count]

        intervals = []
        run_start = None
        for idx, is_valid in enumerate(valid_by_time):
            if bool(is_valid) and np.isfinite(ls_values[idx]):
                if (
                    run_start is not None
                    and idx > run_start
                    and float(ls_values[idx]) - float(ls_values[idx - 1]) > MAX_COVERAGE_GAP_LS
                ):
                    run_ls = ls_values[run_start:idx]
                    intervals.append({
                        "start": cls._round_ls(np.nanmin(run_ls)),
                        "end": cls._round_ls(np.nanmax(run_ls)),
                    })
                    run_start = idx
                if run_start is None:
                    run_start = idx
                continue
            if run_start is not None:
                run_ls = ls_values[run_start:idx]
                intervals.append({
                    "start": cls._round_ls(np.nanmin(run_ls)),
                    "end": cls._round_ls(np.nanmax(run_ls)),
                })
                run_start = None

        if run_start is not None:
            run_ls = ls_values[run_start:time_count]
            intervals.append({
                "start": cls._round_ls(np.nanmin(run_ls)),
                "end": cls._round_ls(np.nanmax(run_ls)),
            })

        return intervals

    def get_ozone_coverage(self) -> dict:
        coverage = {
            "mcd": {},
            "openmars": {},
            "nomad": {},
        }

        for mars_year, year_data in sorted(self.overview.items()):
            mcd_intervals = self._coverage_intervals_from_source(year_data)
            if mcd_intervals:
                coverage["mcd"][mars_year] = mcd_intervals

            try:
                openmars = self.base.get_openmars_data(mars_year)
            except ValueError:
                openmars = None
            if openmars is not None:
                openmars_intervals = self._coverage_intervals_from_source(openmars)
                if openmars_intervals:
                    coverage["openmars"][mars_year] = openmars_intervals

        for mars_year, nomad_data in sorted(self.nomad.items()):
            nomad_intervals = self._coverage_intervals_from_source(nomad_data, require_count=True)
            if nomad_intervals:
                coverage["nomad"][mars_year] = nomad_intervals

        return coverage

    def get_ozone_capabilities(self) -> dict:
        diff_pairs = ["MCD-OpenMARS"]
        if bool(self.nomad):
            diff_pairs.append("MCD-NOMAD")
        return {
            "openmars": True,
            "nomad": bool(self.nomad),
            "diff_pairs": diff_pairs,
            "coverage": self.get_ozone_coverage(),
        }

    def get_ozone_overlay_payload(self, mars_year: int, ls: float) -> dict:
        mcd = self._build_layer_from_source(self.get_openmars_data(mars_year), ls, "mcd")
        openmars = self._match_openmars_layer(mars_year, mcd["ls"])
        nomad = self._match_nomad_layer(mars_year, mcd["ls"])
        nomad_validation = self._build_nomad_validation(mcd, nomad)

        available_sources = [
            source
            for source, layer in (("mcd", mcd), ("openmars", openmars), ("nomad", nomad))
            if layer is not None
        ]
        diff_candidates = []
        if openmars is not None:
            diff_candidates.append("MCD-OpenMARS")
        if nomad is not None:
            diff_candidates.append("MCD-NOMAD")

        return {
            "mars_year": int(mars_year),
            "requested_ls": float(ls),
            "anchor_ls": float(mcd["ls"]),
            "mcd": mcd,
            "openmars": openmars,
            "nomad": nomad,
            "available_sources": available_sources,
            "diff_candidates": diff_candidates,
            "validation": {
                "nomad": nomad_validation,
            },
            "capabilities": self.get_ozone_capabilities(),
        }
