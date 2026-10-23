from __future__ import annotations

import numpy as np

OVERVIEW_ENV_FIELDS = ["Temperature", "U_Wind", "V_Wind", "Solar_Flux_DN"]


def _as_1d(data: dict, key: str) -> np.ndarray:
    values = data.get(key)
    if values is None:
        raise ValueError(f"uploaded dataset missing {key}")
    return np.asarray(values, dtype=np.float32).reshape(-1)


def _as_3d(data: dict, key: str) -> np.ndarray:
    values = data.get(key)
    if values is None:
        raise ValueError(f"uploaded dataset missing {key}")
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 4:
        arr = np.nanmean(arr, axis=1)
    if arr.ndim != 3:
        raise ValueError(f"uploaded dataset field {key} must be 3D, got {arr.shape}")
    return arr


def _sort_by_ls(data: dict, field_names: list[str]) -> dict:
    ls = _as_1d(data, "ls")
    order = np.argsort(ls)
    out = dict(data)
    out["ls"] = ls[order]
    for name, values in list(out.items()):
        if name in {"ls", "lat", "lon", "data_type", "mars_year", "source_file", "raw_3h_source_file"}:
            continue
        arr = np.asarray(values)
        if arr.ndim >= 1 and arr.shape[0] == len(order):
            out[name] = arr[order]
    return out


class UserMcdOverviewDataView:
    """DataService-like adapter for one uploaded MCD overview dataset."""

    def __init__(self, upload_id: int, mars_year: int, data: dict, hourly_data: dict | None = None):
        self.upload_id = int(upload_id)
        self.mars_year = int(mars_year)
        self._data = _sort_by_ls(data, ["o3col", *OVERVIEW_ENV_FIELDS])
        self._hourly_data = _sort_by_ls(hourly_data or {}, ["O3COL_hourly", "o3col_hourly"]) if hourly_data else {}

    def _check_year(self, mars_year: int) -> None:
        if int(mars_year) != self.mars_year:
            raise ValueError(f"uploaded MCD source only supports MY{self.mars_year}")

    def get_openmars_data(self, mars_year: int) -> dict:
        self._check_year(mars_year)
        return {
            "o3col": _as_3d(self._data, "o3col"),
            "ls": _as_1d(self._data, "ls"),
            "lat": _as_1d(self._data, "lat"),
            "lon": _as_1d(self._data, "lon"),
        }

    def get_aligned_mcd_data(self, mars_year: int) -> dict:
        self._check_year(mars_year)
        out = {
            "ls": _as_1d(self._data, "ls"),
            "lat": _as_1d(self._data, "lat"),
            "lon": _as_1d(self._data, "lon"),
        }
        for field_name in OVERVIEW_ENV_FIELDS:
            if field_name in self._data:
                out[field_name] = _as_3d(self._data, field_name)
        return out

    def get_mcd_data(self, mars_year: int) -> dict:
        self._check_year(mars_year)
        out = self.get_aligned_mcd_data(mars_year)
        if self._hourly_data:
            out.update(self._hourly_data)
        return out

    def get_available_years(self) -> list[int]:
        return [self.mars_year]

    def get_ls_range(self, mars_year: int) -> tuple[float, float]:
        self._check_year(mars_year)
        ls = _as_1d(self._data, "ls")
        return float(ls[0]), float(ls[-1])

    @staticmethod
    def get_nearest_ls_index(ls_array: np.ndarray, target_ls: float) -> int:
        return int(np.argmin(np.abs(np.asarray(ls_array, dtype=float) - float(target_ls))))


def _points_from_field(
    field: np.ndarray,
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    count: np.ndarray | None = None,
) -> list[dict]:
    points = []
    for i, lat in enumerate(lat_arr):
        for j, lon in enumerate(lon_arr):
            val = float(field[i, j])
            if not np.isfinite(val):
                continue
            lon_value = float(lon) if float(lon) <= 180 else float(lon) - 360
            point = {"lat": float(lat), "lng": lon_value, "val": val}
            if count is not None:
                point["count"] = int(count[i, j])
            points.append(point)
    return points


def build_uploaded_ozone_layer(data: dict, ls: float, source: str) -> dict:
    sorted_data = _sort_by_ls(data, ["o3col"])
    ls_arr = _as_1d(sorted_data, "ls")
    idx = int(np.argmin(np.abs(ls_arr.astype(float) - float(ls))))
    field = _as_3d(sorted_data, "o3col")[idx]
    count = None
    if source == "nomad" and "count" in sorted_data:
        count_arr = np.asarray(sorted_data["count"])
        if count_arr.ndim == 3 and count_arr.shape[0] > idx:
            count = count_arr[idx]
            field = np.where(count > 0, field, np.nan)
    valid = field[np.isfinite(field)]
    return {
        "source": source,
        "points": _points_from_field(field, _as_1d(sorted_data, "lat"), _as_1d(sorted_data, "lon"), count=count),
        "minVal": float(np.nanmin(valid)) if valid.size else 0.0,
        "maxVal": float(np.nanmax(valid)) if valid.size else 1.0,
        "ls": float(ls_arr[idx]),
    }


def _safe_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2 or b.size < 2 or float(np.nanstd(a)) == 0.0 or float(np.nanstd(b)) == 0.0:
        return None
    corr = float(np.corrcoef(a, b)[0, 1])
    return corr if np.isfinite(corr) else None


def build_uploaded_nomad_validation(mcd_layer: dict, nomad_layer: dict | None) -> dict | None:
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
        if key not in mcd_by_key:
            continue
        nomad_value = float(point.get("val", np.nan))
        if not np.isfinite(nomad_value):
            continue
        mcd_value = mcd_by_key[key]
        diff = float(mcd_value - nomad_value)
        points.append({
            "lat": float(point["lat"]),
            "lng": float(point["lng"]),
            "val": diff,
            "mcd_value": mcd_value,
            "nomad_value": nomad_value,
            "count": int(point.get("count", 1)),
        })
        mcd_values.append(mcd_value)
        nomad_values.append(nomad_value)
        diffs.append(diff)
    if not points:
        return None
    diff_arr = np.asarray(diffs, dtype=np.float64)
    return {
        "source": "nomad",
        "comparison": "MCD-NOMAD",
        "matched_ls": float(nomad_layer.get("ls", mcd_layer.get("ls", 0.0))),
        "sample_count": int(len(points)),
        "bias": float(np.mean(diff_arr)),
        "mae": float(np.mean(np.abs(diff_arr))),
        "rmse": float(np.sqrt(np.mean(diff_arr ** 2))),
        "correlation": _safe_correlation(np.asarray(mcd_values), np.asarray(nomad_values)),
        "minDiff": float(np.min(diff_arr)),
        "maxDiff": float(np.max(diff_arr)),
        "points": points,
    }
