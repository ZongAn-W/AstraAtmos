"""
前端探索大屏的专用可视化数据服务。
聚焦于数据空间降阶、降采样、抽稀聚合成图表格式。
"""
import logging
import warnings
from threading import RLock
import numpy as np

from config import MAX_LS_POINTS, LATITUDE_BANDS, MCD_VARIABLES
from services.data_service import DataService

logger = logging.getLogger("astraatmos.analysis")


class AnalysisService:
    def __init__(self, data_service: DataService, mcd_variables: list[str] | None = None):
        self.data_service = data_service
        self.mcd_variables = list(mcd_variables or MCD_VARIABLES)
        self._cache: dict[str, dict] = {}
        self._cache_lock = RLock()

    def get_globe_data(self, mars_year: int, ls: float, variable: str = "o3col") -> dict:
        om = self.data_service.get_openmars_data(mars_year)
        idx = self.data_service.get_nearest_ls_index(om["ls"], ls)
        if variable == "o3col":
            field = om["o3col"][idx]
        else:
            am = self.data_service.get_aligned_mcd_data(mars_year)
            if variable not in am:
                raise ValueError(f"变量 {variable} 不可用")
            field = am[variable][idx]

        points = []
        for i, lat in enumerate(om["lat"]):
            for j, lon in enumerate(om["lon"]):
                val = float(field[i, j])
                if not np.isnan(val):
                    points.append({
                        "lat": float(lat),
                        "lng": float(lon) if lon <= 180 else float(lon - 360),
                        "val": val,
                    })

        valid_vals = field[~np.isnan(field)]
        return {
            "points": points,
            "minVal": float(np.nanmin(valid_vals)) if len(valid_vals) > 0 else 0,
            "maxVal": float(np.nanmax(valid_vals)) if len(valid_vals) > 0 else 1,
            "ls": float(om["ls"][idx]),
            "mars_year": mars_year,
            "variable": variable,
        }

    @staticmethod
    def _normalize_lng(lng: float) -> float:
        normalized = ((float(lng) + 180.0) % 360.0) - 180.0
        return 0.0 if normalized == -0.0 else normalized

    @staticmethod
    def _longitude_distance(values: np.ndarray, target: float) -> np.ndarray:
        delta = np.abs(np.asarray(values, dtype=float) - float(target)) % 360.0
        return np.minimum(delta, 360.0 - delta)

    @staticmethod
    def _list_with_none(values: np.ndarray) -> list[float | None]:
        return [float(value) if np.isfinite(value) else None for value in np.asarray(values, dtype=float)]

    def _field_for_variable(self, mars_year: int, variable: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if variable == "o3col":
            data = self.data_service.get_openmars_data(mars_year)
            return (
                np.asarray(data["o3col"], dtype=np.float32),
                np.asarray(data["ls"], dtype=np.float32),
                np.asarray(data["lat"], dtype=np.float32),
                np.asarray(data["lon"], dtype=np.float32),
            )

        aligned = self.data_service.get_aligned_mcd_data(mars_year)
        if variable not in aligned:
            raise ValueError(f"Variable {variable} is not available")
        openmars = self.data_service.get_openmars_data(mars_year)
        return (
            np.asarray(aligned[variable], dtype=np.float32),
            np.asarray(aligned["ls"], dtype=np.float32),
            np.asarray(openmars["lat"], dtype=np.float32),
            np.asarray(openmars["lon"], dtype=np.float32),
        )

    def get_point_probe(self, mars_year: int, lat: float, lng: float, ls: float, variable: str = "o3col") -> dict:
        data_3d, ls_arr, lat_arr, lon_arr = self._field_for_variable(mars_year, variable)
        if data_3d.ndim != 3:
            raise ValueError(f"Variable {variable} must be a time/lat/lon field")

        current_idx = self.data_service.get_nearest_ls_index(ls_arr, ls)
        lon_norm = self._normalize_lng(lng)
        lon_display = np.asarray([self._normalize_lng(float(value)) for value in lon_arr], dtype=float)
        lat_distance = np.abs(lat_arr.astype(float) - float(lat))
        lon_distance = self._longitude_distance(lon_display, lon_norm)
        grid_distance = lat_distance[:, None] ** 2 + lon_distance[None, :] ** 2

        current_field = np.asarray(data_3d[current_idx], dtype=float)
        finite_mask = np.isfinite(current_field)
        if not np.any(finite_mask):
            raise ValueError(f"MY{mars_year} {variable} has no valid point data near Ls={ls}")

        masked_distance = np.where(finite_mask, grid_distance, np.inf)
        lat_index, lon_index = np.unravel_index(int(np.argmin(masked_distance)), masked_distance.shape)

        point_series = np.asarray(data_3d[:, lat_index, lon_index], dtype=float)
        global_series = self._nanmean_no_warn(data_3d, axis=(1, 2))
        latitude_series = self._nanmean_no_warn(data_3d[:, lat_index, :], axis=1)
        current_value = point_series[current_idx]
        current_global_mean = global_series[current_idx]
        current_latitude_mean = latitude_series[current_idx]

        def delta_from(value: float, baseline: float) -> float | None:
            if not np.isfinite(value) or not np.isfinite(baseline):
                return None
            return float(value - baseline)

        return {
            "requested": {"lat": float(lat), "lng": float(lon_norm), "ls": float(ls)},
            "gridPoint": {"lat": float(lat_arr[lat_index]), "lng": float(lon_display[lon_index])},
            "current": {
                "ls": float(ls_arr[current_idx]),
                "value": float(current_value) if np.isfinite(current_value) else None,
            },
            "series": {
                "ls": [float(value) for value in ls_arr],
                "point": self._list_with_none(point_series),
                "globalMean": self._list_with_none(global_series),
                "latitudeMean": self._list_with_none(latitude_series),
            },
            "comparison": {
                "globalMean": float(current_global_mean) if np.isfinite(current_global_mean) else None,
                "latitudeMean": float(current_latitude_mean) if np.isfinite(current_latitude_mean) else None,
                "pointMinusGlobal": delta_from(current_value, current_global_mean),
                "pointMinusLatitudeMean": delta_from(current_value, current_latitude_mean),
            },
            "variable": variable,
            "mars_year": mars_year,
        }

    def get_seasonal_heatmap(self, mars_year: int, variable: str = "o3col") -> dict:
        cache_key = f"heatmap_{mars_year}_{variable}"
        with self._cache_lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        if variable == "o3col":
            om = self.data_service.get_openmars_data(mars_year)
            data_3d = om["o3col"]
            ls_arr = om["ls"]
            lat_arr = om["lat"]
        else:
            am = self.data_service.get_aligned_mcd_data(mars_year)
            data_3d = am.get(variable)
            if data_3d is None:
                raise ValueError(f"变量 {variable} 不可用")
            ls_arr = am["ls"]
            om = self.data_service.get_openmars_data(mars_year)
            lat_arr = om["lat"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            zonal_mean = np.nanmean(data_3d, axis=2)

        n_time = len(ls_arr)
        step = max(1, n_time // MAX_LS_POINTS)
        ls_ds = ls_arr[::step]
        zm_ds = zonal_mean[::step]
        min_val, max_val = self._finite_min_max(zonal_mean)

        result = {
            "x": [float(v) for v in ls_ds],
            "y": [float(v) for v in lat_arr],
            "z": self._to_nested_list(zm_ds.T),
            "min": min_val,
            "max": max_val,
            "variable": variable,
        }
        with self._cache_lock:
            self._cache[cache_key] = result
        return result

    def get_seasonal_bands(self, mars_year: int) -> dict:
        cache_key = f"bands_{mars_year}"
        with self._cache_lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        om = self.data_service.get_openmars_data(mars_year)
        o3 = om["o3col"]
        ls_arr = om["ls"]
        lat_arr = om["lat"]

        n_time = len(ls_arr)
        step = max(1, n_time // MAX_LS_POINTS)
        ls_ds = ls_arr[::step]
        o3_ds = o3[::step]

        bands = []
        for band_def in LATITUDE_BANDS:
            mask = (lat_arr >= band_def["lat_min"]) & (lat_arr <= band_def["lat_max"])
            band_mean = np.nanmean(o3_ds[:, mask, :], axis=(1, 2))
            bands.append({
                "name": band_def["name"],
                "values": [float(v) for v in band_mean],
            })

        result = {
            "ls": [float(v) for v in ls_ds],
            "bands": bands,
        }
        with self._cache_lock:
            self._cache[cache_key] = result
        return result

    def get_env_variable_heatmap(self, mars_year: int, variable_name: str) -> dict:
        return self.get_seasonal_heatmap(mars_year, variable=variable_name)

    def get_correlation_matrix(self, mars_year: int) -> dict:
        cache_key = f"corr_{mars_year}"
        with self._cache_lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        om = self.data_service.get_openmars_data(mars_year)
        am = self.data_service.get_aligned_mcd_data(mars_year)

        var_names = ["o3col"] + self.mcd_variables
        n_vars = len(var_names)

        series_list = []
        o3_mean = np.nanmean(om["o3col"], axis=(1, 2))
        series_list.append(o3_mean)

        for var in self.mcd_variables:
            if var in am:
                v_3d = am[var]
                v_mean = np.nanmean(v_3d, axis=(1, 2))
                min_len = min(len(v_mean), len(o3_mean))
                series_list.append(v_mean[:min_len])
            else:
                series_list.append(np.full(len(o3_mean), np.nan))

        min_len = min(len(s) for s in series_list)
        series_list = [s[:min_len] for s in series_list]

        data_matrix = np.stack(series_list, axis=0)

        valid_mask = ~np.any(np.isnan(data_matrix), axis=0)
        data_clean = data_matrix[:, valid_mask]

        if data_clean.shape[1] < 10:
            corr = np.eye(n_vars)
        else:
            corr = np.corrcoef(data_clean)

        result = {
            "matrix": self._to_nested_list(corr),
            "variable_names": var_names,
        }
        with self._cache_lock:
            self._cache[cache_key] = result
        return result

    def get_diurnal_data(self, mars_year: int, ls: float, lat_band_name: str) -> dict:
        mc = self.data_service.get_mcd_data(mars_year)
        om = self.data_service.get_openmars_data(mars_year)

        band_def = next((b for b in LATITUDE_BANDS if b["name"] == lat_band_name), LATITUDE_BANDS[2])

        lat_arr = np.asarray(mc.get("lat", om["lat"]))
        lat_mask = (lat_arr >= band_def["lat_min"]) & (lat_arr <= band_def["lat_max"])

        hourly_key = next(
            (key for key in ("o3col_hourly", "O3COL_hourly", "Ozone_hourly") if key in mc),
            None,
        )
        hourly_data = np.asarray(mc[hourly_key]) if hourly_key else None
        if hourly_data is None:
            for key in ("o3col", "O3COL"):
                if key in mc and np.asarray(mc[key]).ndim == 4:
                    hourly_key = key
                    hourly_data = np.asarray(mc[key])
                    break

        if hourly_data is not None and "ls" in mc:
            mcd_ls = mc["ls"]
            sol_idx = self.data_service.get_nearest_ls_index(mcd_ls, ls)

            if sol_idx < hourly_data.shape[0]:
                data_at_sol = hourly_data[sol_idx]
                if data_at_sol.ndim != 3 or data_at_sol.shape[1] != lat_mask.shape[0]:
                    return self._unavailable_diurnal(ls, band_def, "Hourly ozone grid does not match latitude axis.")
                band_mean = np.nanmean(data_at_sol[:, lat_mask, :], axis=(1, 2))
                n_hours = data_at_sol.shape[0]
                hours = np.linspace(0, 24, n_hours, endpoint=False)
                return {
                    "hours": [float(h) for h in hours],
                    "ozone_values": [float(v) for v in band_mean],
                    "lat_band": band_def["name"],
                    "ls": float(ls),
                    "available": True,
                    "variable": "o3col",
                    "source_key": hourly_key,
                    "message": None,
                }

        return self._unavailable_diurnal(
            ls,
            band_def,
            "Hourly ozone is not available for this source. Use an MCD source with O3COL/o3col hourly data.",
        )

    @staticmethod
    def _to_nested_list(arr: np.ndarray) -> list[list[float | None]]:
        return [[float(v) if np.isfinite(v) else None for v in row] for row in arr]

    @staticmethod
    def _finite_min_max(arr: np.ndarray) -> tuple[float, float]:
        values = np.asarray(arr, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 0.0, 0.0
        return float(np.min(finite)), float(np.max(finite))

    @staticmethod
    def _unavailable_diurnal(ls: float, band_def: dict, message: str) -> dict:
        return {
            "hours": [],
            "ozone_values": [],
            "lat_band": band_def["name"],
            "ls": float(ls),
            "available": False,
            "variable": "o3col",
            "source_key": None,
            "message": message,
        }

    @staticmethod
    def _generate_simulated_diurnal(ls: float, band_def: dict) -> dict:
        hours = np.linspace(0, 24, 8, endpoint=False)
        base = 0.03
        amplitude = 0.008
        phase = 6.0
        values = base + amplitude * np.cos(2 * np.pi * (hours - phase) / 24)
        lat_center = (band_def["lat_min"] + band_def["lat_max"]) / 2
        values *= 1 + abs(lat_center) / 90 * 0.5

        return {
            "hours": [float(h) for h in hours],
            "ozone_values": [float(v) for v in values],
            "lat_band": band_def["name"],
            "ls": float(ls),
        }

    def get_coupling_data(self, mars_year: int, var1: str, var2: str) -> dict:
        cache_key = f"coupling_{mars_year}_{var1}_{var2}"
        with self._cache_lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        om = self.data_service.get_openmars_data(mars_year)
        am = self.data_service.get_aligned_mcd_data(mars_year)

        ls_arr = om["ls"]
        
        def get_global_mean(var_name):
            if var_name == "o3col":
                return np.nanmean(om["o3col"], axis=(1, 2))
            elif var_name in am:
                return np.nanmean(am[var_name], axis=(1, 2))
            else:
                return np.full(len(ls_arr), np.nan)

        v1_mean = get_global_mean(var1)
        v2_mean = get_global_mean(var2)

        n_time = len(ls_arr)
        step = max(1, n_time // MAX_LS_POINTS)
        
        result = {
            "ls": [float(v) for v in ls_arr[::step]],
            "var1": [float(v) for v in v1_mean[::step]],
            "var2": [float(v) for v in v2_mean[::step]],
            "var1_name": var1,
            "var2_name": var2
        }
        with self._cache_lock:
            self._cache[cache_key] = result
        return result

    def get_zonal_anomalies(self, mars_year: int, variable: str = "o3col") -> dict:
        cache_key = f"zonal_anomaly_{mars_year}_{variable}"
        with self._cache_lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        if variable == "o3col":
            data = self.data_service.get_openmars_data(mars_year)
            field = data["o3col"]
        else:
            am = self.data_service.get_aligned_mcd_data(mars_year)
            if variable not in am:
                raise ValueError(f"变量 {variable} 不可用")
            field = am[variable]
            data = self.data_service.get_openmars_data(mars_year)

        lat_arr = data["lat"]
        lon_arr = data["lon"]

        time_mean = np.nanmean(field, axis=0) # [lat, lon]
        zonal_mean = np.nanmean(time_mean, axis=1, keepdims=True) # [lat, 1]
        anomaly = time_mean - zonal_mean # [lat, lon]

        step_lat = max(1, len(lat_arr) // 45)
        step_lon = max(1, len(lon_arr) // 90)

        anomaly_ds = anomaly[::step_lat, ::step_lon]
        lat_ds = lat_arr[::step_lat]
        lon_ds = lon_arr[::step_lon]

        lon_ds_adj = []
        for lon in lon_ds:
            lon_ds_adj.append(float(lon) if float(lon) <= 180 else float(lon) - 360)

        sort_idx = np.argsort(lon_ds_adj)
        lon_ordered = np.array(lon_ds_adj)[sort_idx]
        anomaly_ordered = anomaly_ds[:, sort_idx]

        result = {
            "x": [float(v) for v in lon_ordered],
            "y": [float(v) for v in lat_ds],
            "z": self._to_nested_list(anomaly_ordered.T), # 转置使其符合 plotly 要求：y为纬度，x为经度。实际上 plotly heatmap 需要 z 为 2D array [len(y), len(x)] 即 [纬度, 经度]。
            "min": float(np.nanmin(anomaly_ordered)),
            "max": float(np.nanmax(anomaly_ordered)),
            "variable": variable,
        }
        # Plotly heatmap expects z rows to follow y (latitude), columns to follow x (longitude).
        result["z"] = self._to_nested_list(anomaly_ordered)
        with self._cache_lock:
            self._cache[cache_key] = result
        return result

    def get_solar_photochemical(self, mars_year: int, lat_band_name: str) -> dict:
        cache_key = f"solar_photo_{mars_year}_{lat_band_name}"
        with self._cache_lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        om = self.data_service.get_openmars_data(mars_year)
        am = self.data_service.get_aligned_mcd_data(mars_year)

        if "Solar_Flux_DN" not in am:
            raise ValueError("Solar_Flux_DN 缺失")

        band_def = next((b for b in LATITUDE_BANDS if b["name"] == lat_band_name), LATITUDE_BANDS[2])
        lat_arr = om["lat"]
        lat_mask = (lat_arr >= band_def["lat_min"]) & (lat_arr <= band_def["lat_max"])

        o3 = om["o3col"]
        solar = am["Solar_Flux_DN"]

        o3_band = np.nanmean(o3[:, lat_mask, :], axis=(1, 2))
        solar_band = np.nanmean(solar[:, lat_mask, :], axis=(1, 2))

        n_points = len(o3_band)
        step = max(1, n_points // 300)

        result = {
            "solar": [float(v) for v in solar_band[::step]],
            "ozone": [float(v) for v in o3_band[::step]],
            "ls": [float(v) for v in om["ls"][::step]],
            "lat_band": band_def["name"]
        }
        with self._cache_lock:
            self._cache[cache_key] = result
        return result

    def get_polar_dynamics(self, mars_year: int) -> dict:
        cache_key = f"polar_dyn_{mars_year}"
        with self._cache_lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        om = self.data_service.get_openmars_data(mars_year)
        am = self.data_service.get_aligned_mcd_data(mars_year)
        
        lat_arr = om["lat"]
        north_mask = lat_arr >= 60
        south_mask = lat_arr <= -60

        def get_polar_means(data_3d):
            n_mean = np.nanmean(data_3d[:, north_mask, :], axis=(1, 2))
            s_mean = np.nanmean(data_3d[:, south_mask, :], axis=(1, 2))
            return n_mean, s_mean

        o3_n, o3_s = get_polar_means(om["o3col"])
        
        if "U_Wind" in am and "V_Wind" in am:
            wind_speed = np.sqrt(am["U_Wind"]**2 + am["V_Wind"]**2)
            wind_n, wind_s = get_polar_means(wind_speed)
        else:
            wind_n = wind_s = np.zeros_like(o3_n)

        if "Temperature" in am:
            temp_n, temp_s = get_polar_means(am["Temperature"])
        else:
            temp_n = temp_s = np.zeros_like(o3_n)

        ls_arr = om["ls"]
        step = max(1, len(ls_arr) // MAX_LS_POINTS)

        result = {
            "ls": [float(v) for v in ls_arr[::step]],
            "north": {
                "ozone": [float(v) for v in o3_n[::step]],
                "wind": [float(v) for v in wind_n[::step]],
                "temp": [float(v) for v in temp_n[::step]],
            },
            "south": {
                "ozone": [float(v) for v in o3_s[::step]],
                "wind": [float(v) for v in wind_s[::step]],
                "temp": [float(v) for v in temp_s[::step]],
            }
        }
        with self._cache_lock:
            self._cache[cache_key] = result
        return result

    def get_research_suite(self, mars_year: int) -> dict:
        cache_key = f"research_suite_{mars_year}"
        with self._cache_lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        om = self.data_service.get_openmars_data(mars_year)
        am = self.data_service.get_aligned_mcd_data(mars_year)

        ls_arr = np.array(om["ls"])
        lat_arr = np.array(om["lat"])
        o3 = np.array(om["o3col"])

        heat_y = []
        heat_z = []
        band_amp = []
        band_peak_ls = []

        for band_def in LATITUDE_BANDS:
            mask = (lat_arr >= band_def["lat_min"]) & (lat_arr <= band_def["lat_max"])
            o3_band = self._nanmean_no_warn(o3[:, mask, :], axis=(1, 2))

            row = []
            for var in self.mcd_variables:
                if var not in am:
                    row.append(float("nan"))
                    continue
                var_band = self._nanmean_no_warn(am[var][:, mask, :], axis=(1, 2))
                row.append(float(self._safe_corr(o3_band, var_band)))

            valid_o3 = o3_band[np.isfinite(o3_band)]
            if valid_o3.size == 0:
                amp = float("nan")
                peak_ls = float("nan")
            else:
                amp = float(np.nanmax(valid_o3) - np.nanmin(valid_o3))
                peak_idx = int(np.nanargmax(o3_band))
                peak_ls = float(ls_arr[peak_idx]) if peak_idx < len(ls_arr) else float("nan")

            heat_y.append(band_def["name"])
            heat_z.append(row)
            band_amp.append(amp)
            band_peak_ls.append(peak_ls)

        global_o3 = self._nanmean_no_warn(o3, axis=(1, 2))
        step = max(1, len(ls_arr) // MAX_LS_POINTS)
        ls_ds = ls_arr[::step]

        trend_series = {}

        def add_trend_series(key: str, values: np.ndarray) -> None:
            sampled = np.asarray(values, dtype=float)[::step]
            if np.any(np.isfinite(sampled)):
                trend_series[key] = self._zscore(sampled)

        add_trend_series("o3", global_o3)
        if "Dust_Optical_Depth" in self.mcd_variables and "Dust_Optical_Depth" in am:
            add_trend_series("dust", self._nanmean_no_warn(am["Dust_Optical_Depth"], axis=(1, 2)))
        if "Temperature" in self.mcd_variables and "Temperature" in am:
            add_trend_series("temp", self._nanmean_no_warn(am["Temperature"], axis=(1, 2)))
        if "Solar_Flux_DN" in self.mcd_variables and "Solar_Flux_DN" in am:
            add_trend_series("solar", self._nanmean_no_warn(am["Solar_Flux_DN"], axis=(1, 2)))
        if "U_Wind" in self.mcd_variables and "V_Wind" in self.mcd_variables and "U_Wind" in am and "V_Wind" in am:
            wind_speed = np.sqrt(am["U_Wind"] ** 2 + am["V_Wind"] ** 2)
            add_trend_series("wind", self._nanmean_no_warn(wind_speed, axis=(1, 2)))

        time_mean = self._nanmean_no_warn(o3, axis=0)
        zonal_mean = self._nanmean_no_warn(time_mean, axis=1, keepdims=True)
        anomaly = time_mean - zonal_mean

        wave_bands = []
        wave_rms = []
        wave_span = []
        for band_def in LATITUDE_BANDS:
            mask = (lat_arr >= band_def["lat_min"]) & (lat_arr <= band_def["lat_max"])
            vals = anomaly[mask, :].reshape(-1)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                rms = float("nan")
                span = float("nan")
            else:
                rms = float(np.sqrt(np.mean(vals ** 2)))
                span = float(np.max(vals) - np.min(vals))
            wave_bands.append(band_def["name"])
            wave_rms.append(rms)
            wave_span.append(span)

        result = {
            "driver_band_heatmap": {
                "x": self.mcd_variables,
                "y": heat_y,
                "z": self._to_nested_list(np.array(heat_z, dtype=float)),
                "min": -1.0,
                "max": 1.0,
            },
            "seasonal_extremes": {
                "bands": heat_y,
                "amplitude": [float(v) if np.isfinite(v) else float("nan") for v in band_amp],
                "peak_ls": [float(v) if np.isfinite(v) else float("nan") for v in band_peak_ls],
            },
            "trend_lines": {
                "ls": [float(v) for v in ls_ds],
                "series": trend_series,
            },
            "wave_bands": {
                "bands": wave_bands,
                "rms": wave_rms,
                "peak_to_peak": wave_span,
            },
        }
        with self._cache_lock:
            self._cache[cache_key] = result
        return result

    def get_phase_space(self, mars_year: int, driver: str = "Dust_Optical_Depth") -> dict:
        cache_key = f"phase_space_{mars_year}_{driver}"
        with self._cache_lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        if driver not in self.mcd_variables:
            raise ValueError(f"变量 {driver} 不可用")

        om = self.data_service.get_openmars_data(mars_year)
        am = self.data_service.get_aligned_mcd_data(mars_year)
        if driver not in am:
            raise ValueError(f"变量 {driver} 不可用")

        ls_arr = np.array(om["ls"])
        o3_global = self._nanmean_no_warn(om["o3col"], axis=(1, 2))
        d_global = self._nanmean_no_warn(am[driver], axis=(1, 2))

        step = max(1, len(ls_arr) // MAX_LS_POINTS)
        ls_ds = ls_arr[::step]
        x_ds = d_global[::step]
        y_ds = o3_global[::step]

        valid_mask = np.isfinite(x_ds) & np.isfinite(y_ds) & np.isfinite(ls_ds)
        x_valid = x_ds[valid_mask]
        y_valid = y_ds[valid_mask]
        ls_valid = ls_ds[valid_mask]

        if x_valid.size >= 2:
            slope, intercept = np.polyfit(x_valid, y_valid, 1)
            corr_val = self._safe_corr(x_valid, y_valid)
        else:
            slope, intercept, corr_val = float("nan"), float("nan"), float("nan")

        result = {
            "driver": driver,
            "x": [float(v) for v in x_valid],
            "y": [float(v) for v in y_valid],
            "ls": [float(v) for v in ls_valid],
            "corr": float(corr_val) if np.isfinite(corr_val) else float("nan"),
            "trend": {
                "slope": float(slope) if np.isfinite(slope) else float("nan"),
                "intercept": float(intercept) if np.isfinite(intercept) else float("nan"),
            },
        }
        with self._cache_lock:
            self._cache[cache_key] = result
        return result

    @staticmethod
    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return float("nan")
        n = min(len(a), len(b))
        if n < 3:
            return float("nan")
        x = np.asarray(a[:n], dtype=float)
        y = np.asarray(b[:n], dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if np.sum(valid) < 3:
            return float("nan")
        xv = x[valid]
        yv = y[valid]
        if np.std(xv) == 0 or np.std(yv) == 0:
            return float("nan")
        return float(np.corrcoef(xv, yv)[0, 1])

    @staticmethod
    def _zscore(arr: np.ndarray) -> list[float | None]:
        values = np.asarray(arr, dtype=float)
        valid = values[np.isfinite(values)]
        if valid.size == 0:
            return [None] * len(values)
        mean_v = float(np.mean(valid))
        std_v = float(np.std(valid))
        if std_v == 0:
            return [0.0 if np.isfinite(v) else None for v in values]
        out = (values - mean_v) / std_v
        return [float(v) if np.isfinite(v) else None for v in out]

    @staticmethod
    def _nanmean_no_warn(arr: np.ndarray, axis=None, keepdims=False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(arr, axis=axis, keepdims=keepdims)
