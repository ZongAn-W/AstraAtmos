import sys
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.analysis_service import AnalysisService  # noqa: E402


class FakeOverviewChartDataService:
    def __init__(self, mcd_data=None):
        self.lat = np.array([-75.0, 0.0, 75.0], dtype=np.float32)
        self.lon = np.array([-120.0, 0.0, 120.0], dtype=np.float32)
        self.ls = np.array([10.0, 20.0], dtype=np.float32)
        self.o3 = np.ones((2, 3, 3), dtype=np.float32)
        self.mcd_data = mcd_data or {}

    def get_openmars_data(self, mars_year: int) -> dict:
        return {
            "o3col": self.o3,
            "ls": self.ls,
            "lat": self.lat,
            "lon": self.lon,
        }

    def get_mcd_data(self, mars_year: int) -> dict:
        return self.mcd_data

    def get_aligned_mcd_data(self, mars_year: int) -> dict:
        return {"ls": self.ls, "lat": self.lat, "lon": self.lon}

    @staticmethod
    def get_nearest_ls_index(ls_array: np.ndarray, target_ls: float) -> int:
        return int(np.argmin(np.abs(ls_array - target_ls)))


def test_diurnal_data_does_not_report_temperature_as_ozone():
    temperature_hourly = np.zeros((2, 8, 3, 3), dtype=np.float32)
    temperature_hourly[0, :, 1, :] = np.arange(8, dtype=np.float32).reshape(8, 1) + 180.0
    service = AnalysisService(
        FakeOverviewChartDataService({
            "ls": np.array([10.0, 20.0], dtype=np.float32),
            "Temperature_hourly": temperature_hourly,
        })
    )

    result = service.get_diurnal_data(27, 10.0, "Equatorial (30S-30N)")

    assert result["variable"] == "o3col"
    assert result["available"] is False
    assert result["ozone_values"] == []


def test_diurnal_data_uses_hourly_ozone_when_available():
    ozone_hourly = np.zeros((2, 8, 3, 3), dtype=np.float32)
    ozone_hourly[0, :, 1, :] = np.arange(8, dtype=np.float32).reshape(8, 1) + 0.01
    service = AnalysisService(
        FakeOverviewChartDataService({
            "ls": np.array([10.0, 20.0], dtype=np.float32),
            "O3COL_hourly": ozone_hourly,
        })
    )

    result = service.get_diurnal_data(27, 10.0, "Equatorial (30S-30N)")

    assert result["available"] is True
    assert result["variable"] == "o3col"
    assert result["hours"] == [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0]
    np.testing.assert_allclose(result["ozone_values"], np.arange(8, dtype=np.float32) + 0.01)


def test_zonal_anomaly_matrix_matches_latitude_by_longitude_axes():
    fake = FakeOverviewChartDataService()
    fake.lat = np.array([-45.0, 45.0], dtype=np.float32)
    fake.lon = np.array([-120.0, 0.0, 120.0], dtype=np.float32)
    fake.o3 = np.array(
        [
            [[1.0, 2.0, 3.0], [10.0, 13.0, 16.0]],
            [[1.0, 2.0, 3.0], [10.0, 13.0, 16.0]],
        ],
        dtype=np.float32,
    )
    service = AnalysisService(fake)

    result = service.get_zonal_anomalies(27)

    assert result["x"] == [-120.0, 0.0, 120.0]
    assert result["y"] == [-45.0, 45.0]
    assert len(result["z"]) == len(result["y"])
    assert all(len(row) == len(result["x"]) for row in result["z"])
    np.testing.assert_allclose(result["z"], [[-1.0, 0.0, 1.0], [-3.0, 0.0, 3.0]])
