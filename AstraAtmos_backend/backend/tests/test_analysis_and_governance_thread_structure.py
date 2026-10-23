from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
ANALYSIS_ROUTER_SOURCE = (BACKEND_DIR / "routers" / "analysis.py").read_text(encoding="utf-8")
ANALYSIS_SERVICE_SOURCE = (BACKEND_DIR / "services" / "analysis_service.py").read_text(encoding="utf-8")
DATA_GOVERNANCE_SOURCE = (BACKEND_DIR / "services" / "data_governance_service.py").read_text(encoding="utf-8")


def test_analysis_routes_offload_heavy_sync_work_to_threads():
    assert "import asyncio" in ANALYSIS_ROUTER_SOURCE
    assert ANALYSIS_ROUTER_SOURCE.count("await asyncio.to_thread(") >= 10
    assert "await asyncio.to_thread(service.get_globe_data," in ANALYSIS_ROUTER_SOURCE
    assert "await asyncio.to_thread(service.get_point_probe," in ANALYSIS_ROUTER_SOURCE
    assert "await asyncio.to_thread(service.get_seasonal_heatmap," in ANALYSIS_ROUTER_SOURCE
    assert "await asyncio.to_thread(service.get_seasonal_bands," in ANALYSIS_ROUTER_SOURCE
    assert "await asyncio.to_thread(service.get_env_variable_heatmap," in ANALYSIS_ROUTER_SOURCE
    assert "await asyncio.to_thread(service.get_correlation_matrix," in ANALYSIS_ROUTER_SOURCE
    assert "await asyncio.to_thread(service.get_coupling_data," in ANALYSIS_ROUTER_SOURCE
    assert "await asyncio.to_thread(service.get_zonal_anomalies," in ANALYSIS_ROUTER_SOURCE
    assert "await asyncio.to_thread(service.get_solar_photochemical," in ANALYSIS_ROUTER_SOURCE
    assert "await asyncio.to_thread(service.get_polar_dynamics," in ANALYSIS_ROUTER_SOURCE
    assert "await asyncio.to_thread(service.get_research_suite," in ANALYSIS_ROUTER_SOURCE
    assert "await asyncio.to_thread(service.get_phase_space," in ANALYSIS_ROUTER_SOURCE


def test_analysis_service_protects_its_result_cache_under_threaded_access():
    assert "from threading import RLock" in ANALYSIS_SERVICE_SOURCE
    assert "self._cache_lock" in ANALYSIS_SERVICE_SOURCE
    assert ANALYSIS_SERVICE_SOURCE.count("with self._cache_lock:") >= 8


def test_data_governance_service_offloads_dataset_scans_and_quality_scoring():
    assert "import asyncio" in DATA_GOVERNANCE_SOURCE
    assert "from threading import RLock" in DATA_GOVERNANCE_SOURCE
    assert "self._meta_cache_lock" in DATA_GOVERNANCE_SOURCE
    assert "self._quality_cache_lock" in DATA_GOVERNANCE_SOURCE
    assert DATA_GOVERNANCE_SOURCE.count("await asyncio.to_thread(") >= 3
    assert "await asyncio.to_thread(self._get_dataset_meta," in DATA_GOVERNANCE_SOURCE
    assert "await asyncio.to_thread(self._get_quality_metrics," in DATA_GOVERNANCE_SOURCE
    assert DATA_GOVERNANCE_SOURCE.count("with self._meta_cache_lock:") >= 2
    assert DATA_GOVERNANCE_SOURCE.count("with self._quality_cache_lock:") >= 2
