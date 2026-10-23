from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
SCHEMA_SOURCE = (BACKEND_DIR / "schemas" / "predict.py").read_text(encoding="utf-8")
ROUTER_SOURCE = (BACKEND_DIR / "routers" / "predict.py").read_text(encoding="utf-8")
SERVICE_SOURCE = (BACKEND_DIR / "services" / "inference_service.py").read_text(encoding="utf-8")


def test_prediction_schemas_and_queries_allow_the_training_range():
    assert SCHEMA_SOURCE.count("ge=1, le=30") >= 2
    assert "ge=1, le=3)" not in SCHEMA_SOURCE
    assert ROUTER_SOURCE.count("Query(3, ge=1, le=30)") >= 2
    assert "Query(3, ge=1, le=3)," not in ROUTER_SOURCE


def test_inference_service_validates_every_prepared_task_context():
    assert "from services.prediction_horizon import validate_prediction_horizon" in SERVICE_SOURCE
    assert SERVICE_SOURCE.count("validate_prediction_horizon(hypers, horizon)") == 7


def test_trained_analysis_routes_preserve_horizon_validation_as_bad_requests():
    error_distribution_route = ROUTER_SOURCE.split(
        "async def get_error_distribution", 1
    )[1].split("async def get_permutation_importance", 1)[0]
    pfi_route = ROUTER_SOURCE.split(
        "async def get_permutation_importance", 1
    )[1]

    for route_source in (error_distribution_route, pfi_route):
        assert "except ValueError as e:" in route_source
        assert "raise HTTPException(status_code=400, detail=str(e))" in route_source


def test_training_inference_cache_paths_offload_sync_work_to_threads():
    assert "import asyncio" in SERVICE_SOURCE
    assert SERVICE_SOURCE.count("await asyncio.to_thread(") >= 4
    assert "await asyncio.to_thread(self._predict_task_with_context," in SERVICE_SOURCE
    assert "await asyncio.to_thread(self._official_task_test_set_metrics," in SERVICE_SOURCE
    assert "await asyncio.to_thread(self._official_task_test_set_arrays," in SERVICE_SOURCE
    assert "await asyncio.to_thread(self._task_permutation_importance_with_context," in SERVICE_SOURCE
