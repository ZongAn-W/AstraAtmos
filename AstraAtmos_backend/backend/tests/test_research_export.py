"""Scientific data contracts, real figure files, and authenticated HTTP export.

All model fields are explicitly synthetic; these tests perform no inference.
"""
import copy
from io import BytesIO
import json
import re
from types import SimpleNamespace
from zipfile import ZipFile
import xml.etree.ElementTree as ET

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest
from scipy.io import loadmat

from auth.dependencies import get_current_user
from routers import research_export as route
from schemas.research_export import ResearchExportRequest
from services.research_export_sources import ExportSourceStore, ExportUnavailable, register_mars
from services.research_figure import prepare_figure, render_figure, build_bundle


def synthetic_prediction():
    lat, lon = [65., 15., -40.], [-170., -40., 15., 135.]
    ref = np.asarray([[10, 11, 12, 13], [5, 6, 7, 8], [1, 2, 3, 4]], dtype=float)
    prediction = ref + np.asarray([[-2, -1, 0, 1], [1, -1, 2, -2], [3, -3, 1, 0]])
    def rows(field):
        return [dict(field=field.tolist(), lat=lat, lon=lon, minVal=float(field.min()), maxVal=float(field.max()))]
    return dict(ground_truth=rows(ref), prediction=rows(prediction), residual=rows(prediction - ref),
                horizon=1, ls_values=[93.2], input_ls_values=[90.], selected_variables=["Temperature"], model_info={})


@pytest.fixture
def env(tmp_path, monkeypatch):
    weight = tmp_path / "synthetic-model.bin"
    weight.write_bytes(b"synthetic fixture, not a trained model")
    task = SimpleNamespace(id=1, user_id=7, output_model_path=str(weight), model_source="official",
                           custom_model_name="Synthetic fixture A", hyperparameters='{"window":3}',
                           dataset_id="mcd_overview", dataset_snapshot=None, status="completed")
    dirs = {key: str(tmp_path) for key in ("ASTRAATMOS_OPENMARS_DIR", "ASTRAATMOS_MCD_DIR", "MCD_RAW_3H_DIR")}
    store = ExportSourceStore()
    import services.research_export_sources as sources_module
    monkeypatch.setattr(sources_module, "EXPORT_SOURCES", store)
    monkeypatch.setattr(route, "EXPORT_SOURCES", store)
    result = register_mars(synthetic_prediction(), task=task, hypers={"window": 3, "training_dataset": "mcd_overview"},
                           data_dirs=dirs, user_id=7, analysis="prediction", horizon=1, origin="90.0", variables=["Temperature"])
    ref = result["export_ref"]
    source = store.get(ref, 7)
    return SimpleNamespace(task=task, weight=weight, dirs=dirs, store=store, ref=ref, source=source, tmp=tmp_path)


def request(env, kind="triptych", **options):
    return ResearchExportRequest(sources=[env.ref], kind=kind, options={"language": "en", **options})


def test_units_limits_residual_and_coordinates(env):
    data = prepare_figure([env.source], request(env, unit="DU"))
    np.testing.assert_array_equal(data["latitude"], [65, 15, -40])
    np.testing.assert_array_equal(data["longitude"], [-170, -40, 15, 135])
    np.testing.assert_allclose(data["reference"][0], [1, 1.1, 1.2, 1.3])
    np.testing.assert_allclose(data["residual"], data["prediction"] - data["reference"])
    np.testing.assert_allclose(data["physical_limits"], [-.1, 1.4])
    np.testing.assert_allclose(data["residual_limits"], [-.3, .3])
    scatter = prepare_figure([env.source], request(env, "scatter", unit="DU"))
    np.testing.assert_array_equal(scatter["x"], data["reference"].ravel())
    assert "density" not in scatter


@pytest.mark.parametrize("change,reason", [
    (lambda d: d["prediction_latitude"][0].__setitem__(0, 60), "coordinates"),
    (lambda d: d["latitude"][0].__setitem__(1, 65), "monotonic"),
    (lambda d: d["residual"][0].__setitem__((0, 0), 2), "direction"),
    (lambda d: d["prediction"][0].__setitem__((0, 0), np.nan), "non-finite"),
])
def test_invalid_scientific_data_is_rejected(env, change, reason):
    source = copy.deepcopy(env.source)
    change(source.data)
    with pytest.raises(ValueError, match=reason):
        prepare_figure([source], request(env))


def test_missing_time_step_and_coordinates_rejected(env):
    req = request(env); req.step = 1
    with pytest.raises(ValueError, match="step"):
        prepare_figure([env.source], req)
    source = copy.deepcopy(env.source)
    source.metadata["ls_values"] = []
    with pytest.raises(ValueError, match="time/Ls"):
        prepare_figure([source], request(env))


@pytest.mark.parametrize("fmt", ["png", "pdf", "svg"])
def test_real_output_size_and_vectors(env, fmt):
    data = prepare_figure([env.source], request(env, format=fmt, width_mm=180, height_mm=85, dpi=300))
    content = render_figure(data)
    if fmt == "png":
        image = Image.open(BytesIO(content))
        assert image.size == (int(180 / 25.4 * 300), int(85 / 25.4 * 300))
        assert abs(image.info["dpi"][0] - 300) < .1
    elif fmt == "svg":
        svg = ET.fromstring(content)
        assert abs(float(svg.attrib["width"].removesuffix("pt")) - 180 / 25.4 * 72) < .01
        ns = {"s": "http://www.w3.org/2000/svg"}
        assert len(svg.findall(".//s:text", ns)) > 10  # Actual text, not paths.
        assert len(svg.findall(".//s:image", ns)) >= 3  # Local rasterization of meshes.
    else:
        media = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)", content)
        assert media
        assert abs(float(media[1]) - 180 / 25.4 * 72) < .01
        assert b"/Font" in content


def test_chinese_single_column_and_600dpi_bundle(env):
    req = request(env, language="zh", format="png", dpi=600, width_mm=85, height_mm=215, font_size=8)
    data = prepare_figure([env.source], req)
    content = render_figure(data)
    image = Image.open(BytesIO(content))
    assert image.size == (int(85 / 25.4 * 600), int(215 / 25.4 * 600))
    assert abs(image.info["dpi"][0] - 600) < .1
    with ZipFile(BytesIO(build_bundle(data, content))) as archive:
        assert set(archive.namelist()) == {"figure.png", "figure.mat", "rebuild.m", "README.txt", "metadata.json"}
        mat = loadmat(BytesIO(archive.read("figure.mat")), simplify_cells=True)["figure_data"]
        np.testing.assert_array_equal(mat["reference"], data["reference"])
        np.testing.assert_array_equal(mat["latitude"], data["latitude"])
        np.testing.assert_array_equal(mat["physical_limits"], data["physical_limits"])
        metadata = archive.read("metadata.json").decode()
        assert str(env.tmp) not in metadata
        assert "JWT" not in metadata
        assert json.loads(metadata)["matlab_execution_verified"] is False
        assert json.loads(mat["metadata_json"])["suggested_caption"] == json.loads(metadata)["suggested_caption"]
        assert json.loads(metadata)["selected_step"] == 1
        assert "eval(" not in archive.read("rebuild.m").decode()


def test_pfi_negative_values_and_test_scope(env):
    data = dict(items=[dict(name="Temperature", importance=.2), dict(name="U_Wind", importance=-.03)],
                baseline_metric="r2", baseline_value=.6, sampling={"sample_size": 20, "test_windows": 100})
    ref = env.store.register(user_id=7, task_id=1, planet="mars", analysis="pfi", horizon=1, origin="", variables=[],
                             guard=env.source.guard, data=data, metadata={**env.source.metadata, "scope": "sampled_test_set"})
    req = ResearchExportRequest(sources=[ref], kind="pfi", options={"language": "en"})
    prepared = prepare_figure([env.store.get(ref, 7)], req)
    np.testing.assert_array_equal(prepared["values"], [.2, -.03])
    content = render_figure(prepared)
    with ZipFile(BytesIO(build_bundle(prepared, content))) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        assert metadata["value_unit"] == "dimensionless"
        assert metadata["selected_step"] is None


@pytest.mark.parametrize("model_count", [2, 8])
def test_curves_convert_metrics_preserve_step_order_and_mat_struct(env, model_count):
    refs = []
    for task_id in range(1, model_count + 1):
        refs.append(env.store.register(user_id=7, task_id=task_id, planet="mars", analysis="metrics", horizon=3,
                                      origin="", variables=[], guard=env.source.guard,
                                      data=dict(per_step=[dict(step=i, rmse=i * task_id, r2=.5) for i in (1, 2, 3)]),
                                      metadata={**env.source.metadata, "model": f"Synthetic {task_id}", "scope": "full_test_set"}))
    req = ResearchExportRequest(sources=refs, kind="step_curves", options={"language": "en", "unit": "DU", "height_mm": 100})
    data = prepare_figure([env.store.get(ref, 7) for ref in refs], req)
    np.testing.assert_allclose(data["curves"][1]["y"], [.2, .4, .6])
    req.metric = "r2"
    r2_data = prepare_figure([env.store.get(ref, 7) for ref in refs], req)
    np.testing.assert_array_equal(r2_data["curves"][0]["y"], [.5] * 3)
    assert r2_data["value_unit"] == "dimensionless"
    content = render_figure(data)
    with ZipFile(BytesIO(build_bundle(data, content))) as archive:
        curves = loadmat(BytesIO(archive.read("figure.mat")), struct_as_record=True)["figure_data"][0, 0]["curves"]
        assert curves.dtype.names == ("name", "x", "y")
        assert curves.shape == (1, model_count)


def test_expiry_scope_cost_and_whitelist(env):
    with pytest.raises(PermissionError):
        env.store.get(env.ref, 9)
    mismatched = {**env.ref, "horizon": 2}
    with pytest.raises(ExportUnavailable, match="conditions"):
        env.store.get(mismatched, 7)
    env.store.entries[env.ref["id"]] = copy.copy(env.source)
    object.__setattr__(env.store.entries[env.ref["id"]], "expires", 0)
    with pytest.raises(ExportUnavailable, match="expired"):
        env.store.get(env.ref, 7)
    with pytest.raises(ValueError):
        ResearchExportRequest(sources=[env.ref], kind="triptych", options={"python_code": "arbitrary"})
    with pytest.raises(ValueError):
        request(env, width_mm=240, height_mm=260, dpi=600)


@pytest.fixture
def http_env(env):
    app = FastAPI(); app.include_router(route.router, prefix="/api")
    user = SimpleNamespace(id=7, role="user")
    app.dependency_overrides[get_current_user] = lambda: user
    class ResultOnlyService:
        async def _prepare_task_prediction_context(self, task_id, current_user, **kwargs):
            if env.task.user_id != current_user.id:
                raise PermissionError("Task ownership changed")
            return env.task, {}, env.dirs, None
        def _cleanup_temp_data_root(self, temp):
            assert temp is None
        def predict_task(self, *args, **kwargs):
            raise AssertionError("Export must not perform inference")
    app.state.training_inference_service = ResultOnlyService()
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, app=app, user=user, env=env)


def test_http_preview_download_and_permissions(http_env):
    req = request(http_env.env).model_dump()
    res = http_env.client.post("/api/predict/research-export/preview", json=req)
    assert res.status_code == 200, res.text[:500]
    assert res.headers["content-type"] == "image/png"
    assert res.headers["cache-control"] == "no-store"
    req["bundle"] = True
    res = http_env.client.post("/api/predict/research-export/download", json=req)
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/zip"
    assert ZipFile(BytesIO(res.content)).testzip() is None
    http_env.user.id = 8
    assert http_env.client.post("/api/predict/research-export/download", json=req).status_code == 403
    http_env.app.dependency_overrides.clear()
    assert http_env.client.post("/api/predict/research-export/download", json=req).status_code == 401


def test_http_rejects_more_than_eight_models_before_rendering(http_env, monkeypatch):
    def forbidden_render(*args, **kwargs):
        raise AssertionError("Invalid model count must not invoke rendering")
    monkeypatch.setattr(route, "render_figure", forbidden_render)
    req = request(http_env.env).model_dump()
    req.update(kind="step_curves", sources=[http_env.env.ref] * 31)
    for endpoint in ("preview", "download"):
        res = http_env.client.post(f"/api/predict/research-export/{endpoint}", json=req)
        assert res.status_code == 422
        assert res.json()["detail"][0]["type"] == "too_long"
        assert res.json()["detail"][0]["loc"] == ["body", "sources"]


def test_http_prediction_ref_reaches_triptych_and_scatter_export(http_env):
    from auth.dependencies import get_optional_user
    from routers import predict
    http_env.app.include_router(predict.router, prefix="/api")
    http_env.app.dependency_overrides[get_optional_user] = lambda: http_env.user
    service = http_env.app.state.training_inference_service
    forbid_inference = service.predict_task
    calls = []
    async def existing_result(**kwargs):
        calls.append(kwargs)
        return {**synthetic_prediction(), "export_ref": http_env.env.ref}
    service.predict_task = existing_result
    response = http_env.client.post("/api/predict/run", json={"training_task_id": 1, "ls_start": 90, "horizon": 1})
    assert response.status_code == 200, response.text
    assert response.json()["export_ref"] == http_env.env.ref
    # Only the result retrieval endpoint may retrieve a prediction. Neither
    # export is allowed to call it again or regenerate a missing result.
    service.predict_task = forbid_inference
    for kind in ("triptych", "scatter"):
        payload = request(http_env.env, kind, format="png").model_dump()
        payload["sources"] = [response.json()["export_ref"]]
        exported = http_env.client.post("/api/predict/research-export/download", json=payload)
        assert exported.status_code == 200, exported.text[:100] if exported.status_code != 200 else ""
        assert Image.open(BytesIO(exported.content)).format == "PNG"
    assert len(calls) == 1


def test_http_metrics_and_pfi_keep_export_refs(http_env):
    from auth.dependencies import get_optional_user
    from routers import predict
    http_env.app.include_router(predict.router, prefix="/api")
    http_env.app.dependency_overrides[get_optional_user] = lambda: http_env.user
    service = http_env.app.state.training_inference_service
    metric = dict(step=1, rmse=1., mae=.5, r2=.7, ssim=.8)
    metrics = dict(overall={**metric, "step": 0}, per_step=[metric])
    pfi = dict(items=[dict(name="Temperature", importance=-.03)], baseline_metric="r2", baseline_value=.7)
    refs = {}
    for analysis, data in (("metrics", metrics), ("pfi", pfi)):
        refs[analysis] = http_env.env.store.register(user_id=7, task_id=1, planet="mars", analysis=analysis,
            horizon=1, origin="", variables=[], guard=http_env.env.source.guard,
            data=data, metadata=http_env.env.source.metadata)
    async def existing_metrics(**kwargs):
        return {**metrics, "export_ref": refs["metrics"]}
    async def existing_pfi(**kwargs):
        return {**pfi, "export_ref": refs["pfi"]}
    service.task_test_set_metrics = existing_metrics
    service.task_permutation_importance = existing_pfi
    response = http_env.client.post("/api/predict/metrics", json={"training_task_id": 1, "horizon": 1})
    assert response.status_code == 200, response.text
    assert response.json()["export_ref"] == refs["metrics"]
    response = http_env.client.get("/api/predict/permutation-importance", params={"training_task_id": 1, "horizon": 1})
    assert response.status_code == 200, response.text
    assert response.json()["export_ref"] == refs["pfi"]


def test_http_stale_files_missing_conditions_and_ownership(http_env):
    req = request(http_env.env).model_dump()
    req["sources"][0]["origin"] = "120.0"
    assert http_env.client.post("/api/predict/research-export/download", json=req).status_code == 409
    req = request(http_env.env).model_dump()
    http_env.env.task.user_id = 8
    assert http_env.client.post("/api/predict/research-export/download", json=req).status_code == 403
    http_env.env.task.user_id = 7
    http_env.env.weight.write_bytes(b"changed synthetic model")
    assert http_env.client.post("/api/predict/research-export/download", json=req).status_code == 409


def test_result_changed_during_render_and_busy_capacity(http_env, monkeypatch):
    req = request(http_env.env).model_dump()
    for _ in range(2):
        assert route.EXPORT_SLOTS.acquire(blocking=False)
    try:
        assert http_env.client.post("/api/predict/research-export/download", json=req).status_code == 429
    finally:
        route.EXPORT_SLOTS.release(); route.EXPORT_SLOTS.release()
    real_render = route.render_figure
    def changed_render(*args, **kwargs):
        content = real_render(*args, **kwargs)
        http_env.env.weight.write_bytes(b"synthetic change during rendering")
        return content
    monkeypatch.setattr(route, "render_figure", changed_render)
    assert http_env.client.post("/api/predict/research-export/download", json=req).status_code == 409
