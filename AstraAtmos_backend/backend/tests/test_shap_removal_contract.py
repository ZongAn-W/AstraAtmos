import re
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parents[1]
SHAP_IDENTIFIER = re.compile(r"\b(?:shap|shapley)\b", re.IGNORECASE)


def _text(relative_path: str) -> str:
    return (REPO_DIR / relative_path).read_text(encoding="utf-8")


def test_backend_and_release_sources_have_no_shap_dependency_or_surface():
    paths = [
        "AstraAtmos_backend/backend/requirements.txt",
        "AstraAtmos_backend/backend/routers/predict.py",
        "AstraAtmos_backend/backend/schemas/predict.py",
        "scripts/release/build_portable_windows.ps1",
        "scripts/release/repair-runtime.ps1",
    ]

    for path in paths:
        assert SHAP_IDENTIFIER.search(_text(path)) is None, path


def test_prediction_modules_import_when_shap_is_unavailable():
    code = """
import builtins
original_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == 'shap' or name.startswith('shap.'):
        raise ModuleNotFoundError('blocked shap import')
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked_import
import routers.predict
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_registered_routes_drop_shap_and_keep_supported_analysis():
    from routers.predict import router

    app = FastAPI()
    app.include_router(router)
    paths = set(app.openapi()["paths"])

    assert "/predict/shapley" not in paths
    assert "/predict/shapley-global" not in paths
    assert "/predict/permutation-importance" in paths
    assert "/predict/error-distribution" in paths


def test_frontend_product_source_has_no_shap_clients_components_state_copy_or_nodes():
    frontend_src = REPO_DIR / "frontend" / "src"
    offenders = []
    for path in frontend_src.rglob("*"):
        if (
            path.is_file()
            and path.suffix in {".js", ".jsx"}
            and ".test." not in path.name
            and SHAP_IDENTIFIER.search(path.read_text(encoding="utf-8"))
        ):
            offenders.append(path.relative_to(REPO_DIR).as_posix())

    assert offenders == []
    assert not (frontend_src / "pages/PredictPage/ShapleyImportanceChart.jsx").exists()
    assert not (frontend_src / "pages/PredictPage/ShapImportanceChart.jsx").exists()


def test_readme_has_no_shap_feature_or_stack_claim():
    assert SHAP_IDENTIFIER.search(_text("README.md")) is None
