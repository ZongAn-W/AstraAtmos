import ast
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from routers import user_models  # noqa: E402


def test_uploaded_model_download_assets_are_declared_and_exist():
    assets = user_models.get_uploaded_model_download_assets()

    assert set(assets) == {"guide", "template", "earth-3hourly-guide", "earth-3hourly-template",
                           "earth-3hourly-shared-gate-template", "earth-3hourly-fullgrid-guide",
                           "earth-3hourly-fullgrid-template"}
    assert assets["guide"]["download_name"] == "astraatmos_uploaded_model_guide.md"
    assert assets["template"]["download_name"] == "astraatmos_uploaded_model_template.py"

    for asset in assets.values():
        path = asset["path"]
        assert path.is_file()
        assert path.stat().st_size > 0


def test_uploaded_model_download_asset_rejects_unknown_kind():
    try:
        user_models.get_uploaded_model_download_asset("missing")
    except user_models.HTTPException as exc:
        assert exc.status_code == 404
        assert exc.detail == "Uploaded model download asset not found"
    else:
        raise AssertionError("Expected HTTPException")


def test_uploaded_model_guide_documents_optional_ls_contract_without_changing_template():
    assets = user_models.get_uploaded_model_download_assets()
    guide = assets["guide"]["path"].read_text(encoding="utf-8")
    template = assets["template"]["path"].read_text(encoding="utf-8")

    assert "auxiliary_inputs" in guide
    assert "def forward(self, x, ls):" in guide
    assert "[batch, window]" in guide
    assert '"dtype": "float32"' in guide
    assert '"unit": "degree"' in guide
    assert "#     \"auxiliary_inputs\": {" in template
    assert "# def forward(self, x, ls):" in template
    assert '"topography": {' in guide
    assert '["batch", 1, "height", "width"]' in guide
    assert '"unit": "meter"' in guide
    assert "def forward(self, x, topography):" in guide
    assert "def forward(self, x, ls, topography):" in guide
    assert "# def forward(self, x, topography):" in template
    assert "# def forward(self, x, ls, topography):" in template
    assert "terrain_encoder" in template

    module = ast.parse(template)
    model_class = next(node for node in module.body if isinstance(node, ast.ClassDef))
    forward = next(
        node
        for node in model_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    assert [argument.arg for argument in forward.args.args] == ["self", "x"]


if __name__ == "__main__":
    test_uploaded_model_download_assets_are_declared_and_exist()
    test_uploaded_model_download_asset_rejects_unknown_kind()
    print("user model download tests passed")
