import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


BACKEND_DIR = Path(__file__).resolve().parents[1]


def _load_data_service_module(monkeypatch):
    xarray = types.ModuleType("xarray")
    data_align = types.ModuleType("core.data_align")
    data_align.interpolate_mcd_to_openmars = lambda *args, **kwargs: None

    with monkeypatch.context() as import_context:
        import_context.setitem(sys.modules, "xarray", xarray)
        import_context.setitem(sys.modules, "core.data_align", data_align)
        spec = importlib.util.spec_from_file_location(
            "_mcd_data_service_under_test",
            BACKEND_DIR / "services" / "data_service.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    return module


class _FakeXarrayDataset(dict):
    def close(self):
        return None


class _FakeNetcdfVariable:
    def __init__(self, values):
        self._values = np.asarray(values)
        self.shape = self._values.shape

    def __getitem__(self, key):
        return self._values[key]


class _FakeNetcdfDataset:
    def __init__(self, variables):
        self.variables = variables

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def test_data_service_does_not_fall_back_to_a_different_mars_year(
    tmp_path: Path, monkeypatch
):
    module = _load_data_service_module(monkeypatch)
    wrong_year = tmp_path / "MCD_MY27_Lat-90-90_real.nc"
    wrong_year.touch()
    opened = []

    def open_dataset(path):
        opened.append(Path(path))
        return _FakeXarrayDataset(
            {"Ls": types.SimpleNamespace(values=np.array([27.0]))}
        )

    monkeypatch.setattr(module, "MCD_DIR", tmp_path)
    monkeypatch.setattr(module.xr, "open_dataset", open_dataset, raising=False)
    service = module.DataService.__new__(module.DataService)
    service.mcd = {}

    service._load_mcd(28)

    assert opened == []
    assert 28 not in service.mcd
