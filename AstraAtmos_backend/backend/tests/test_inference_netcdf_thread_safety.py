import asyncio
import threading
import time
from pathlib import Path

import numpy as np

from services import inference_service as inference_module


class NonThreadSafeDataset:
    _guard = threading.Lock()
    _active = False

    def __init__(self, path, *args, **kwargs):
        with self._guard:
            if self.__class__._active:
                raise RuntimeError("concurrent NetCDF open")
            self.__class__._active = True
        time.sleep(0.03)
        path = Path(path)
        if "openmars" in {part.lower() for part in path.parts}:
            self.variables = {
                "o3col": np.arange(
                    6 * 36 * 72, dtype=np.float32
                ).reshape(6, 36, 72),
                "Ls": np.linspace(0.0, 5.0, 6, dtype=np.float32),
            }
        else:
            self.variables = {
                "Ls": np.array([0.0, 2.0, 4.0], dtype=np.float32),
                "Temperature": np.arange(
                    3 * 2 * 36 * 72, dtype=np.float32
                ).reshape(3, 2, 36, 72),
            }

    def close(self):
        with self._guard:
            self.__class__._active = False


def test_prepare_data_serializes_netcdf_dataset_access(monkeypatch, tmp_path):
    openmars_dir = tmp_path / "openmars"
    mcd_dir = tmp_path / "mcd"
    openmars_dir.mkdir()
    mcd_dir.mkdir()
    (openmars_dir / "openmars_ozo_my27.nc").write_bytes(b"stub")
    (mcd_dir / "MCD_MY27.nc").write_bytes(b"stub")

    monkeypatch.setattr(inference_module.nc, "Dataset", NonThreadSafeDataset)

    service = inference_module.InferenceService()
    data_dirs = {
        "ASTRAATMOS_OPENMARS_DIR": str(openmars_dir),
        "ASTRAATMOS_MCD_DIR": str(mcd_dir),
    }

    async def run():
        return await asyncio.gather(
            asyncio.to_thread(
                service._prepare_data,
                [("Temperature", "temp")],
                2,
                2,
                data_dirs,
            ),
            asyncio.to_thread(
                service._prepare_data,
                [("Temperature", "temp")],
                2,
                2,
                data_dirs,
            ),
        )

    first, second = asyncio.run(run())

    assert first[0].shape == second[0].shape
    assert first[1].shape == second[1].shape
