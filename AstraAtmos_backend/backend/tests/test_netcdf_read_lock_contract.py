"""NetCDF 读锁契约：官方模型与上传模型路径必须共用同一把可重入锁。

netCDF4/HDF5 不是线程安全的，并发读取会间歇性报
``NetCDF: Can't open HDF5 attribute``（预测接口表现为 500）。平台在
``asyncio.to_thread`` 线程池中执行预测，因此两条推理路径都必须串行化读取：

- 官方模型：``services/mars_data_service.py``（由 inference_service 调用）
- 上传模型：``training_backbones/user_model_runner.py`` 与
  ``training_backbones/mola_topography.py``
"""

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from services.netcdf_read_lock import NETCDF_READ_LOCK, netcdf_read_lock

BACKEND_DIR = Path(__file__).resolve().parents[1]

# 每种模块里「取锁」与「构造 Dataset」都应逐点配对。
LOCKED_READER_MODULES = (
    "training_backbones/user_model_runner.py",
    "training_backbones/mola_topography.py",
    "services/mars_data_service.py",
)


class _FakeOpenMarsDataset:
    """记录开关次数的最小 OpenMars 数据集替身。

    ``_load_openmars`` 使用 ``with ... as dataset``，关闭走 ``__exit__``，
    因此打开/关闭区间以 ``__enter__`` / ``__exit__`` 为准。
    """

    opened = 0
    closed = 0
    _counter_guard = threading.Lock()

    def __init__(self, path, *args, **kwargs):
        self.variables = {
            "o3col": np.zeros((6, 36, 72), dtype=np.float32),
            "Ls": np.linspace(0.0, 5.0, 6, dtype=np.float32),
        }

    def __enter__(self):
        with _FakeOpenMarsDataset._counter_guard:
            _FakeOpenMarsDataset.opened += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with _FakeOpenMarsDataset._counter_guard:
            _FakeOpenMarsDataset.closed += 1
        return False

    def close(self):
        # 正常路径下由 __exit__ 关闭；保留该方法以免替身与真实对象接口不一致。
        return None


@pytest.fixture(autouse=True)
def _reset_fake_dataset_counters():
    _FakeOpenMarsDataset.opened = 0
    _FakeOpenMarsDataset.closed = 0
    yield


def test_all_reader_paths_share_one_reentrant_lock():
    from services import mars_data_service
    from training_backbones import mola_topography, user_model_runner

    assert user_model_runner.netcdf_read_lock is netcdf_read_lock
    assert mars_data_service.netcdf_read_lock is netcdf_read_lock
    assert mola_topography.netcdf_read_lock is netcdf_read_lock
    # 可重入：同一线程嵌套取锁不会死锁。
    with netcdf_read_lock():
        with netcdf_read_lock():
            pass
    assert isinstance(NETCDF_READ_LOCK, type(threading.RLock()))


def test_lock_is_process_wide_even_if_module_is_loaded_twice():
    """config.py 会改 sys.path，同一模块可能被加载两次，锁仍须是同一把。"""
    import importlib.util

    module_path = BACKEND_DIR / "services" / "netcdf_read_lock.py"
    spec = importlib.util.spec_from_file_location(
        "_netcdf_read_lock_loaded_again", module_path
    )
    second_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(second_module)

    assert second_module.NETCDF_READ_LOCK is NETCDF_READ_LOCK

    acquired = threading.Event()
    released = threading.Event()

    def hold_second_module_lock():
        with second_module.netcdf_read_lock():
            acquired.set()
            released.wait(timeout=5.0)

    holder = threading.Thread(target=hold_second_module_lock, daemon=True)
    holder.start()
    assert acquired.wait(timeout=5.0) is True

    # 第二份模块持有的锁若与本模块不是同一把，这里就能立刻取到。
    got_it = NETCDF_READ_LOCK.acquire(timeout=0.3)
    released.set()
    holder.join(timeout=5.0)

    assert got_it is False, "两份模块各自持有不同的锁对象"


def test_uploaded_model_loader_waits_for_the_shared_lock(monkeypatch, tmp_path):
    from training_backbones import user_model_runner

    openmars_dir = tmp_path / "openmars"
    openmars_dir.mkdir()
    (openmars_dir / "openmars_ozo_my27.nc").write_bytes(b"stub")

    monkeypatch.setattr(user_model_runner.netCDF4, "Dataset", _FakeOpenMarsDataset)

    loader_done = threading.Event()
    loader_error: list[BaseException] = []

    def run_loader():
        try:
            user_model_runner._load_openmars(
                openmars_dir,
                require_ls=True,
                require_coordinates=False,
            )
        except BaseException as exc:  # noqa: BLE001 - 转发到主线程断言
            loader_error.append(exc)
        finally:
            loader_done.set()

    with netcdf_read_lock():
        loader = threading.Thread(target=run_loader, daemon=True)
        loader.start()
        # 锁被本线程持有时，上传模型读取不得打开任何 Dataset。
        assert loader_done.wait(timeout=0.5) is False
        assert _FakeOpenMarsDataset.opened == 0

    assert loader_done.wait(timeout=10.0) is True, "读取在锁释放后仍未完成"
    loader.join(timeout=5.0)

    assert loader_error == []
    assert _FakeOpenMarsDataset.opened == 1


@pytest.mark.parametrize("readers", [
    ("uploaded", "uploaded"), ("official", "official"), ("uploaded", "official"),
])
def test_model_loaders_serialize_concurrent_reads(monkeypatch, tmp_path, readers):
    """官方与上传读取各自及相互并发时，Dataset 打开区间均不得重叠。"""
    from services import mars_data_service
    from training_backbones import user_model_runner

    openmars_dir = tmp_path / "openmars"
    openmars_dir.mkdir()
    (openmars_dir / "openmars_ozo_my27.nc").write_bytes(b"stub")

    guard = threading.Lock()
    state = {"active": 0, "peak": 0}

    class OverlapDetectingDataset(_FakeOpenMarsDataset):
        def __enter__(self):
            with guard:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.05)
            return super().__enter__()

        def __exit__(self, exc_type, exc_value, traceback):
            with guard:
                state["active"] -= 1
            return super().__exit__(exc_type, exc_value, traceback)

    monkeypatch.setattr(
        user_model_runner.netCDF4, "Dataset", OverlapDetectingDataset
    )

    errors: list[BaseException] = []

    def run_loader(reader):
        try:
            if reader == "official":
                mars_data_service._load_openmars(openmars_dir)
            else:
                user_model_runner._load_openmars(
                    openmars_dir,
                    require_ls=True,
                    require_coordinates=False,
                )
        except BaseException as exc:  # noqa: BLE001 - 转发到主线程断言
            errors.append(exc)

    threads = [threading.Thread(target=run_loader, args=(reader,), daemon=True)
               for reader in readers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15.0)

    assert not any(thread.is_alive() for thread in threads), "并发读取未在锁内完成"
    assert errors == []
    assert state["peak"] == 1, "两个 Dataset 读取区间发生重叠"
    assert _FakeOpenMarsDataset.opened == 2
    assert _FakeOpenMarsDataset.closed == 2


@pytest.mark.parametrize("relative_path", LOCKED_READER_MODULES)
def test_every_dataset_construction_is_guarded_by_the_lock(relative_path):
    lines = (BACKEND_DIR / relative_path).read_text(encoding="utf-8").splitlines()

    lock_calls = sum(1 for line in lines if "netcdf_read_lock()" in line)
    dataset_constructions = sum(
        1 for line in lines if "netCDF4.Dataset(" in line or "nc.Dataset(" in line
    )

    assert dataset_constructions > 0, f"{relative_path} 未找到 NetCDF 读取点，契约已失效"
    assert lock_calls == dataset_constructions, (
        f"{relative_path} 有 {dataset_constructions} 处 Dataset 构造，但只有 {lock_calls} 处取锁；"
        "新增 NetCDF 读取必须包在 netcdf_read_lock() 内"
    )


def test_prediction_routes_log_a_traceback_for_unexpected_failures():
    """500 兜底分支必须留下堆栈，否则线上只能看到一句 detail。"""
    lines = (
        BACKEND_DIR / "routers" / "predict.py"
    ).read_text(encoding="utf-8").splitlines()

    generic_handlers = [
        index for index, line in enumerate(lines) if line.strip() == "except Exception as e:"
    ]
    assert generic_handlers, "predict.py 未找到兜底异常分支"

    for index in generic_handlers:
        following = "\n".join(lines[index + 1:index + 4])
        assert "logger.exception(" in following, (
            f"routers/predict.py:{index + 1} 的兜底分支缺少 logger.exception，"
            "500 将不会记录堆栈"
        )
