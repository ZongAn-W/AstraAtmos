"""NetCDF / HDF5 读取串行化锁

netCDF4 背后的 HDF5 C 库不是线程安全的：同一进程内多个线程并发读取同一批
NetCDF 文件时，属性（attribute）读取可能失败，典型报错为
``NetCDF: Can't open HDF5 attribute``（预测接口表现为 HTTP 500）。

平台在 ``asyncio.to_thread`` 线程池中执行预测与评估，因此所有 NetCDF 读取必须
取得同一把锁，覆盖：

- 官方模型推理：``services/inference_service.py``
- 上传模型推理与训练数据加载：``training_backbones/user_model_runner.py``
- MOLA 地形资源：``training_backbones/mola_topography.py``

锁存放在 ``netCDF4`` 模块属性上，而不是本模块的全局变量里：``config.py`` 会向
``sys.path`` 追加后端目录，同一份源码可能以 ``services.netcdf_read_lock`` 和
``netcdf_read_lock`` 两种模块名各加载一次；只有以已被所有读取方共享的
``netCDF4`` 模块作为锚点，才能保证全进程范围内只有一把锁。
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

import netCDF4

_LOCK_ATTRIBUTE = "_astraatmos_netcdf_read_lock"


def _get_lock() -> threading.RLock:
    lock = getattr(netCDF4, _LOCK_ATTRIBUTE, None)
    if lock is None:
        lock = threading.RLock()
        setattr(netCDF4, _LOCK_ATTRIBUTE, lock)
    return lock


NETCDF_READ_LOCK = _get_lock()
"""全局唯一的 NetCDF 读锁；需要按对象断言「各路径共用同一把锁」时引用它。"""


@contextmanager
def netcdf_read_lock():
    """串行化一段 NetCDF / HDF5 读取。

    典型用法（锁与 Dataset 合并为一条语句，读取区间与持锁区间完全一致）::

        with netcdf_read_lock(), netCDF4.Dataset(path) as dataset:
            ...

    可重入，因此同一线程内的嵌套读取不会死锁。
    """
    with _get_lock():
        yield
