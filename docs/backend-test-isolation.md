# 后端测试隔离

测试应通过调用方实际使用的入口提供依赖替身，不覆盖全进程的认证、数据库、指标或科学计算模块，也不依靠文件执行顺序恢复状态。对模块属性的必要替换使用 pytest `monkeypatch` 或上下文管理器，使测试退出时恢复原值。

## 本轮修复范围

- `test_trained_model_predict_contract.py` 使用真实路由及其导入，通过 `request.app.state.training_inference_service` 提供推理替身，移除顶层 `sys.modules` 替换。
- `test_uploaded_training_contract.py` 的 session factory 和路由训练对象由 `monkeypatch` 自动恢复；文件夹具使用 `tmp_path`。
- `test_inference_legacy_official_weights.py` 使用真实推理和指标模块，注入无持久化缓存的替身，只在单个推理实例上提供合成输入，不查询生产缓存数据库。
- `test_training_dataset_loader.py` 使用环境中的真实 SciPy/sklearn，合成 NetCDF 全部写入 `tmp_path`；不替换全局依赖或在仓库内建立临时数据。
- `test_training_dataset_identity.py` 从当前 `Base.metadata` 创建临时 SQLite 表，包含队列、用户、标签与上传模型。上传模型的源码/摘要/归属经过真实查询；记录器等待执行准备完成，只替换子进程。HTTP 使用独立排队实例并在关闭客户端时释放引擎。日频请求断言 409 `dataset_retired`，活动三小时缺包仍断言 503 且无数据库副作用。

本轮只修复测试夹具与测试依赖隔离，没有改变应用装配、运行时权限、训练或预测行为。旧数据库迁移的专门测试继续保留；当前身份测试的数据库不是旧结构迁移夹具。

`test_netcdf_read_lock_contract.py` 的静态读取点检查跟随实际读取模块 `services/mars_data_service.py`；`inference_service.py` 调用该模块，不再直接构造 NetCDF Dataset。测试继续核对上传加载器、MOLA 与官方数据服务使用同一把进程级锁，并覆盖官方/上传各自及相互并发时 Dataset 打开区间不重叠。

## 组合运行

在 `AstraAtmos_backend/backend/` 使用工作区指定的 工作区约定的 Conda 解释器运行 pytest，启用 `--asyncio-mode=auto`，禁用 pytest 缓存，并为每次执行指定仓库外**新的**英文 `--basetemp`。重复使用已有目录会触发 pytest 清理，因此不得复用临时路径。

以下是传给该解释器的参数；最后的临时目录需由运行者换成新的绝对路径：

```text
-m pytest -q --tb=short -p no:cacheprovider --asyncio-mode=auto
  --basetemp <new-external-absolute-directory>
  tests/test_trained_model_predict_contract.py
  tests/test_uploaded_training_contract.py
  tests/test_inference_legacy_official_weights.py
  tests/test_training_dataset_loader.py
  tests/test_training_dataset_identity.py
  tests/test_training_queue_recovery.py
  tests/test_training_model_artifacts.py
  tests/test_training_tags.py
```

2026-10-10 本轮同会话组合为 **154 passed**，身份测试逐文件为 **41 passed**；五个相关文件换序组合为 **67 passed**，旧数据库迁移、日频停用、数据加载及测试集指标补充组合为 **37 passed**。这些范围有重叠，不累加为独立测试总数。全库 `--collect-only` 为 **2,173 tests collected**。组合覆盖临时数据库、训练队列、产物与标签，旧权重使用小型合成模型；未运行完整真实训练、精度验收或全量后端测试执行。依赖警告包括既有 NumPy、Pydantic 与 FastAPI 警告。
