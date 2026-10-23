# 测试索引

从原 README「测试与构建」拆出的逐测试文件说明，用于定位回归入口；测试命令见 [README 测试与构建](../README.md#测试与构建)。这些说明记录编写时的验证范围，不代表当前全量通过。

共用单模型预测的适配、展示范围与交互回归为 `singleModelAdapters.test.js`、`predictionPresentation.test.js`、`predictionDisplayModel.test.js` 及既有请求/缓存隔离测试；合成浏览器脚本为 [browser-check.js](../scripts/audit/shared-prediction/browser-check.js)。脚本只在隔离浏览器中替换 API，检查桌面与 390px、主题/色带/精度、行星/任务/时间步/全屏、错误重试及旧响应清理；不启动真实训练。后端使用工作区规定的 工作区约定的 Conda 解释器逐文件执行预测/诊断/身份回归，未标记的 async 契约文件加 `--asyncio-mode=auto`，并为每个文件指定仓库外新英文 `--basetemp`。实际范围与已知 Mars 坐标失败见[验证入口与边界](shared-single-model-prediction.md#验证入口与边界)。

科研图导出契约与文件验证为 `tests/test_research_export.py`，Earth 三小时路由另核对第 24 步导出无再次推理、DU 值与 UTC 时间。前端测试为 `frontend/src/pages/PredictPage/researchExportModel.test.js`。合成布局样例脚本位于 `scripts/audit/research-export/generate-fixtures.py`，输出目录设在仓库外；命令与验证边界见[科研图导出](research-figure-export.md)。

测试覆盖数据读取与对齐、模型接入、训练配置、预测步长、缓存隔离、请求一致性及前端交互逻辑。运行所需数据或依赖以各测试为准。首页交互工具函数测试为 `frontend/src/pages/HomePage/homePointerInteraction.test.js`，覆盖指针坐标归一化、分层视差幅度与边界、鼠标/触控笔/触屏判断、`prefers-reduced-motion`、地球水平与垂直旋转限制、惯性阻尼与键盘按键映射。

训练草稿的转换序列与最终提交内容由 `frontend/src/pages/ModelTrainingPage/trainingDraftSession.test.js` 验证；[浏览器草稿回归](../scripts/audit/training-draft/browser-check.js) 在隔离会话替换全部 API 与 WebSocket，检查两个 Mars 数据集往返、同模型分场景参数、重复选择、窗口/步长恢复、迁移撤销、复制/新建重置、390px 和拦截提交载荷，不创建真实训练任务。运行方式见[实验中心草稿约定](experiment-center.md#训练草稿切换与参数保留)。

训练执行准备与恢复回归为 `tests/test_training_queue_recovery.py`：使用临时 SQLite、真实任务/权重/上传模型查询和子进程替身，覆盖官方/上传 Mars 的两种迁移来源、新建队列与实例重建、依赖缺失/失效/权限撤销、FIFO、队列取消及失败后继续。Earth 的启动参数测试改为通过同一准备入口，冻结划分/源码和产物门禁仍由三小时与任务划分测试验证。使用工作区规定的 conda 解释器逐文件运行，并指定仓库外新的英文 `--basetemp`；这些回归不代表真实模型训练或精度验收。

NetCDF 并发读锁由 `tests/test_netcdf_read_lock_contract.py` 覆盖：断言官方数据服务、上传模型与 MOLA 地形三处共用同一把进程级锁，官方/上传交叉并发时 `Dataset` 打开区间不重叠，并静态检查每个 `Dataset` 构造点都包在 `netcdf_read_lock()` 内；Earth ETOPO 的新读取点由地形测试覆盖。`tests/test_inference_netcdf_thread_safety.py` 覆盖官方模型数据准备路径的串行化。

预测体积缓存由 `tests/test_prediction_volume_cache.py` 覆盖：断言 `return_scaled_volume` 切窗与逐样本展开路径逐元素一致（bitwise）、测试集分区滑窗与展开切片一致、第二次调用命中缓存返回同一对象、文件变化后缓存签名失效。`tests/test_uploaded_model_ls_inference.py` 的夹具已改为按 `ScaledVolume` 提供数据，继续断言各上传推理路径收到的 Ls 与历史窗口一致。

Mars 检查点通道契约回归见 `tests/test_mars_checkpoint.py`：覆盖无辅助输入、U、U/V 与 U/V/T，实际运行官方及上传模型 runner，校验完整通道顺序、统计量数量与数值，并验证两条预测数据路径应用检查点统计量及缓存隔离。需从 `AstraAtmos_backend/backend/` 用项目规定的 工作区约定的 Conda 解释器执行 `-m pytest tests/test_mars_checkpoint.py`，临时目录使用新的纯英文路径。

OpenMARS 年份边界回归见 `tests/test_openmars_year_boundaries.py`，覆盖单文件 Ls 回绕、多文件的全局/文件内 segment 索引、完整输入与目标窗口归属、真实 block 定位、三条预测路径以及 OpenMARS 缓存策略隔离；本地存在跨年原始文件时还会只读取 Ls 与字段形状作轻量检查。需同时运行 `tests/test_mars_ls_window.py` 和 `tests/test_training_dataset_loader.py`，核对周期定位与原始 MCD 加载行为。

Mars 数据身份回归见 `tests/test_mars_dataset_identity.py`：直接断言每种数据集在数据加载、任务快照、检查点与预测复核四个阶段的指纹完全相同，验证双目录绑定、两类 runner 的预测入口在文件变化后返回 409 且不访问缓存/窗口/推理，并区分新格式与 legacy。`tests/test_training_model_artifacts.py` 验证父进程从检查点重建相同快照，并拒绝身份不完整的训练产物。

Mars 坐标回归见 `tests/test_mars_prediction_coordinates.py`：以北高南低的纬度梯度验证两种数据集、官方/上传模型与新/旧权重，逐项检查三类空间场的坐标和值，另覆盖缺失/非法坐标及缓存策略。前端 `frontend/src/pages/PredictPage/marsPredictionGrid.test.js` 验证真实轴的绘图位置、非均匀经度、三维坐标传递和无坐标时不生成替代标签。

数据集注册相关测试为 `tests/test_dataset_identity.py`、`tests/test_dataset_registry.py`、`tests/test_dataset_routes.py`、`tests/test_training_dataset_identity_migration.py` 和 `tests/test_training_dataset_identity.py`；地球总览与分析为 `tests/test_earth_overview_service.py`、`tests/test_earth_overview_routes.py`、`tests/test_earth_research_service.py` 与 `tests/test_earth_research_routes.py`。共用工作台前端测试位于 `frontend/src/pages/DataOverviewPage/workbench/` 与 `frontend/src/pages/DataOverviewPage/EarthOverview/`。`tests/conftest.py` 提供显式引用的临时 Earth 发布 fixture（`earth_release`、`earth_spatial_release`、`earth_global_release`），不读取生产数据；该文件在 Windows 上把 `tempfile` 临时目录的 POSIX 权限位从 `0o700` 放宽到 `0o777`（POSIX 行为不变），否则受限文件策略会拒绝写入 `tmp_path`。这些测试需要新的纯英文临时目录（`--basetemp`）。预测契约、旧权重和训练数据加载测试已移除全局依赖模块替换；上传训练测试的依赖替身由 fixture 自动恢复，可在同一会话组合运行。训练身份夹具使用当前 ORM 表结构、真实临时用户/上传包，并明确区分日频 409 `dataset_retired` 与三小时缺包 503；历史无身份行的迁移仍由独立迁移测试覆盖。组合命令和本轮范围见[后端测试隔离](backend-test-isolation.md)，全库收集成功不等同于全量执行通过。

三小时发布协议与目录测试为 `tests/test_earth_3hourly_data_contract.py`、`tests/test_earth_3hourly_registry.py`，其中包含 `fingerprint_rule` 规则分派与 fail-closed、真实发布 manifest 副本上的 7 条身份不变量、既有发布包指纹冻结复算，以及不带 `fingerprint_rule` 的 legacy manifest 走完整注册校验的回归；训练契约、惰性窗口、checkpoint 和 runner/service 测试见 `tests/test_earth_3hourly_training_contract.py`、`tests/test_earth_3hourly_training_data.py`、`tests/test_earth_3hourly_artifact.py`、`tests/test_earth_3hourly_training_runner.py`、`tests/test_earth_3hourly_training_service.py`。先运行 smoke，再分文件运行训练链路回归，命令与实际范围见 [三小时数据构建与训练](earth-merra2-3hourly.md#官方-dlinear-后端训练)。

三小时历史回测测试为 `tests/test_earth_3hourly_prediction_time.py`、`tests/test_earth_3hourly_prediction_service.py`、`tests/test_earth_3hourly_prediction_routes.py`，覆盖 UTC 时间窗、完整 240×480 场及发布参考值、物理指标、缓存身份、稳定错误码、认证与 Earth/Mars 隔离。六项公式/流式/新旧契约与服务重启恢复由 `tests/test_earth_metrics_v2.py` 验证，官方/上传 71→9 步合成训练和旧产物预测/比较由 `tests/test_earth_metrics_pipeline.py` 验证；前端为 `frontend/src/pages/ModelTrainingPage/earthMetrics.test.js`。使用固定解释器及仓库外新的英文临时目录，详见[指标专题](earth-evaluation-metrics.md)和[三小时回测验证](earth-merra2-3hourly.md#三小时回测验证)。

三小时上传模型契约、隔离 dry-run、任务/源码冻结、训练与严格 checkpoint 重载、历史回测及兼容回归见 `tests/test_earth_3hourly_uploaded.py` 和 `tests/test_earth_topography*.py`；真实地形集成冒烟脚本为 `scripts/smoke_earth_topography.py`，须明确指定新的仓库外输出目录。完整包只读集成见 `tests/test_earth_3hourly_uploaded_readonly.py`，须显式设置 `ASTRAATMOS_TEST_EARTH_3HOURLY_PACKAGE`。本轮真实冒烟只读原发布包并创建三天独立子包，不训练完整两年或评价预测精度。
