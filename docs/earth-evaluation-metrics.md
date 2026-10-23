# Earth TO3 训练与评估指标

活动数据集仅为 `earth_merra2_3hourly_v1`。官方 DLinear 与独立契约上传模型共用同一评价实现；新三小时训练的 loss 和评价都使用球面 cell-area 权重，模型结构、归一化和优化器保持不变。日频仍返回 `dataset_retired`。

## 公式与单位

令 `y` 为反归一化后的参考 TO3、`p` 为反归一化后的预测 TO3（均为 DU），`e=p-y`，`n` 为本次聚合的全部元素数，`ε=1e-8 DU`。所有元素均计入分母，包括零值与近零真值；不屏蔽或裁剪百分比误差。三小时新训练与验证、测试、诊断指标使用发布 `lat_bounds` 推导的球面 cell-area 权重；历史 checkpoint 的 `forecast_origin_lead_grid_uniform` 保持逐格等权，不能重新解释。

| 指标 | English | 公式 | 单位 | 最优方向 |
| --- | --- | --- | --- | --- |
| MSE 均方误差 | Mean squared error | `sum(e²)/n` | DU² | 越低越好 |
| RMSE 均方根误差 | Root mean squared error | `sqrt(sum(e²)/n)` | DU | 越低越好 |
| MAE 平均绝对误差 | Mean absolute error | `sum(abs(e))/n` | DU | 越低越好 |
| R² 决定系数 | Coefficient of determination | `1-sum(e²)/sum((y-mean(y))²)` | 无量纲 | 越高越好，保留负值 |
| MAPE 平均绝对百分比误差 | Mean absolute percentage error | `100*mean(abs(e)/(abs(y)+ε))` | % | 越低越好 |
| SMAPE 对称平均绝对百分比误差 | Symmetric mean absolute percentage error | `100*mean(2*abs(e)/(abs(y)+abs(p)+ε))` | % | 越低越好 |

R² 的真值方差为零时，平方误差和为零返回 1，否则返回 0。这与 sklearn 默认 force-finite 的恒定真值规则一致；单个元素也采用同一规则。完全一致预测的五项误差为 0、R² 为 1。零真值与非零预测可能使 MAPE 很大，但不能改成 0；SMAPE 在 0–200% 内。对输入或结果的 NaN/Inf 明确拒绝。

Mars 官方 `demo3.py` 与上传 `user_model_runner.py` 已在 `_evaluate_metrics` 中用目标均值/尺度还原物理值，百分比公式同样使用加性 `1e-8` 和 `×100`。Earth 的 ε 单位明确为 DU，不能照搬训练标准化 MSE 或将 Mars 物理单位直接当作 DU。本改动保留 Mars 既有算法与顶层字段解析。

## 聚合与内存

上传模型 eval 需通过批次独立性和模型状态不变检查，使训练完整测试、诊断与单起点回测共用逐样本预测定义。检查覆盖单独/合批、顺序、其他成员、重复调用和任务实际 batch size；checkpoint 加载训练权重后再次执行。固定探针采用 float32 容差，具体模型仍需实际数据验证，见[上传模型契约](earth-3hourly-uploaded-model.md)。

新训练指标使用 `forecast_origin_lead_cell_area_weighted`：完整分区内每个「预测起点 × 提前步」的空间格点贡献按 `sin(lat_north)-sin(lat_south)` 加权，权重由已验证发布 `lat_bounds` 提供并归一化到完整 240×480 网格。它不是中心纬度 `cos(lat)` 近似，也不是独立日期均值。历史 checkpoint 的 `forecast_origin_lead_grid_uniform` 不变；validation 与 test 分开保存，页面训练结果及实验矩阵默认读取 `metrics.splits.test.overall`，结果页另外标注验证集。

总体、每步 `by_lead` 和累计 `by_horizon` 都支持六项指标。`horizon` 使用任务配置（1–240 步），每步 3 小时；累计汇总为每 24 小时的里程碑及最终提前量，例如 9 步对应累计 24/27 小时，25 步对应 24/48/72/75 小时。总体和累计 RMSE 从合并平方误差计算，R² 从合并真值统计量计算，不能平均批次或逐步 RMSE/R²。

`ErrorAccumulator` 使用 float64 加权误差和及加权 Chan 目标矩统计。按批次、24×48 空间块和尾批累计，内存仅随 horizon 增长；不保存完整测试集预测/真值，不重新物化滑窗。历史预测读取 checkpoint 的真实 aggregation/policy，新的起点回测使用 `user_forecast_origin_lead_cell_area_weighted`，旧 checkpoint 保留 `user_forecast_origin_lead_grid_uniform`，仅评价本次起点的预测窗口，不代表完整测试集。

新 `earth_raw_utc_timeline_v1` checkpoint 的训练指标、诊断 test dataset 与历史预测共享同一任务分区边界。每个可用历史起点的完整输入窗口和目标窗口都必须位于同一分区；跨分区窗口不会被标记为该起点所属分区后继续计算，而是拒绝为 `earth_prediction_origin_out_of_range`。因此历史预测响应的 `origin_split` 表示完整窗口分区，官方 DLinear 与上传模型的窗口集合和边界口径完全一致。没有 task split 的旧 checkpoint 继续按已发布 manifest 兼容逻辑读取。

## 产物与兼容

新训练保存 `earth_training_metrics_3hourly_v3`，checkpoint 外层仍为 `astraatmos_earth_forecast_checkpoint_3hourly_v1`。指标块含 `target=TO3`、物理基准 `unit=DU`、`metric_units`（MSE 为 `DU^2`，R² 为 `1`，百分比为 `%`）、`metric_policy`（含 `spatial_weighting=spherical_cell_area`、公式和归一化）、aggregation，以及 validation/test 的窗口数和三种聚合结果。每步另保存 `target_statistics={count,mean,m2,weight_sum}`（v3；v2 不含 `weight_sum`），用于严格核验 R² 及累计统计。v1/v2 的 grid-uniform policy 保持可读、可预测和可比较。

新 v3 读回时要求全部六项为真正有限数值（拒绝 bool、字符串、缺字段、NaN/Inf），验证单位、公式策略、空间权重策略、步顺序、MSE/RMSE 一致性、非负误差、R²/SMAPE 范围与累计统计。完成门禁要求 checkpoint 与任务冻结的版本和 aggregation 严格一致，再隔离严格 CPU 重建模型；新任务使用完整 v3，旧 v2 队列任务继续使用完整 v2 等权契约。任务 metrics 直接复制已核验 checkpoint，不通过日志重建指标。

新任务由服务器同时冻结 `_earth_metrics_schema=earth_training_metrics_3hourly_v3` 与 `_earth_metrics_aggregation=forecast_origin_lead_cell_area_weighted`，客户端不能提供这些字段。排队恢复把同一契约作为 spec 顶层 `metrics_contract` 传给子进程；数据准备前核对合法版本及口径，训练 loss、验证/测试统计、保存、完成验收和后续读取均遵循该契约。旧 v2 队列任务没有 aggregation 标记时由 v2 唯一推导等权，执行仍产生 v2；真正无标记的旧执行 spec 保持等权兼容。未知版本、v2/面积加权或 v3/等权混搭在执行前拒绝，不到结束才报错。任务读取与服务重启恢复也严格核对冻结标记；不能仅凭日志 `Model saved:` 完成任务，旧 checkpoint 不被改写。

v2 checkpoint 的每步元素数必须等于保存的分区窗口数 × 240 × 480，缺块/漏尾批不能被当作完整分区指标。历史预测中的统计量属于本次窗口，不用于补齐旧 checkpoint。

旧 `earth_training_metrics_3hourly_v1` 只要求原有 RMSE/MAE，继续读取、严格重载及历史预测。缺失 MSE/R²/MAPE/SMAPE 显示“未提供”，没有 0 占位，不计算或伪造缺失训练指标，不自动训练/推理，不改写旧文件。用户主动运行历史预测时，本次窗口可按新公式获得六项指标，仍不补写旧完整测试集评价。

比较只读取已保存且严格核验的完整 test 指标，复核发布身份、窗口、测试 UTC 范围及数量，遵守原有用户权限。兼容的 v1/v2 可以比较共有指标；新指标的缺失值排除排名及曲线，R² 按原值降序，不能取绝对值。Earth/Mars 混合比较仍拒绝。

预测页的地球多模型输出窗口仅作只读展示，不提供自选步长；选择至少两个已完成且权重可用的三小时模型即可提交 `{task_ids}`。任务摘要未提供共同 horizon 时仍可请求，服务器以 checkpoint 为准校验窗口、测试范围和指标口径，不兼容返回 `409 earth_comparison_incompatible` 并在页面显示原因。

指标版本错误导致的已保存任务失败只能走[显式恢复门禁](earth-metrics-task-recovery.md)：完整核验真实 checkpoint、数据身份、任务划分、归一化、源码与训练参数，隔离模型重载，SQLite backup API 备份并保存原失败记录后才迁移指定任务。普通读取、服务重启和队列启动不会自动把 v2 声明改成 v3。

## 科研导出与验证

当前科研逐步曲线契约只支持 RMSE/MAE/R²/SSIM；Earth 接入实际已有的 RMSE/MAE/R²。MSE/MAPE/SMAPE 可以在页面比较和逐步曲线查看，但不开放其科研导出按钮。旧模型仍可导出已有 RMSE/MAE；导出某指标只选择逐步值完整的模型，至少两个。快照附带指标 schema/单位/公式策略，导出不再推理，原有身份与权限复核保持有效。详见[科研导出](research-figure-export.md)。

回归入口：`tests/test_earth_metrics_v2.py`（手算、完全一致、负 R²、恒定/零/近零真值、分批/分块/尾批、可配置 horizon、新旧 checkpoint），三小时 runner/uploaded/prediction service/routes 以及 `frontend/src/pages/ModelTrainingPage/earthMetrics.test.js`。后端使用项目指定 conda 解释器，临时目录放仓库外；前端执行 `node --test` 和 `npm run build`。训练/HTTP/浏览器验证使用合成发布及指标，不代表真实两年训练或真实预测精度验收。

2026-10-08 实际验证覆盖六项公式、完整网格计数门禁、服务重启、新旧三小时产物、官方/上传 71→9 合成训练与主动预测、无推理比较、HTTP 序列化、权限及科研导出。前端训练/预测相关 355 项测试与生产构建通过；Playwright 使用合成任务核对桌面和 390px 指标显示、9 步/27 小时、负 R² 排名和旧产物缺项。旧日频 `test_earth_training_artifact.py` 中 40 项仍在 fixture 构造阶段被现有 `dataset_retired` 拒绝；用原始 HEAD 产物模块在内存中运行代表用例同样失败，未修改工作区源码或恢复日频能力。这些旧测试不是本轮三小时指标回归通过的范围。

后端按文件去重的相关验证共 579 项通过：六项指标 110、三小时 artifact 39、training contract 30、runner 12、training service 12、uploaded 80、task split 72、71→9 pipeline 2、prediction routes 28、prediction service 55、日频停用及科研导出共 39。分会话执行以避免已有测试模块桩相互污染；重复验证不重复计数。运行后的后端 health、生产页面与数据目录代理返回 200。既有 NumPy/FastAPI/Pydantic 提示和前端大 bundle 警告仍存在。

## 本轮修改文件

以下仅列评价指标及必要配套涉及的文件；工作区已有的任务划分修改被保留，同文件可以同时包含已有修改。

- [earth_training_artifact.py](../AstraAtmos_backend/backend/services/earth_training_artifact.py)
- [earth_training_contract.py](../AstraAtmos_backend/backend/services/earth_training_contract.py)
- [earth_prediction_service.py](../AstraAtmos_backend/backend/services/earth_prediction_service.py)
- [training_service.py](../AstraAtmos_backend/backend/services/training_service.py)
- [research_export_sources.py](../AstraAtmos_backend/backend/services/research_export_sources.py)
- [earth_predict.py](../AstraAtmos_backend/backend/schemas/earth_predict.py)
- [test_earth_metrics_v2.py](../AstraAtmos_backend/backend/tests/test_earth_metrics_v2.py)
- [test_earth_metrics_pipeline.py](../AstraAtmos_backend/backend/tests/test_earth_metrics_pipeline.py)
- [test_earth_3hourly_artifact.py](../AstraAtmos_backend/backend/tests/test_earth_3hourly_artifact.py)
- [test_earth_3hourly_training_runner.py](../AstraAtmos_backend/backend/tests/test_earth_3hourly_training_runner.py)
- [test_earth_3hourly_training_service.py](../AstraAtmos_backend/backend/tests/test_earth_3hourly_training_service.py)
- [test_earth_3hourly_uploaded.py](../AstraAtmos_backend/backend/tests/test_earth_3hourly_uploaded.py)
- [test_earth_3hourly_prediction_service.py](../AstraAtmos_backend/backend/tests/test_earth_3hourly_prediction_service.py)
- [test_earth_3hourly_prediction_routes.py](../AstraAtmos_backend/backend/tests/test_earth_3hourly_prediction_routes.py)
- [earthMetricMeta.js](../frontend/src/utils/earthMetricMeta.js)
- [experimentCenterModel.js](../frontend/src/pages/ModelTrainingPage/experimentCenterModel.js)
- [ExperimentResultPanel.jsx](../frontend/src/pages/ModelTrainingPage/ExperimentResultPanel.jsx)
- [experimentMatrixModel.js](../frontend/src/pages/ModelTrainingPage/experimentMatrixModel.js)
- [earthMetrics.test.js](../frontend/src/pages/ModelTrainingPage/earthMetrics.test.js)
- [experimentResultStructure.test.js](../frontend/src/pages/ModelTrainingPage/experimentResultStructure.test.js)
- [experimentCenter.css](../frontend/src/pages/ModelTrainingPage/experimentCenter.css)
- [earthPredictModel.js](../frontend/src/pages/PredictPage/earthPredictModel.js)
- [EarthPredictPanel.jsx](../frontend/src/pages/PredictPage/EarthPredictPanel.jsx)
- [EarthCompareWorkspace.jsx](../frontend/src/pages/PredictPage/EarthCompareWorkspace.jsx)
- [PredictSidebar.jsx](../frontend/src/pages/PredictPage/PredictSidebar.jsx)
- [singleModelWorkbench.css](../frontend/src/pages/PredictPage/singleModelWorkbench.css)
- [CompareTrainingModelsPanel.jsx](../frontend/src/pages/PredictPage/CompareTrainingModels/CompareTrainingModelsPanel.jsx)
- [compareTrainingModelsData.js](../frontend/src/pages/PredictPage/CompareTrainingModels/compareTrainingModelsData.js)
- [zh.js](../frontend/src/i18n/zh.js)
- [en.js](../frontend/src/i18n/en.js)
- [README.md](../README.md)
- [earth-training.md](earth-training.md)
- [earth-merra2-3hourly.md](earth-merra2-3hourly.md)
- [earth-3hourly-uploaded-model.md](earth-3hourly-uploaded-model.md)
- [experiment-center.md](experiment-center.md)
- [research-figure-export.md](research-figure-export.md)
- [earth-evaluation-metrics.md](earth-evaluation-metrics.md)
