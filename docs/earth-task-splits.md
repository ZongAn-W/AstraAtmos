# 地球训练任务级数据划分

适用于唯一活动数据集 `earth_merra2_3hourly_v1`，官方 DLinear 与独立 v1 契约上传模型共用策略 `earth_raw_utc_timeline_v1`。日频数据集继续停用。本功能不修改发布 NetCDF、manifest、发布 `split` 标记、SHA 或数据集 fingerprint。

## 请求与计算口径

训练页“训练策略”可编辑三个百分比，默认 70% / 20% / 10%。超参数请求以 0–1 数值发送 `train_ratio`、`validation_ratio`、`test_ratio`。每项必须为有限 JSON 数值且大于 0，合计为 1；共享校验允许 `1e-6` 的数值表示误差。字符串、布尔、null、NaN、Inf 和错误总和返回 `422 invalid_earth_training_parameters`。Earth 的验证、早停与 checkpoint 契约要求非空验证集，不支持比例为 0；Mars 原有规则不变。

比例针对完整发布的**原始 UTC 三小时时间轴**，不针对滑窗、空间块或训练批次。计算每项 `N × ratio / sum(ratios)`，先向下取整，再按小数余数从大到小分配剩余步；余数相同按 train、validation、test 顺序分配。除浮点总和容差外不缩放比例，不预留最小分区后重新分配。按这些整数步数从头到尾连续划分三个互斥区间，完整覆盖时间轴。边界可以位于日内，不要求完整 UTC 日。

随后在每个区间内独立生成窗口。分区原始索引为半开区间 `[raw_start, raw_end)`，UTC 日期边界包含首尾实际时间戳；窗口数为 `step_count - window - horizon + 1`。完整输入和目标均属于同一区间，不能跨界、随机划分窗口或向相邻分区借数据。任何区间少于 `window + horizon` 步，会在数据库任务写入、上传模型执行和排队前返回具体分区、实际步数及最低要求；不会为凑窗口调整比例。

完整两年 5848 步、默认 56→24 的新任务为：

| 分区 | 原始索引（半开） | UTC 首尾 | 时间步数 | 时间窗口数 |
| --- | --- | --- | --- | --- |
| train | `[0,4094)` | 2020-01-01T01:30:00Z → 2021-05-26T16:30:00Z | 4094 | 4015 |
| validation | `[4094,5263)` | 2021-05-26T19:30:00Z → 2021-10-19T19:30:00Z | 1169 | 1090 |
| test | `[5263,5848)` | 2021-10-19T22:30:00Z → 2021-12-31T22:30:00Z | 585 | 506 |

这些值是确定性分配的计算示例，不代表已执行真实两年训练。页面按目录时间轴、当前比例及窗口预览；任务参数详情和实验矩阵展示服务器保存的边界与计数。设置默认值、Earth/Mars 场景草稿、复制配置沿用原机制。复制配置创建新任务，只复制请求参数，不复制原任务的服务器固定划分身份。若 Mars 默认值的验证比例为 0，进入或新建 Earth 实验使用 70/20/10。

## 任务身份与 normalization

发布身份保留在任务的 `dataset_id/version/fingerprint/snapshot`。其中 snapshot 的 `splits` 始终是发布 manifest 分区，不替换成任务分区。

服务器在任务 `hyperparameters._earth_task_split` 保存策略版本、请求比例、完整时间步数、window/horizon 和三个 `ranges`，并独立保存 `_earth_task_split_policy=earth_raw_utc_timeline_v1`。每个 range 包含 `raw_start/raw_end`、`date_start/date_end`、`step_count/window_count`。这些是服务端内部字段，不接受客户端设置；排队 spec 的独立 `task_split` 传递同一固定副本。重启恢复会对照发布的原始时间轴和已保存请求严格校验，不能重新选择边界。策略标记或 `_earth_metrics_schema`（v2/v3）及 `_earth_metrics_aggregation` 均要求完整冻结划分；缺失/损坏的划分、未知标记或版本不一致返回 409，不会降级为发布 manifest。新任务冻结 v3 面积加权；此前已创建、只有 v2 标记和完整划分的任务继续读取，排队执行仍使用 v2 等权，无需迁移。版本与 aggregation 严格对应，未知或混搭在执行前拒绝，见[指标契约](earth-evaluation-metrics.md)。

仅以当前任务完整 train 区间拟合逐通道人口均值/标准差（ddof=0、epsilon=1e-6），保留逐通道最多八步的有界读取、连续性、有效掩码及有限值检查。验证、测试与预测复用保存值，不重新拟合。normalization 同时保存 UTC 拟合边界、步数及完整任务划分策略。共享统计量按数据身份、train 实际 UTC 边界/步数、通道顺序及方法版本匹配，并将当前任务的完整冻结划分绑定回返回值。训练缓存覆盖完整已归一化时间轴，按数据、通道、归一化内容、布局和缓存版本寻址；仅修改窗口或 validation/test 边界而 train 边界不变时可以共享缓存字节，窗口索引与 checkpoint 划分仍分别严格校验。train 实际边界变化必须重新拟合并建立另一缓存身份，详见[训练准备缓存](earth-preparation-cache.md)。预测 LRU 的 task ID 和 checkpoint SHA 隔离不同策略、边界及统计量，命中前仍严格校验产物和当前发布。

## Checkpoint、预测与比较

新 checkpoint 继续使用三小时 artifact schema；`training_contract.split_policy` 声明 `earth_raw_utc_timeline_v1`，`task_split` 保存完整策略；`split_ratios`、`split_ranges` 与 `split_window_counts` 必须与其一致。新产物的 `run.task_split_policy` 独立记录同一策略，存在时必须与 contract 一致。严格加载对照发布时间轴、请求比例、实际索引/UTC 边界、窗口数、run 参数、normalization 拟合范围、validation/test 指标窗口数及全分区指标口径，并严格重建模型权重。新策略声明存在但元数据缺失、类型损坏或不一致时明确拒绝，不能退回 manifest。带新任务标记的任务不能读取 manifest checkpoint，训练结束和服务重启后的完成恢复也执行相同校验。

新 `earth_raw_utc_timeline_v1` checkpoint（官方 DLinear 与上传模型）使用同一套严格历史回测边界：完整 `window + horizon` 个时间步必须在同一**任务分区**内，forecast origin、输入窗口和目标窗口的分区必须相同。跨分区起点不进入 context 的 origins，也不会返回预测结果，而是返回稳定的 `422 earth_prediction_origin_out_of_range`。`origin_split` 表示完整输入加目标窗口所属的分区；合法窗口满足 `input_split == target_split == origin_split`。normalization 永远复用 checkpoint train 统计量。

没有 task split 的旧三小时 checkpoint 继续使用已发布 manifest 的兼容边界和原有历史行为，不自动迁移；旧 checkpoint 不要求新增 split 字段。训练、测试指标、诊断抽样和历史预测对新 checkpoint 复用同一组完整窗口边界，不能各自重新计算起点范围。预测缓存 key 继续包含任务、数据集指纹、checkpoint SHA、origin 和 target timestamps，并额外包含完整窗口边界策略版本；策略版本变化不会复用旧结果。

多模型比较只读取保存的完整 test 指标，要求发布身份、测试 UTC 起止、输入/输出窗口、实际窗口数量和指标口径一致。不一致返回 `409 earth_comparison_incompatible`，说明测试时间范围或窗口不同，不能混合排名。不同 train/validation 划分但相同 test 范围及窗口口径仍可比较。

## 旧任务兼容与验证

真正旧三小时任务没有冻结划分、划分策略标记或指标版本/aggregation 标记时，继续按原发布 manifest 的 train/validation/test 解释，包括原有日块边界、normalization 和上传回测限制；比例兼容字段不改变其历史分区。旧 checkpoint 没有独立 run 策略标记仍可读取，单独的 v2 指标 block 不改变 artifact 的历史划分语义；但将其绑定到新任务时必须满足新任务的冻结划分。不自动改写、迁移或重新训练。已有日频 artifact 可保留为历史记录，活动运行入口仍返回 `dataset_retired`。

从 `AstraAtmos_backend/backend` 使用项目 conda `AstraAtmos` 的解释器执行 `python -m pytest tests/test_earth_task_split.py` 及相关三小时训练、上传、预测、队列和停用回归；Windows 的 NetCDF 测试应通过 `--basetemp` 指定一个新的 ASCII 临时路径。从 `frontend` 执行 `node --test` 对应训练页测试和 `npm run build`。

新增回归使用完整 240×480 的合成三小时源：官方/上传模型均执行一轮训练和 strict reload，覆盖自定义比例与非默认输入窗口、原始时间轴覆盖/互斥、非法比例、过短分区、train-only normalization、缓存隔离、队列恢复、旧 manifest checkpoint、预测分区识别及不同 test 范围比较拒绝。未执行真实两年训练，也不将合成链路验证视为真实模型精度验收。

2026-10-08 本轮实际验证：上述后端相关 12 个测试文件合计 **400 passed**（119.52 秒，5 条既有依赖警告），训练页 `node --test src/pages/ModelTrainingPage/*.test.js` 为 **244 passed**，生产构建成功（23.94 秒，既有大 chunk 警告）。隔离浏览器以合成 API 验证可编辑比例、0 验证比例阻塞、Earth/Mars 草稿分别恢复、请求 0–1 载荷、复制配置、实际 UTC/步数/窗口展示，以及详情和矩阵读取保存元数据；启动请求被拦截，未创建真实训练任务。现有服务只读检查 health、前端主页及数据代理均返回 200。

实现入口：[划分策略](../AstraAtmos_backend/backend/services/earth_task_split.py)、[训练 runner](../AstraAtmos_backend/backend/models/training_scripts/earth_daily.py)、[checkpoint](../AstraAtmos_backend/backend/services/earth_training_artifact.py)、[预测与比较](../AstraAtmos_backend/backend/services/earth_prediction_service.py)、[前端配置](../frontend/src/pages/ModelTrainingPage/earthTrainingConfig.js)。
