# 地球 MERRA-2 DLinear 训练与历史预测

任务级自定义比例、整数分配、normalization、checkpoint 与旧任务兼容详见[地球任务级划分](earth-task-splits.md)；模型测试、DU 散点/残差直方图与 PFI 的范围、参数和缓存见[地球训练后诊断](earth-post-training-diagnostics.md)。

当前地球训练与历史回测只使用 `earth_merra2_3hourly_v1`，支持官方 DLinear / 独立契约上传模型按任务配置窗口（默认 56→24）和 UTC datetime。日频 v1/v2 的总览、训练与预测已停用，返回 409 `dataset_retired`；旧记录和产物保留，须重新训练三小时模型。下文日频固定契约、示例和验收记录均为历史资料。停用规则见[日频数据集停用](earth-dataset-retirement.md)，当前活动契约见[三小时专题](earth-merra2-3hourly.md)及[专用上传契约](earth-3hourly-uploaded-model.md)。

日频与三小时具有不同时间轴和 checkpoint 契约，不互相转换。日频旧任务仅用于追溯，不能继续预测或复制配置。活动三小时网格为 0.75°×0.75°、240×480，字段为 TO3（DU）、U10M/V10M（m s-1）、T2M（K）、SWGDN（W m-2）；预测缓存按数据身份、任务和时间窗隔离。

- 数据包、网格与源处理见[地球 MERRA-2 数据包](earth-compact-dataset.md)
- 数据集身份、发布指纹与目录接口见[服务器数据集注册表](dataset-registry.md)
- 训练页交互与前端结构见[实验中心](experiment-center.md)
- 地球/火星单模型组件、适配层与交互验证见[共用单模型预测工作台](shared-single-model-prediction.md)

## 已开放与未开放

| 能力 | 状态 |
| --- | --- |
| 日频 v1/v2 总览、训练和历史预测 | 已停用，返回 409 `dataset_retired` |
| 官方 DLinear，过去 56 步 → 后续 24 步，目标 TO3 | 已开放，三小时数据 |
| 日频上传模型及旧 checkpoint | 仅保留历史产物，不能启动训练或预测 |
| 三小时官方 DLinear 后端训练，默认 56→24、窗口可配置为 1–240 个三小时时间步、240×480、TO3 DU | 已实现；按需读窗与空间块，完成前严格重载 |
| 三小时历史回测 API，按任务 window/horizon 运行（默认 56→24） | 已实现；UTC datetime、总体/逐步/累计时段 DU 指标，默认输出 72 小时 |
| 三小时数据总览 | 已开放 UTC 时间轴、降采样地图、原生点位及后端日聚合；见[总览协议](earth-overview.md#三小时-utc-总览) |
| 三小时训练/预测前端选项与预测展示 | 已开放；训练显示 UTC/240×480 及当前窗口，地球与火星共用单模型侧栏、三联图/单图、时间步、指标及全屏外壳 |
| 三小时用户上传模型 | 已开放独立 v1/v2 契约，上传/手动重新校验运行隔离 dry-run；创建任务复用报告，训练/验证/测试、严格重载与历史回测保留；见[专用模板说明](earth-3hourly-uploaded-model.md) |
| TO3 必选 + U10M / V10M / T2M / SWGDN 四个可选输入（允许只用 TO3） | 已开放 |
| 训练期归一化（只拟合 train 划分），验证/测试/预测复用 | 已开放 |
| 单文件 checkpoint（权重 + 模型配置 + 数据身份 + 归一化 + 指标） | 已开放 |
| 训练完成前 CPU 严格重载校验 | 已开放；三小时完成门禁由父进程调度有超时的独立进程 |
| 保存验证集与测试集总体 / 逐步 / 累计时段六项指标 | 已开放；MSE（DU²）、RMSE/MAE（DU）、R²（无量纲）、MAPE/SMAPE（%），见[公式及兼容规则](earth-evaluation-metrics.md) |
| 地球多模型完整测试集比较 | 已开放；读取严格验证 checkpoint 的六项 test 指标，旧产物缺项排除该项排名；要求发布身份与测试窗口一致，支持任务 horizon 曲线、参数矩阵与实际契约内的科研导出；不重新推理 |
| 地球训练后诊断（测试入口、预测/真值散点、残差直方图、Earth PFI） | 已开放；固定任务 test 分区，流式空间块计算，页面诊断图不开放科研导出 |
| 预测页按**历史 UTC 起点**回测完整任务 horizon（默认 3 天） | 已开放；参考/预测/残差保持 DU，展示步只切换查看内容，不重新推理 |
| 预测上下文与结果显示所用模型身份（官方 DLinear 或上传模型名称/版本/内容指纹） | 已开放 |
| 服务端返回 UTC 可选起点、任务 horizon 个目标时间戳与真实经纬网格 | 已开放；提前小时与 UTC 时间并列，放大视图使用地球地图 |
| 训练任务的启动、进度、日志、停止、历史、重命名、标签 | 复用现有机制，已开放 |
| 无参考真值的未来日期外推 | 未开放（数据集只到 2021-12-31） |
| Earth / Mars 混合比较、持久性基线 | 未开放 |
| SPHERE、迁移学习、地球其他**官方**架构 | 未开放（返回 409） |
| 地球上传模型的 Ls / MOLA 地形等火星辅助输入 | 不开放（校验期即拒绝并给出原因） |
| Earth 预测结果的持久化缓存 | 未实现；三小时有独立进程内 LRU |

## 日频固定契约（历史资料，已停用）

| 项目 | 值 |
| --- | --- |
| 数据集 ID / 版本 | `earth_merra2_daily_v2` / `v2`（`earth_merra2_daily_v1` / `v1` 为保留身份） |
| 训练脚本 | 服务端固定 `models/training_scripts/earth_daily.py`，客户端不能指定 |
| 模型来源 | `official`（官方 DLinear）或 `uploaded`（用户上传模型，须通过 Earth 兼容校验） |
| 架构 | 官方 `dlinear`（`implementation_id = astraatmos_gridpoint_dlinear_v1`）；上传模型记为 `uploaded`，实际代码由 checkpoint 固定的模型引用决定 |
| 输入 / 输出 | 窗口 7 天、步长 3 天，一次输出三天，不做递归滚动 |
| 上传模型张量契约 | 输入 `[B, 7, C, 36, 72]`，输出 `[B, 3, 1, 36, 72]`（C = 实际选用通道数） |
| 通道顺序 | `TO3` 固定第一，其后按 `U10M, V10M, T2M, SWGDN`；请求顺序不影响模型顺序 |
| 单位 | `TO3` DU；风 `m s-1`；`T2M` K；`SWGDN` W m-2 |
| 目标 | `TO3`，反归一化后为 DU |

日期划分来自已发布 manifest，页面只读展示：

| 划分 | 日期范围（含首尾） | 天数 | 7→3 窗口数 |
| --- | --- | --- | --- |
| train | 2020-01-01…2020-12-31 | 366 | 357 |
| validation | 2021-01-01…2021-06-30 | 181 | 172 |
| test | 2021-07-01…2021-12-31 | 184 | 175 |

窗口严格留在所属划分内：样本 `i` 使用输入索引 `[i, i+6]`、目标索引 `[i+7, i+9]`，不向前一划分借 7 天。

## 日频归一化

- 每个有效通道在 **366 个训练日的全部 36×72 网格**上拟合一个均值与标准差；float64 累加，保存 float32 有效值，`ddof=0`。
- 标准差小于 `1e-6` 的通道保留 `scale = 1.0`，并在 `constant_channel_mask` 中标记，不产生 NaN 或 Inf。
- validation / test / 预测只复用保存的统计量；`fit_split` 必须是 `train`，通道顺序必须与模型输入一致，否则拒绝加载。
- 不做 MinMax、对数、插值、缺值补零或负值裁剪。
- 已发布 v2 包的训练期统计量：`TO3` 286.762064 / 47.314837；`U10M` −0.001669 / 4.970813；`V10M` 0.220144 / 4.115977；`T2M` 278.888111 / 21.438948；`SWGDN` 158.234657 / 106.847826。

## 训练请求与参数范围

沿用 `POST /api/training/start`；数据集既可用顶层 `dataset_id`，也可用旧字段 `hyperparameters.training_dataset`，两处冲突仍按既有规则拒绝。

```json
{
  "model_script": "demo3.py",
  "model_name": "Earth DLinear 7d-3d",
  "model_source": "official",
  "data_source": "default",
  "dataset_id": "earth_merra2_daily_v2",
  "tag_ids": [],
  "hyperparameters": {
    "training_dataset": "earth_merra2_daily_v2",
    "model_architecture": "dlinear",
    "selected_channels": ["U10M", "V10M", "T2M", "SWGDN"],
    "window": 7,
    "horizon": 3,
    "epochs": 10,
    "batch_size": 8,
    "learning_rate": 0.001,
    "seed": 11,
    "early_stopping_patience": 0,
    "linear_hidden_layers": 2,
    "use_sphere": false
  }
}
```

新三小时任务在页面编辑 `train_ratio / validation_ratio / test_ratio`（默认 70/20/10，请求以 0–1 数值发送），按完整原始 UTC 三小时时间轴连续分配；三项均须大于 0、合计 100%。实际边界、步数与窗口数保存为独立任务配置，发布 manifest 不改写。旧三小时任务继续解释为发布日期块，日频运行入口仍停用。

`model_script` 仅为兼容现有客户端保留；服务端会按数据集选择 `earth_daily.py` 并写入任务，实际执行的脚本与客户端提交值无关。

严格范围（越界即 422，不静默夹取）：`epochs` 1…1000、`batch_size` 1…64、`learning_rate` (0, 1]、`seed` 0…2³²−1、`early_stopping_patience` 0…200、`linear_hidden_layers` 1…4。归档日频 `window`/`horizon` 为 7/3；活动三小时各为 1…240 的整数，默认 56/24。布尔值不能充当整数。

### 三小时官方 DLinear

同一训练接口选择 `dataset_id=earth_merra2_3hourly_v1` 后，服务端 profile 为三小时 UTC、任务配置的 window/horizon（默认 56→24）、240×480 全球网格。TO3 必选，四个辅助输入可选，通道顺序固定；目标仅 TO3，单位 DU。支持官方 DLinear 及独立 v1/v2 契约上传模型，创建任务前校验保存的兼容报告、参数 schema、batch 范围及分区长度；上传模型冻结源码/schema/参数后直接排队，不运行实际配置 dry-run，不调用日频上传兼容门或降级模型。上传/手动重新校验的 dry-run、实际训练输出/梯度检查及完成前严格重载保留。

新任务按原始时间轴比例划分后各自独立切窗，完整两年默认 70/20/10 和 56→24 对应 4094/1169/585 步、4015/1090/506 个窗口。无新策略的旧任务仍按 manifest，分别为 2928/1448/1472 步、2849/1369/1393 个窗口。归一化只扫描当前任务 train，逐通道最多 8 步，以 float64 合并统计量；选中通道的缺失显式拒绝。分块生成 normalized float32 磁盘内存映射后按需读取窗口与 24×48 空间块，避免压缩整场反复解压；每个窗口 100 块，`batch_size` 以块为单位，checkpoint 仍绑定完整 240×480 网格。缓存按源数据、实际 train 边界、通道、归一化内容和布局复用，任务的窗口索引和 checkpoint 仍严格保存自己的完整划分；准备日志显示验证/统计量/缓存的命中或生成、进度及耗时，见[训练准备缓存](earth-preparation-cache.md)。缓存通过 `ASTRAATMOS_EARTH_TRAINING_CACHE_DIR` 配置，默认位于 Git checkout 外的工作区父级，完整五通道约 12.55 GiB；全图与分块布局分别使用 `earth_full_grid_v1` 和 `earth_spatial_tiles_v1`，未认证/损坏/中断缓存不得挂载。

三小时 checkpoint schema 为 `astraatmos_earth_forecast_checkpoint_3hourly_v1`，实现 ID 为 `astraatmos_gridpoint_dlinear_3hourly_v1`。保存 window/horizon、hour/step=3、UTC、区间中心规则和 90 分钟中心偏移（01:30…22:30）、完整网格、通道/单位、独立任务分区策略与边界、训练期 normalization、发布 snapshot/SHA/fingerprint 和任务身份；日频 checkpoint 不能加载到三小时任务。验证/测试保存总体、任务 horizon 个 lead step 及累计时长的六项指标（指标 schema v2，单位与公式策略逐项保存），默认 24 步为 3…72 小时和累计前 24/48/72 小时。v1 指标 checkpoint 保持只读兼容；[评价专题](earth-evaluation-metrics.md)定义公式和门禁。

历史 smoke 使用合成 30 天/240 步，测试专用连续 split 各 10 天、各 1 个时间窗口；五通道 CPU 训练 1 epoch，最终内存映射版本严格重载通过。当前另有真实地形冒烟：从完整 MERRA-2 发布只读创建三天仓库外子包，v1/v2 各以 7→1、1 epoch CPU 完成上传校验、训练、严格重载、历史预测和诊断；这验证真实数据读取与地形消费，不代表完整两年训练或精度验收。命令见[三小时训练验证](earth-merra2-3hourly.md#官方-dlinear-后端训练)和[Earth 地形资产](earth-topography-asset.md)。

### 用用户上传模型训练日频 Earth（历史资料，已停用）

把 `model_source` 设为 `uploaded` 并在顶层传 `uploaded_model_id`。自定义参数值放在 `hyperparameters.custom_model_params`，服务端按该上传包自己的 `MODEL_SPEC.parameters` 逐项校验（类型、范围、未知键）。

```json
{
  "model_script": "unified_training.py",
  "model_name": "Earth uploaded conv",
  "model_source": "uploaded",
  "uploaded_model_id": "<UserModelPackage.id>",
  "dataset_id": "earth_merra2_daily_v2",
  "hyperparameters": {
    "training_dataset": "earth_merra2_daily_v2",
    "model_source": "uploaded",
    "selected_channels": ["U10M", "T2M"],
    "epochs": 5,
    "batch_size": 64,
    "custom_model_params": {"hidden_dim": 16, "dropout": 0.0}
  }
}
```

上传模型用于 Earth 的前提条件（任一不满足即拒绝并返回原因）：

| 条件 | 不满足时的表现 |
| --- | --- |
| 上传包属于当前账号且 `validation_status = valid` | 404 / 422 |
| 源文件存在且内容哈希与记录一致 | 422（`uploaded_model_not_earth_compatible`，说明文件已被改动） |
| `MODEL_SPEC.datasets` 显式声明 `earth_merra2` | 422，提示模型未声明该数据源 |
| 声明不要求 Ls / MOLA 地形等火星辅助输入 | 422，提示 Earth 不接受辅助输入 |
| Earth dry-run 真实构建并跑通 `[1,7,{1,5},36,72] → [1,3,1,36,72]` | 422，附带 `build_model` 或形状失败的具体原因 |

**“适用于火星”不等于“适用于地球”**：没有 `datasets` 声明的历史上传包一律按仅火星处理，不会因为它在火星校验通过就被 Earth 接受。

启动训练时服务端把这些身份与代码一起固定进任务与 checkpoint：

- `_uploaded_model_id` / `_uploaded_model_version` / `_uploaded_model_content_hash` / `_uploaded_model_name` / `_uploaded_model_param_schema` / `custom_model_params` 写入任务 `hyperparameters`（客户端提交值不参与身份决定）。
- 训练 spec 的 `uploaded_model` 块只保留服务端白名单字段，并内嵌**哈希校验过的源码文本**。
- 因此任务创建之后用户再上传新版本、或改动/删除原文件，都不会改变该任务实际训练的代码；原始文件被删除或篡改时，重建改用 checkpoint 内嵌副本，并在预测上下文与结果里给出警告。

#### Earth 上传模型的 checkpoint 与重载

`model_ref` 是 checkpoint 内的模型引用块：

| 字段 | 含义 |
| --- | --- |
| `model_source` | `official` 或 `uploaded` |
| `architecture` | 官方为 `dlinear`，上传为 `uploaded` |
| `uploaded_model.package_id` / `version` / `content_hash` / `display_name` | 固定的上传包身份；`package_id` 与 `content_hash` 缺失即拒绝写入 |
| `uploaded_model.source_path` | 用于检测原文件是否仍在/是否被改动，不参与执行 |
| `uploaded_model.source_text` | 哈希校验通过后实际执行的源码副本（必需） |
| `uploaded_model.param_schema` / `custom_model_params` / `build_config` | 重建 `build_model(config)` 所需的参数与配置 |
| `uploaded_model.input_channel_order` | 与 checkpoint 通道顺序一致性校验 |

重建时不使用任何 pickle 反序列化对象：源码先过与上传校验同一套 AST 安全门（只允许 `torch` / `numpy`，禁止 `open`/`eval`/`exec`/`compile`/`__import__` 与 `system`/`popen`/`Popen`/`run`），再 `exec` 后调用 `build_model(config)`；权重用 `torch.load(..., weights_only=True)` 加载。缺少 `model_ref` 的旧 checkpoint（含既有官方 DLinear 任务）一律按官方处理，保持可读、可预测。

## 服务端执行与产物

1. 父进程在**任何数据库写入、上传模型加载或子进程调度之前**解析数据集身份、校验配置支持性与参数，并构造绑定（`dataset_identity_status = verified`）。
2. 任务行 flush 后生成内部 spec（`earth_training_spec_v1`：任务 ID、五列绑定、规范超参数），通过环境变量 `ASTRAATMOS_EARTH_TRAINING_SPEC` 传给子进程。子进程只接收 `--output_path`，身份与快照从不作为命令行参数。
3. 子进程重新校验 spec 与实际快照身份，用训练期统计量构建三份窗口，训练官方 DLinear（Adam + 标准化空间 MSE），按 validation 选择最佳 epoch，恢复最佳权重后评估 validation 与 test。
4. 产物先写同目录临时文件，严格重载并比对同输入输出，全部通过才原子替换为最终 `.pth`。
5. 父进程在**标记 completed 之前**再次 CPU 严格重载：校验 schema、字段、通道顺序、网格、窗口、归一化、指标与任务身份；成功才写 `completed`，并把 checkpoint 里的 DU 指标直接写入任务 `metrics`（不从日志正则凑另一套）。
6. 用户停止后到达的 exit 0 完成回调不会把任务改成 completed；保持既有的 `failed` + `Stopped by user` 约定。

日频 checkpoint 继续使用单文件 `astraatmos_earth_forecast_checkpoint_v1`，三小时使用独立的 `astraatmos_earth_forecast_checkpoint_3hourly_v1`；均包含 `model_state_dict`、`model_config`、`dataset_binding`、`training_contract`、`normalization`、`run`、`metrics`，只保存张量与 `weights_only=True` 可读的普通类型。加载校验 dataset_id 与时间契约，不能跨日频/三小时绑定。

## 日频历史预测（历史资料，已停用）

### `GET /api/earth/predict/context?training_task_id=<id>`（需认证）

返回可选预测起点与已训练信息：

```json
{
  "planet": "earth",
  "task_id": 25,
  "dataset_id": "earth_merra2_daily_v2",
  "dataset_version": "v2",
  "dataset_fingerprint": "8871…ea01",
  "target": "TO3",
  "target_unit": "DU",
  "window": 7,
  "horizon": 3,
  "input_channel_order": ["TO3", "U10M", "V10M", "T2M", "SWGDN"],
  "input_units": ["DU", "m s-1", "m s-1", "K", "W m-2"],
  "grid": {"shape": [36, 72], "latitude": ["…36 个中心纬度…"], "longitude": ["…72 个中心经度…"]},
  "origins": {
    "start": "2020-01-07", "end": "2021-12-28", "count": 722,
    "dates": ["…722 个可选日期…"], "window": 7, "horizon": 3,
    "input_offset_days": -6, "target_offset_days": 1
  },
  "training_split_end": "2020-12-31",
  "metrics": {"schema": "earth_training_metrics_v1", "unit": "DU", "aggregation": "…", "splits": {"…": "…"}},
  "run": {"best_epoch": 1, "epochs_completed": 1, "seed": 11, "device": "cpu", "linear_hidden_layers": 2},
  "model": {
    "model_source": "uploaded",
    "model_architecture": "uploaded",
    "uploaded_model_id": "<UserModelPackage.id>",
    "uploaded_model_name": "EarthConvBaseline",
    "uploaded_model_version": 3,
    "uploaded_model_content_hash": "<sha256>",
    "uploaded_model_source_embedded": true
  },
  "warnings": []
}
```

起点合法条件：起点当天计入输入窗口，输入索引为 `[origin-6 … origin]`，之后还要有 3 个已发布参考日。因此可选范围是 **2020-01-07 … 2021-12-28**（731 日轴上索引 6…727），共 722 个起点。首日起点 `2020-01-07` 的输入日期是 `2020-01-01 … 2020-01-07`，目标日期是 `2020-01-08 … 2020-01-10`。

### `POST /api/earth/predict/run`（需认证）

```json
{"training_task_id": 25, "forecast_origin": "2021-07-08"}
```

响应要点：

- `forecast_origin`、`origin_split`（`train` / `validation` / `test`）、`input_dates`（7 天，`origin-6 … origin`）、`target_dates`（`origin+1 … origin+3`）
- `grid`：真实 36×72 中心经纬、范围、步长、`coverage`、`wrap_longitude`
- `prediction` / `reference` / `residual`：各 3 天，每天 `{field: [36][72], minVal, maxVal, valid_cells}`，单位 DU；`residual = prediction − reference`
- `metrics`：`unit=DU`、`target=TO3`、`aggregation=user_forecast_origin_lead_grid_uniform`、`reference_available=true`、`overall{rmse,mae}` 与 `by_lead[{lead_day,rmse,mae}]`
- `dataset_id` / `dataset_version` / `dataset_fingerprint` / `input_channel_order`，供前端缓存键区分身份
- `model`：本次预测所用模型身份。官方为 `model_source=official`、`model_architecture=dlinear`；上传模型额外给出 `uploaded_model_id` / `uploaded_model_name` / `uploaded_model_version` / `uploaded_model_content_hash` / `uploaded_model_source_embedded`，界面据此显示模型名称、版本与内容指纹前缀
- `warnings`：重建期间原文件缺失或被改动等提示（例如 "the original uploaded model file is unavailable; prediction uses the verified copy stored inside the checkpoint"）。出现警告不影响预测结果，它说明代码取自 checkpoint 内嵌的哈希校验副本

参考场取自同一发布数据集的对应日期，因此指标是**有真值的历史回测**，不是无真值外推。指标在当前 3 天窗口内按「起点 × 提前量 × 格点」等权聚合，不代表独立日样本数。

预测不写入后端持久化缓存，也不进入火星预测缓存。

## 三小时历史回测

同一 `GET /api/earth/predict/context` 和 `POST /api/earth/predict/run` 按已完成任务的 `dataset_id=earth_merra2_3hourly_v1` 选择三小时分支。`forecast_origin` 使用明确 UTC 的 ISO datetime，例如 `2021-07-08T01:30:00Z`；输入包含起点共 window 步，目标共 horizon 步。默认 56→24 时目标从 `2021-07-08T04:30:00Z` 到 `2021-07-11T01:30:00Z`。请求只发送任务 ID 与起点，horizon 取任务冻结契约；前端展示步不能缩短模型输出。

context 返回 datetime 可选范围、数量、完整时间戳数组和 hour/step=3；默认 56→24 的完整两年发布有 5769 个有效起点。结果返回 `input_timestamps`、`target_timestamps`，预测/参考/残差各 `[horizon,240,480]`、单位 DU；`input_dates` / `target_dates` 在三小时响应中为相同时间戳的兼容别名。本次预测总体、每步与累计时段均返回六项指标，采用同一[物理公式](earth-evaluation-metrics.md)，并与完整测试集评价分开标注。参考场来自相同 fingerprint 的真实发布，不重新拟合 normalization。

任务、checkpoint、数据发布、时间规则、窗口和网格均严格一致后才读取缓存。三小时有独立的进程内 LRU（128 MiB、至多 4 条），键包含 planet、dataset_id/version/fingerprint、task_id、规范 origin、全部 target timestamps 和 checkpoint SHA；缓存不会持久化，也不进入 Mars 或旧日频路径。预测页每步显示提前小时与对应 UTC 时间，默认 24 个 lead，切换查看内容不请求预测。完整 API 示例、错误码与验证命令见[三小时历史回测 API](earth-merra2-3hourly.md#三小时历史回测-api)。

预测参考场必须来自同一已绑定发布，当前只支持历史回测；无参考真值的未来外推未开放。三小时每次成功回测都序列化 reference、prediction、residual 三组 `[horizon,240,480]` 完整场。默认 24 步的既有合成 smoke 实际响应 JSON 约 157.5 MB，预测约 5.1 秒、采样 RSS 峰值约 1.75 GiB；这是该次运行的测量，不是容量保证。完整响应下载、JSON 解析和浏览器内存是高分辨率预测界面的性能限制；共用工作台的本轮小网格合成浏览器验证不覆盖此性能边界。总览地图默认先降采样到 60×120，点位序列仍基于原生网格。

## 错误码

| 情形 | HTTP / code |
| --- | --- |
| 非 DLinear、未知架构、SPHERE、迁移学习 | 409 `dataset_training_configuration_not_supported` |
| `model_source=uploaded` 但缺少 `uploaded_model_id` | 422 `invalid_earth_training_parameters` |
| `model_source=official` 却传了 `uploaded_model_id` | 422 `invalid_earth_training_parameters` |
| 上传模型不存在 / 不属于当前账号 | 404 `uploaded_model_not_found` |
| 上传包校验状态不是 valid | 422 `uploaded_model_invalid` |
| 上传模型未声明 `earth_merra2`、要求 Ls / 地形、网格或输出形状不符 | 422 `uploaded_model_not_earth_compatible`（message 含具体原因） |
| 上传文件在任务创建后消失，且 checkpoint 无内嵌副本 | 409 `uploaded_model_missing` |
| 上传文件在任务创建后被改动，且 checkpoint 无内嵌副本 | 409 `uploaded_model_tampered` |
| 窗口/通道/范围/类型/未知字段错误 | 422 `invalid_earth_training_parameters` |
| 客户端提交 `dataset_version` / `dataset_fingerprint` / `dataset_identity_status` / `dataset_snapshot` | 400 `client_identity_not_allowed` |
| 发布包缺失、哈希不符、内容非法（数据读取/创建训练） | 503 `dataset_unavailable`（附 `availability_reason`）；已绑定三小时回测复核返回下列 409 |
| checkpoint 缺失、损坏、与任务身份不符 | 409 `invalid_earth_training_artifact` |
| 数据集发布指纹变化 | 409 `dataset_version_changed` |
| 三小时回测绑定发布缺失、损坏，或时间/网格/版本与绑定不一致 | 409 `dataset_version_changed` |
| 三小时回测选中通道或 TO3 参考场缺测/不可读取 | 503 `earth_prediction_data_unavailable`；包读取权限导致无法验证为 503 `dataset_unavailable` |
| 任务未成功完成 | 409 `earth_prediction_task_not_completed` |
| 预测起点不在数据集内、缺少完整输入或参考时间步 | 422 `earth_prediction_origin_out_of_range` |
| 起点格式非法 | 422 `invalid_earth_prediction_origin` |
| Earth 任务进入火星推理 / metrics / 比较 / Mars PFI | 409 `dataset_prediction_not_supported` |
| 火星迁移选择 Earth 来源任务 | 409 `dataset_transfer_not_supported` |
| 非本人且非管理员访问任务 | 403；任务不存在 404 |

## 场景隔离

- `services/dataset_identity.py` 的 `is_earth_training_task()` 是轻量识别函数（优先读任务身份列，其次读合法 legacy key；损坏身份按非 Earth 处理，保持旧 Mars 行为）。
- 火星推理的所有入口（`predict_task`、metrics/comparison/error-distribution/PFI、`get_test_results`）在**用户访问检查之后、火星数据准备与缓存访问之前**调用 `require_mars_prediction_task()`，拒绝 Earth 任务。
- 训练 `action=test` 按任务行星分流；Earth 使用专用诊断，Mars 保留原有推理数据环境和测试行为。
- 火星迁移在读取来源任务后、加载权重前拒绝 Earth 来源。
- 上传权重校验识别日频 `astraatmos_earth_forecast_checkpoint_v1` 和三小时 `astraatmos_earth_forecast_checkpoint_3hourly_v1`，返回明确不兼容并记录为 `invalid`，不会抽取其 `state_dict` 冒充火星权重。
- 前端 Earth 结果身份以 `planet:earth` 开头，包含数据集 ID/版本/指纹、任务、预测起点和全部目标时间戳，与火星键互不命中。切换行星、任务、起点或账号取消旧请求并清空结果和全屏；Earth 回包还须匹配冻结的账号作用域、任务、起点、完整 window/horizon、模型身份、数据身份和真实坐标。

## 前端

- 训练页「数据集」分区包含两个火星身份和一个三小时 Earth 发布 ID；默认 Earth 入口读取 catalog 的 `default_earth_dataset_id`。Earth 显示 UTC、240×480 和 56/24，允许官方 DLinear 或独立契约上传模型，并提供专用模板、自定义参数和 dataset_id 兼容性状态。Earth 与火星草稿隔离；旧日频草稿不恢复，旧任务不能复制配置或运行预测。
- Earth 的「模型」分区提供**官方 DLinear / 用户上传模型**两个来源。选官方时只显示固定架构说明；选上传时复用同一套上传卡片（上传、模板/说明下载、重新校验、删除、自定义参数入口），选中后向 `GET /api/user-models/{id}/earth-compatibility` 取回服务端 Earth 兼容性结论：**未取到结论一律按不可用处理**，不兼容时在区内显示具体原因并禁用「开始实验」，检查器同步列出阻塞项。
- Earth + 上传模型的请求只发送模型 ID 与自定义参数值；版本、内容哈希与数据快照由服务端固定，前端不生成也不接受。
- 地球数据集分区显示当前发布的频率、网格、变量、单位与窗口契约；发布日期划分和发布指纹仍由服务端在创建任务时校验并绑定。数据不可用时会在就绪检查里给出阻塞原因并禁用「开始实验」，新实验默认使用设置中的窗口（默认 56→24），范围为 1–240 个三小时时间步，任务按保存的窗口运行，网格仍为 240×480。
- 预测页地球单模型仅提供已完成且权重有效的三小时任务，选择 UTC 起点并使用绑定 checkpoint 和完整窗口；进入页面只读取训练任务目录，点击“开始预测”后才加载预测上下文、执行 checkpoint/数据身份校验并运行预测，诊断可用性检查也在上下文成功后开始；展示顺序为参考、预测、残差（DU）。`EarthPredictPanel` 只装配上下文、完整测试集指标配置与手动诊断，单模型页面本身不添加行星专属标题、说明段、`TO3 · DU` 胶囊、上下文折叠块、起点说明行或参考场说明，主按钮沿用共用的 `🚀 开始预测`，区块顺序与火星一致；完整测试集卡片下的分区行由 checkpoint 的 `split_policy`/`split_ranges` 映射成与火星相同的 `split_meta` 形状。地球全屏使用真实地球地图，不加载火星纹理、Ls 或火星单位；主题、色带和数值精度遵循全局设置。
- `singleModelAdapters.js` 负责 UTC 起点、Earth 请求/响应、真实网格、DU、时间步与能力配置；`PredictionPlanetAdapter.jsx` 装配地球地图渲染，`predictionPresentation.js` 提供六项指标、评价范围及 ΔRMSE（DU）PFI 展示配置。共用组件不自行转换 Earth DU 或请求 Mars API。当前预测窗口指标、完整测试集指标与抽样诊断分别标注；诊断独立手动触发，保留默认参数和现有 test 分区/缓存身份，诊断图科研导出未开放。
- 地球任务不进入 Earth/Mars 混合比较、火星推理或迁移学习。创建任务检查平台窗口范围与参数 schema，但不额外运行实际窗口 dry-run；运行时的模型构建、输出与梯度错误由训练任务报告。旧日频任务仅用于追溯。组件职责、合成桌面/390px 交互回归和实际后端验证见[共用单模型预测工作台](shared-single-model-prediction.md)。
- 生产站点使用 `frontend/dist`，前端源码修改后必须在 `frontend/` 执行 `npm run build`。

## 日频历史验证入口（当前不能运行）

后端从 `AstraAtmos_backend/backend/` 执行，使用 `AstraAtmos` conda 环境，并为 Windows NetCDF 测试指定新的纯英文临时目录：

```powershell
$earthTrainingTemp = Join-Path $env:TEMP ('.earth-training-test-' + [guid]::NewGuid().ToString('N'))
& "$env:CONDA_PREFIX\python.exe" -m pytest `
  tests/test_earth_training_contract.py tests/test_earth_training_data.py `
  tests/test_earth_dlinear.py tests/test_earth_training_artifact.py `
  tests/test_earth_training_runner.py tests/test_earth_training_service.py `
  tests/test_earth_training_isolation.py tests/test_earth_prediction_service.py `
  tests/test_earth_prediction_routes.py `
  -q --basetemp "$earthTrainingTemp"
```

本节命令采用逐文件回归便于定位；原先全局 `sys.modules` 替换导致的预测/上传训练组合导入问题已修复，相关文件可组合运行。组合范围、临时目录要求及全库收集与执行的区别见[后端测试隔离](backend-test-isolation.md)。

真实端到端验收（需要后端运行在 8000 端口）：

```powershell
& "$env:CONDA_PREFIX\python.exe" scripts\audit\earth-training-acceptance.py
```

脚本只使用公开 HTTP 接口：登录 → 读目录 → 启动真实训练 → 轮询到结束 → 读可选起点 → 运行历史预测 → 校验三天 DU 场/日期/网格/指标 → 逐条验证错误码。

> 本机 `.env` 若保留了 `.env.example` 的占位 `TRAINING_PYTHON_PATH="C:\Path\To\Your\Environment\python.exe"`，训练子进程会以 `FileNotFoundError: [WinError 2]` 失败。请把该值改为真实解释器路径，或删除该行让 `config.TRAINING_PYTHON_PATH` 回退到当前解释器。

## 日频后续设计（历史资料）

第四步及以后可直接读取 `astraatmos_earth_forecast_checkpoint_v1`，用保存的 `model_config` / `input_channel_order` / `normalization` / grid / splits 重建模型。未来比较身份至少包含 `planet`、`dataset_id`、`version`、`fingerprint`、`target`、`horizon`、测试起点集合与指标 aggregation；持久性基线与无参考真值外推须复用同样窗口、目标日期与单位。
## 任务分区与发布 split

新三小时官方/上传任务统一使用 `earth_raw_utc_timeline_v1`，比例针对完整原始时间轴，采用向下取整、最大余数（同余数按 train/validation/test）确定步数。各分区独立生成窗口，完整输入与目标不跨界；每区至少 window+horizon 步，否则创建前返回 422。normalization 仅拟合当前任务 train，并保存策略与实际边界用于缓存及严格重载。

官方历史回测保留完整发布可选起点；上传回测要求完整输入/目标落在同一任务分区。两者 origin_split 使用任务边界。比较要求 test UTC 范围、window/horizon、实际窗口数及指标口径一致，不一致返回 409。新任务保存独立 `_earth_task_split_policy`，恢复和产物读取同时识别该标记、已有 split 及 v2 指标任务证据，缺失冻结划分时拒绝；只有真正旧任务继续使用发布 manifest，不自动迁移。详细元数据、默认两年示例和验证范围见[任务级划分专题](earth-task-splits.md)。上传模型的构建、训练和重载统一初始化 CPU float32，eval 批次一致性在准入和实际权重重载时核验，详见[上传模型契约](earth-3hourly-uploaded-model.md)。

独立的三小时 **36×72（5°）** 训练发布、完整全球窗口与上传模型契约见 [5° 数据集](earth-merra2-5deg.md)；本文件的原 0.75° 契约继续兼容。
