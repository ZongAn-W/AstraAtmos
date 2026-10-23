# Earth MERRA-2 三小时数据构建、注册、训练与历史回测

上传模型新增可选[全图 v2 契约](earth-3hourly-fullgrid-model.md)，一次接收完整全球图，batch 按全球时间窗口计且为 1–64（默认 1，需实测显存容量）。本文下方 24×48/100 块说明适用于官方 DLinear 与 v1 上传模型；v2 不做空间分块并使用带版本的 time-major 全图缓存，验证/test 仍覆盖全部窗口和格点。旧任务不自动迁移。

`earth_merra2_3hourly_v1` 是地球唯一活动数据集，已接入目录、分块校验、官方 DLinear / 独立契约上传模型 56→24 训练与历史回测 API。总览提供 UTC 时间选择、播放、地图、原生点位与三小时源日聚合的年度分析；预测展示 UTC 起点和 24 个 lead。自 2026-10-08 起日频 v1/v2 运行接口均返回 409 `dataset_retired`，旧任务及产物仅归档、不自动迁移；详见[停用约定](earth-dataset-retirement.md)。下文涉及日频可用性和兼容回归的阶段记录为历史验证。

## 数据契约

2026-10-09 训练链路契约补强：新任务的冻结划分缺失时拒绝队列恢复和产物读取，保留真正 legacy manifest 行为；上传模型统一 CPU float32 构建/重载，上传探针与 checkpoint 加载实际权重后核对 eval 样本独立性，重载覆盖保存的任务批次且不改变参数/缓冲区。旧上传校验报告需重新校验。当时创建任务还运行实际配置 dry-run；当前 v1/v2 任务创建改为复用保存的上传兼容报告。细节及兼容边界见[任务划分](earth-task-splits.md)和[上传模型契约](earth-3hourly-uploaded-model.md)。

本轮分文件回归共 415 passed、2 skipped：任务划分 103、训练服务 12、runner 12、artifact 39、上传契约 93、dtype 2、六指标 110、71→9 pipeline 2、诊断集成 15、预测路由 27；跳过的是官方模型不适用的上传源码检查。训练页配置/划分/指标前端测试 36 passed，文档文件链接与 Git diff 空白检查通过。另执行 `scripts/smoke_earth_topography.py`，使用真实 MERRA-2 发布包和真实 ETOPO 资产创建三天仓库外子包，v1/v2 各完成 1 epoch、严格重载、历史预测和诊断；该有限冒烟不代表完整两年训练或预测精度验收。保留既有 NumPy/FastAPI/Pydantic 依赖提示。

| 项目 | 契约 |
| --- | --- |
| 来源 | `raw/slv` 的 TO3、U10M、V10M、T2M；`raw/rad` 的 SWGDN；不读取 `raw/chm` |
| 完整日期范围 | 2020-01-01 至 2021-12-31，共 731 天、5848 个三小时步 |
| 时间 | UTC、Gregorian calendar；连续三小时，无重复、缺步 |
| 时间标签 | 三个小时平均中心 00:30、01:30、02:30 求算术平均，标为 01:30；其余类推至 22:30 |
| 时间边界 | `[00:00,03:00)` 等连续区间；NetCDF 同时保存中心 `time` 和 `time_bounds` |
| 全球目标网格 | 240×480，0.75°；纬度中心 −89.625…89.625，经度中心 −179.625…179.625 |
| 单元边界 | 纬度 −90…90，经度 −180…180；`lat_bounds`、`lon_bounds` 显式保存 |
| 字段 | float32，维度顺序 `time,lat,lon` |
| 单位 | TO3 DU；U10M/V10M `m s-1`；T2M K；SWGDN `W m-2` |
| 掩码 | 每个字段配套 `<变量名>_valid_mask`，1=有效、0=缺失；缺失字段值为 NaN |
| 模型窗口契约 | 输入 56 步、输出 24 步；每步 3 小时，即 7 天→3 天；官方 DLinear 和独立 v1 上传模型使用此 profile |

原始 TO3 的 `Dobsons` 标识规范化为 `DU`，不缩放数值。构建器严格检查 SLV/RAD 日期配对、每天 24 个 00:30…23:30 UTC 小时中心、全局坐标、维度和单位。不采用每三个时间点选一个的抽样方式。

## 重网格与缺失规则

先在原始网格上计算每三个小时的平均，再执行一阶保守球面重网格：源单元采用相邻中心的中点边界，纬度两端裁剪到极点；纬度权重为重叠区间的 `sin(north)-sin(south)`，经度权重为角度重叠。经度同时考虑 −360、0、+360° 周期副本，正确分配位于 −180° 中心的源单元在接缝两侧的贡献。所有目标单元必须有完整空间覆盖。

平均和重网格采用 float64，输出存储为 float32。它是网格单元平均，不是双线性插值或隔点抽样；细目标网格不会恢复原数据没有的小尺度信息。

三个小时中任一成员缺失，该原始单元的三小时值就缺失；任何具有正重叠面积的源单元缺失，对应目标单元也缺失。不填零、不按剩余有效面积重新归一化。manifest 分别记录小时源数据、三小时源单元和目标单元的缺失数量与缺失率。

构建器逐天、逐变量读取和压缩写入，完整两年不会一次物化整个体积。没有缺失的帧会检查面积均值守恒，并记录最大守恒误差和 float32 存储误差。

## 构建与重复运行

从 `AstraAtmos_backend/backend/` 执行，使用项目指定的 工作区约定的 Conda 解释器。将 `$merraRaw` 设置为实际 raw 路径，`$earthOutput` 设置为 Git checkout 和 raw 目录之外的新目录；不要将 smoke 与完整发布写到同一目录。

最小 smoke 是 7 天，允许 `--smoke-days 7..30`：

```powershell
& "$env:CONDA_PREFIX\python.exe" -B -m scripts.build_earth_merra2_3hourly `
  --source-root "$merraRaw" --output-dir "$earthOutput" `
  --start-date 2020-01-01 --smoke-days 7

& "$env:CONDA_PREFIX\python.exe" -B -m scripts.verify_earth_merra2_3hourly `
  --package-dir "$earthOutput" --raw-dir "$merraRaw" `
  --report "$earthOutput\verification.json"
```

完整两年处理省略 smoke 参数，并使用另一个外部目录：

```powershell
& "$env:CONDA_PREFIX\python.exe" -B -m scripts.build_earth_merra2_3hourly `
  --source-root "$merraRaw" --output-dir "$earthFullOutput" `
  --start-date 2020-01-01 --end-date 2021-12-31
```

`--output-dir` 优先；省略时使用 `ASTRAATMOS_EARTH_MERRA2_3HOURLY_DIR`，再回退到工作区父级的 `data/earth/merra2_3hourly_v1/`，不会写入项目 Git checkout。输出为：

```text
merra2_3hourly_v1/
  earth_merra2_3hourly.nc
  manifest.json
  verification.json       # 可选验证报告
```

重复构建相同目录会先核对请求范围、源清单、数据 SHA、manifest 指纹和已存网格/时间/字段，匹配后复用。改变日期范围、源文件或已有产物损坏时拒绝覆盖。中断后的不完整 NetCDF 保留，改用新目录重跑；脚本不会删除文件。验证报告也拒绝覆盖已有文件；重复验证可不传 `--report`，或指定新的报告路径。

## Manifest 与验证

manifest 包含数据集 ID/版本、`fingerprint_rule`、`source_inventory_version`、源日期/产品/相对路径/大小/逐文件完整 SHA-256、顶层 `source_mtimes` 溯源块、来源清单摘要、变量及单位、时间规则、目标坐标与边界、重网格方法、缺失规则与统计、规划窗口、数据文件大小/SHA-256 和数据集 fingerprint。绝对源路径不进入发布 manifest。新构建写 `fingerprint_rule=2`、`source_inventory_version=3`：清单条目只保留 `date`/`product`/`path`/`name`/`bytes`/`sha256`，`mtime_ns` 移到不参与任何身份摘要的顶层 `source_mtimes` 块（`path -> mtime_ns`）。`bytes` 与逐文件 SHA-256 属于内容身份，`mtime_ns` 只作为溯源和快速变化检测信号，不证明文件内容完整。构建开始保存 digest inventory，构建结束及 manifest 发布前都会重新计算全部源文件摘要；内容变化（即使大小和 mtime 恢复原值）会拒绝发布，而纯 mtime 漂移（AV 扫描、复制、touch）既不改变身份，也不让构建器或 registry 判定已有发布损坏。

fingerprint 复用 `build_dataset_fingerprint()` 的组合方式：数据集 ID、版本、manifest 内容摘要、NetCDF SHA-256，组合函数本身不变。manifest 内容摘要的规则由 manifest 自带的整数 `fingerprint_rule` 决定，构建器（`scripts/build_earth_merra2_3hourly.py`）、服务端（`services/earth_dataset_metadata.py`）和独立验证器（`scripts/verify_earth_merra2_3hourly.py`）各自保留一份实现，并由交叉验证测试锁死一致：

- **rule 1（legacy，字段缺失即此值）**：对除 `dataset_fingerprint` 外的**整份 manifest** 取 canonical 摘要（该摘要因此覆盖 manifest 的每一个字段：给既有包补写 `fingerprint_rule: 1` 本身就会改变内容摘要，既有发布包必须保持发布时的原样）。既有已注册发布包（`source_inventory_version` 缺失）声明的就是 rule 1，其 `manifest_content_sha256` 与 `dataset_fingerprint` 逐字节冻结——该摘要会写进 descriptor 与共享验证证明，因此不得漂移。
- **rule 2（新）**：在排除 `dataset_fingerprint` 的基础上再排除顶层 `source_mtimes` 与 `fingerprint_rule`，并把 `source_files` 投影为身份字段元组 `(date, product, path, bytes, sha256)`。`mtime_ns`（文件系统元数据，不是内容）与 `name`（`path` 的冗余后缀）显式排除；因此键顺序变化、`name` 改写、`mtime` 漂移都不改变身份，而 `path`、`bytes`、`sha256` 或清单成员增减必然改变身份。

规则选择 fail-closed：缺失等价 rule 1，`1`/`2` 按上表解释，其他任何值（0、负数、字符串、浮点）都直接报错，不静默回退 rule 1。`source_sha256` 校验同样按 `source_inventory_version` 分流：缺失或 `2` 校验包含 `name` 与 `mtime_ns` 的完整清单，`3` 只校验身份字段投影，两侧语义必须一致。canonical JSON 使用排序键、紧凑分隔符、ASCII 转义且拒绝 NaN；验证报告另记录实际 manifest 文件字节的 SHA-256，并透传 `fingerprint_rule`、`source_inventory_version`、`manifest_content_sha256` 与 `dataset_fingerprint`。日频注册所用的固定发布哈希不变。

既有发布包在消费方走**短路**而不是重算：identity 三元组直接取自 manifest 既有字段（`dataset_id`、`dataset_version`、`data_sha256`）与已冻结的 `manifest_content_sha256`（来自 registry descriptor 或共享验证证明），组合出的 fingerprint 必须与发布值相同；不再对 1462 条源清单重新投影。冷路径（没有可用证明时）退回 rule 1 全文摘要，两条路径必须得到同一个 fingerprint，否则拒绝。

独立验证器扫描所有输出字段及掩码，核对时间中心、三小时边界、坐标/单元边界、单位、缺失率、SHA 和 fingerprint。传入 `--raw-dir` 时，独立读取首/中/尾日期的原始数据，在三组小时区间、九个目标单元（含两极和经度接缝）直接积分源单元重叠面积，与输出比较；该数值检查不调用构建器的平均或重网格函数。

回归测试从后端目录执行，始终使用新的工作区临时目录，避免 pytest 清理已有目录：

```powershell
$earthTestTemp = Join-Path $env:TEMP ('.earth-3hourly-test-' + [guid]::NewGuid().ToString('N'))
& "$env:CONDA_PREFIX\python.exe" -B -m pytest `
  tests/test_build_earth_merra2_3hourly.py tests/test_preprocess_earth_reanalysis.py `
  -q --basetemp "$earthTestTemp" -p no:cacheprovider
```

2026-10-07 已完成完整两年真实构建与独立数据验收：输出目录为 `<数据根目录>\data\earth\merra2_3hourly_v1_full_20200101_20211231`，包含 5848 步 NetCDF、manifest 和 `verification.json`。验证覆盖时间轴、UTC 标签、240×480 网格、变量/单位、缺失值、SHA/fingerprint、周期经度、重网格均值守恒与首/中/末日期原始数据抽查；这不是训练或预测验收。注册、惰性窗口、官方 DLinear 后端训练与历史回测 API 已实现，三小时前端时间轴也已接线。

## 注册与目录 API

后端启动前将 `ASTRAATMOS_EARTH_MERRA2_3HOURLY_DIR` 设置为包含两份发布文件的目录，或在后端 `.env` 中配置；相对路径按后端启动目录解析。未配置时为 `data/earth/merra2_3hourly_v1/`。这是后端运行时默认值，与离线构建器默认的工作区外目录不同；应显式配置到实际构建目录。日频的两个环境变量继续指向原发布。客户端只选择 dataset_id，不能指定目录、版本、fingerprint 或 snapshot。

```powershell
$env:ASTRAATMOS_EARTH_MERRA2_3HOURLY_DIR = "$earthOutput"
```

`GET /api/datasets` 保留原四个身份并追加新 ID，`GET /api/datasets/earth_merra2_3hourly_v1` 返回独立描述符。完整两年包摘要：5848 步，`2020-01-01T01:30:00Z` 至 `2021-12-31T22:30:00Z`，指纹 `76d6fc9f305fe5e6d13a03765beab2feaede5bb8bdf35c268ff2138c33687357`。该包发布于 `fingerprint_rule` 出现之前，因此按 rule 1 解释，内容摘要与 fingerprint 保持冻结（`manifest_content_sha256=3ec08179…3282e`）；新增的规则只影响之后构建的新包，不重算、不改写已注册发布。经校验的 7 天 smoke 详情仍可作为构建器回归夹具参考：

开发与生产默认 Earth 数据集均为 `earth_merra2_3hourly_v1`，作为 `default_earth_dataset_id` 返回。`ASTRAATMOS_DEFAULT_EARTH_DATASET_ID` 只接受该值，旧日频配置须修改或移除；不能通过设为日频恢复入口。缺失三小时包时 catalog 仍返回 `missing/package_missing`，界面不回退到旧包。完整发布的数据验证与模型精度验收仍是不同范围。

```json
{
  "dataset_id": "earth_merra2_3hourly_v1",
  "dataset_version": "v1",
  "schema": "astraatmos_earth_3hourly_v1",
  "availability": "available",
  "frequency_hours": 3,
  "step_unit": "hour",
  "step": 3,
  "grid_shape": [240, 480],
  "time": {
    "kind": "datetime", "time_zone": "UTC",
    "start": "2020-01-01T01:30:00Z", "end": "2020-01-07T22:30:00Z",
    "count": 56, "frequency_hours": 3, "step_unit": "hour", "step": 3
  },
  "training_profile": {
    "window": 56, "horizon": 24, "target": "TO3", "target_unit": "DU",
    "frequency_hours": 3, "step_unit": "hour", "step": 3, "grid_shape": [240, 480],
    "model_architectures": ["dlinear"], "model_sources": ["official"]
  },
  "capabilities": {
    "metadata": true, "training": true, "web_overview": true, "trained_prediction": true
  }
}
```

实际响应还包含完整坐标、五变量/单位/缺失率、时间边界、发布 split 和三个 SHA/fingerprint 字段。`manifest_sha256` 是原文件字节摘要，`manifest_content_sha256` 是 manifest 身份内容摘要（规则由 `fingerprint_rule` 决定：rule 1 为整份 manifest，rule 2 为内容投影）；fingerprint 使用后者，与构建器原记录一致。新包由服务器目录及其 manifest 定义，没有复用或替换日频固定哈希；格式化 manifest 可改变字节摘要而不改变产品 fingerprint。

首次查询或后台预热校验 manifest、NetCDF 完整 SHA、无重复/无缺步的三小时 UTC 轴、精确全球中心与边界、变量及单位、NaN/掩码对应关系、逐项统计与来源清单；扫描每变量最多 8 步，缓存元信息及只读坐标，完整体积留在磁盘。文件签名变化重新校验，未变化的缺失/损坏结果也缓存。全量包独立验证还直接读取首/中/末日期源文件，405 个目标单元样本全部通过，最大绝对误差为 `2.65e-5`。包不存在返回 `missing/package_missing`，哈希或契约不符返回 `invalid` 及稳定原因；读取权限问题返回 `unverified/package_unreadable`。活动注册详情与列表仍返回 200，日频详情返回 409 `dataset_retired`。

新训练与历史回测按任务身份进入独立的 56→24 分支；日频日期工具仍拒绝三小时身份，总览按数据集选择 timestamp 分支，避免落入 7→3、date-only 或 Mars 路径。`trained_prediction=true` 表示后端入口已接线，不表示 7 天 smoke 有足够窗口或已经存在正式 checkpoint。身份伪造沿用顶层 Schema 422、嵌套参数解析 400 `client_identity_not_allowed` 的分工，也拒绝 `fingerprint` / `snapshot` 别名。旧训练 JSON、数据库和 checkpoint 不迁移、不重写。

新增注册/数据协议回归为 `tests/test_earth_3hourly_registry.py` 与 `tests/test_earth_3hourly_data_contract.py`；依照上面的新外部 basetemp 约定，用项目指定解释器执行。

## 三小时上传模型

已开放独立 `astraatmos_earth_3hourly_uploaded_model_v1` 契约，绑定 `earth_merra2_3hourly_v1`。下载[专用模板](earth-3hourly-uploaded-model-template.py)或[v1 四位置共享门控模板](earth-3hourly-four-positions-shared-no-after-decoder-model.py)，调用形状、通道轴、dtype、设备、单位、保留参数及错误码见[模板说明](earth-3hourly-uploaded-model.md)。日频/Mars 上传结论不继承，三小时专属模型也不会自动获得它们的资格。

实际 v1 模型调用为 float32 `[B,window,C,24,48] → [B,horizon,1,24,48]`，默认 56→24，C 是 TO3 加所选辅助变量的规范顺序；100 个无重叠块拼为 240×480。上传或手动重新校验的 dry-run 在有超时的隔离子进程中检查 16 个通道组合、B=1/2、eval/train 前向与 backward。v1/v2 创建任务复用保存的具体 dataset/schema 兼容报告，复查归属、valid 状态、报告证据、源码摘要、参数 schema 和 batch 范围，冻结源码/schema/参数后排队，不重复执行实际配置 dry-run；未知、失败或缺少验证证据不能写入训练任务。实际训练仍检查输出与梯度，完成前仍严格重载。前端开放来源切换及自定义参数，显示可用/不可用/未知和原因，按数据集提供模板下载。

复用服务器 registry、任务级原始 UTC split、仅当前任务训练期 normalization、缺失拒绝、任务调度和训练/评估循环。上传代码不控制路径、版本、fingerprint、snapshot 或 normalization。队列保存冻结源码引用，重启恢复同一版本。上传 checkpoint 仍用独立三小时 artifact schema，另保存上传 implementation `astraatmos_earth_3hourly_uploaded_runner_v1`、契约 schema、块尺寸、实际 build config 和源码摘要；重载核对它们并严格加载权重。任务完成前使用 30 秒超时的独立 CPU 进程重载并检查前向一致性，失败不能 completed。日频和 Mars checkpoint 不能冒充三小时产物。

新 `earth_raw_utc_timeline_v1` checkpoint 的官方 DLinear 与上传模型历史回测都要求完整 window+horizon 步在同一任务分区内；context 只返回满足条件的起点，跨分区返回 `earth_prediction_origin_out_of_range`。旧 manifest 任务在默认 56→24 下共有 2849+1369+1393=5611 个此类起点；新任务依保存的比例、实际边界与窗口计算。没有 task split 的旧 checkpoint 保留各自的 manifest 兼容行为；日频任务不能回测。

## 官方 DLinear 后端训练

沿用 `POST /api/training/start`，顶层 `dataset_id` 和兼容字段 `hyperparameters.training_dataset` 选定同一发布；冲突返回 400 `dataset_id_conflict`。服务端固定 `earth_daily.py`，客户端不能改变数据路径或身份。三小时支持官方 DLinear，或通过上方独立契约的 uploaded 模型；不支持 SPHERE、其他官方架构和迁移学习。上传请求指定 `model_source=uploaded`、自己的 `uploaded_model_id`、`model_architecture=uploaded` 和 schema 内的 `custom_model_params`，其余窗口及训练参数沿用本节规则。

```json
{
  "model_script": "earth_daily.py",
  "model_name": "Earth DLinear 56-24",
  "model_source": "official",
  "dataset_id": "earth_merra2_3hourly_v1",
  "hyperparameters": {
    "model_architecture": "dlinear",
    "window": 56, "horizon": 24,
    "train_ratio": 0.7, "validation_ratio": 0.2, "test_ratio": 0.1,
    "selected_channels": ["U10M", "V10M", "T2M", "SWGDN"],
    "epochs": 1, "batch_size": 8, "learning_rate": 0.001,
    "seed": 11, "linear_hidden_layers": 2
  }
}
```

TO3 始终为首个输入和唯一输出；`selected_channels=[]` 表示只用 TO3，辅助通道按 U10M、V10M、T2M、SWGDN 排序。所选通道缺失时显式拒绝；合法数值零保留，不填零、不插值。

新训练任务使用[任务级原始 UTC 划分](earth-task-splits.md)，默认 70/20/10，可配置且每项大于 0。先按完整原始时间步分配，再在每区内部生成窗口；各区至少 window+horizon 步，窗口数为 steps-window-horizon+1。发布 manifest、NetCDF split 标记及 fingerprint 保持不变。下表仅描述发布分区与无新策略的旧任务，不能作为新任务默认划分：

| split | UTC 日期块 | 时间步数 | 时间窗口数 |
| --- | --- | --- | --- |
| train | 2020-01-01…2020-12-31 | 2928 | 2849 |
| validation | 2021-01-01…2021-06-30 | 1448 | 1369 |
| test | 2021-07-01…2021-12-31 | 1472 | 1393 |

归一化只拟合当前任务完整 train 区间（旧任务使用 manifest train），每次读取一个通道最多 8 个全网格时间步，以 float64 合并均值/总体方差，保存通道顺序、均值、scale、常量通道标记、拟合 UTC 起止与步数。validation / test 复用统计量。训练先以最多 8 步、一个通道的分块读取生成 normalized float32 `.npy` 内存映射，再由 `EarthThreeHourlyWindows` 读取所需 56→24 窗口及 24×48 空间块；每个时间窗口恰好 100 块，`batch_size` 表示空间块数量。缓存避免压缩 NetCDF 整场 chunk 被每个小块反复解压，不复制全部滑窗或将两年体积读入内存。模型共享各网格点的时间权重，checkpoint 的网格仍为完整 240×480。

缓存目录由 `ASTRAATMOS_EARTH_TRAINING_CACHE_DIR` 配置，默认在 Git checkout 的工作区父目录 `earth_training_cache/`；按源包身份、通道顺序、train-only normalization 内容、存储布局和缓存版本查找复用，以只读映射使用。模型参数、batch、学习率和 epoch 不进入键；window/horizon 变化但 train 实际边界不变时也无需重建全时间轴缓存。验证证明和统计量跨后端/训练子进程共享，完整验证仍保留；发布、并发、损坏拒绝和实测计时见[训练准备缓存](earth-preparation-cache.md)。完整两年五通道缓存约 12.55 GiB，TO3-only 约 2.51 GiB；默认 batch=8 的窗口/目标张量约 10.7 MiB，另有模型 activation 与系统文件页缓存。完成、中断和损坏的缓存均保留，不批量删除源包或旧缓存。未发布临时目录不得命中；既有当前布局全图/分块缓存必须先完整对照源值认证一次。

新缓存采用 `layout=earth_spatial_tiles_v1`：物理数组为 `[100,time,C,24,48]`，metadata 的 `shape` 仍保存逻辑全球形状 `[time,C,240,480]`，另存 `storage_shape` 与 `spatial_tile_shape`；全图模型使用 `earth_full_grid_v1` 的 `[time,C,240,480]`。流式构建仍每次只解压最多 8 个时间步、一个通道，再按纬向优先的 100 个空间块重排写入；模型输入/目标及归一化数值不变。读取支持对齐空间块、跨块、反向/跨步切片及全球窗口；旧缓存首次挂载前完整认证，未知 layout、错误物理形状、身份或统计量明确拒绝。运行中的任务不迁移已有缓存。

PyTorch 批量取样通过 `EarthThreeHourlyWindows.read_windows` 在一批读取前后各核对一次源包签名，减少逐样本重复文件状态查询；期间源包变化会使整批失败。样本顺序、重复索引、训练 shuffle 及每轮全部窗口 × 100 块的覆盖保持不变。CUDA loader 使用 `pin_memory`，输入/目标以 `non_blocking` 传输，CPU 路径保持普通传输；未增加多进程 worker 或额外预取副本。每批全部非空梯度仍检查 NaN/Inf，但有限性标志在设备上合并后只同步一次，失败时保留首个异常参数名；没有放宽上传输出、梯度或 checkpoint 门禁。缓存调整不减少批次数，也不承诺端到端固定倍数提速或 100% GPU 利用率。

三小时 checkpoint 使用独立 schema `astraatmos_earth_forecast_checkpoint_3hourly_v1` 和实现 ID `astraatmos_gridpoint_dlinear_3hourly_v1`；保存权重/完整网格模型配置、通道顺序及单位、任务 window/horizon（默认 56/24）、hour/step=3、UTC、`timestamp_rule=interval_center`、三小时平均与 90 分钟中心偏移（01:30…22:30）、独立任务划分策略、比例、原始索引/UTC/步数/窗口范围（旧策略保留发布分区）、归一化、数据版本/SHA/fingerprint 和任务身份。日频 checkpoint 不能绑定三小时任务，反向也拒绝，旧日频 schema 与加载行为保留。产物原子保存前严格重载，父进程验证成功后任务才 completed。

新指标使用 `earth_training_metrics_3hourly_v2`，validation/test 的总体、每步 `by_lead` 与累计 `by_horizon` 均含 MSE/RMSE/MAE/R²/MAPE/SMAPE，单位分别为 DU²/DU/DU/无量纲/%/%。horizon 使用任务配置，累计为每 24 小时及最终 lead；总体/累计由全部 float64 误差和与稳定真值统计量计算，保持 `forecast_origin_lead_grid_uniform`。旧指标 v1 仍可读、可预测，缺项显示“未提供”；公式、零分母、恒定真值及严格门禁见[地球评价指标](earth-evaluation-metrics.md)。

验证命令从 `AstraAtmos_backend/backend/` 执行，先 smoke 再分文件回归，每次使用 `<工作区>` 下的新临时目录：

```powershell
$earthSmokeTemp = Join-Path $env:TEMP ('.earth-3hour-training-smoke-' + [guid]::NewGuid().ToString('N'))
& "$env:CONDA_PREFIX\python.exe" -B -m pytest `
  tests/test_earth_3hourly_training_runner.py::test_smoke_training_publishes_56_to_24_checkpoint `
  -q --basetemp "$earthSmokeTemp" -p no:cacheprovider

$earthTrainingTests = @(
  'tests/test_earth_3hourly_training_contract.py', 'tests/test_earth_3hourly_training_data.py',
  'tests/test_earth_3hourly_artifact.py', 'tests/test_earth_3hourly_training_runner.py',
  'tests/test_earth_3hourly_training_service.py', 'tests/test_earth_training_contract.py',
  'tests/test_earth_training_data.py', 'tests/test_earth_training_artifact.py',
  'tests/test_earth_training_runner.py', 'tests/test_earth_training_service.py'
)
foreach ($earthTrainingTest in $earthTrainingTests) {
  $earthTrainingTemp = Join-Path $env:TEMP ('.earth-training-regression-' + [guid]::NewGuid().ToString('N'))
  & "$env:CONDA_PREFIX\python.exe" -B -m pytest $earthTrainingTest `
    -q --basetemp "$earthTrainingTemp" -p no:cacheprovider
  if ($LASTEXITCODE -ne 0) { throw "Training regression failed: $earthTrainingTest" }
}
```

2026-10-07 早期官方内存映射版本 smoke 历史记录：**1 passed，16.02 秒**。合成 fixture 为 30 天、240 步、完整 240×480 网格，测试专用连续 split 各 10 天/80 步，train/validation/test 各 1 个时间窗口；五通道 CPU 训练 1 epoch，归一化仅拟合前 10 天（结束 `2020-01-10T22:30:00Z`），产物严格重载与指标检查通过。该 fixture 不代表原始 MERRA-2 全两年发布；当时真实 7 天包仅 56 步，没有完整训练窗口或 validation/test，未执行真实全两年构建或训练。完整包后续独立数据验证见前文；本轮上传链路验证范围另记于文末。

最终训练链路五文件 **125 项通过**；旧日频训练契约、数据、checkpoint、runner/service、隔离与预测回归 **195 项通过**。注册、日频上传模型、训练数据与队列相关回归 **238 项通过**；内存映射改动后日频 runner/checkpoint **66 项再次通过**。所有测试使用项目指定解释器，临时数据、映射和权重放在新的工作区临时目录。存在既有 NumPy 二进制大小和 FastAPI/Pydantic 弃用提示，最终测试没有失败。

## 三小时历史回测 API

沿用认证接口 `GET /api/earth/predict/context?training_task_id=<id>` 和 `POST /api/earth/predict/run`。服务端根据已完成任务的 `dataset_id` 选择三小时 profile；客户端只传任务 ID 和起点，不能指定发布目录、数据版本、fingerprint、snapshot、窗口或目标时间戳。任务必须属于当前用户或由管理员访问，并具有严格可重载的三小时 checkpoint；旧日频 checkpoint 不能用于三小时任务。

起点是输入窗口的最后一个时间中心，要求包含起点的 window 个输入步与随后的 horizon 个参考步完整存在。以下 API 数量、时间戳与指标示例采用默认 56→24：输入从 `origin−165h` 到 `origin`，目标为 `origin+3h` 至 `origin+72h`；非默认窗口按保存的任务配置生成。只接受明确 UTC 的 ISO datetime（`Z` 或 `+00:00`），精确匹配发布轴，不截断为日期或吸附到最近采样点；响应统一使用 `Z`。真实参考 TO3 从与 checkpoint 相同 fingerprint 的发布包读取，预测仅复用训练期 normalization。

完整两年、无缺步的 5848 步发布，既有官方模型可选起点共 **5769** 个，从 `2020-01-07T22:30:00Z` 到 `2021-12-28T22:30:00Z`。旧 manifest 任务的上传模型过滤跨发布 split 起点，返回 **5611** 个；新任务的数量取决于任务分区和窗口。下面是官方 context 节选；省略号仅用于示意，实际数组包含该模型来源的全部可选时间戳：

```json
{
  "planet": "earth",
  "task_id": 123,
  "dataset_id": "earth_merra2_3hourly_v1",
  "dataset_version": "v1",
  "dataset_fingerprint": "<64 位 SHA-256>",
  "target": "TO3", "target_unit": "DU",
  "window": 56, "horizon": 24,
  "frequency_hours": 3, "step_unit": "hour", "step": 3,
  "time_zone": "UTC", "timestamp_rule": "interval_center",
  "grid": {"shape": [240, 480], "latitude": ["…240 个中心…"], "longitude": ["…480 个中心…"]},
  "origins": {
    "kind": "datetime", "time_zone": "UTC", "frequency_hours": 3,
    "step_unit": "hour", "step": 3,
    "start": "2020-01-07T22:30:00Z", "end": "2021-12-28T22:30:00Z",
    "count": 5769, "window": 56, "horizon": 24,
    "dates": ["…5769 个 UTC 时间戳…"], "timestamps": ["…相同时间戳…"],
    "input_offset_hours": -165, "target_offset_hours": 3
  }
}
```

运行请求：

```json
{"training_task_id": 123, "forecast_origin": "2021-07-08T01:30:00Z"}
```

此起点的 56 个 `input_timestamps` 从 `2021-07-01T04:30:00Z` 到起点；24 个 `target_timestamps` 从 `2021-07-08T04:30:00Z` 到 `2021-07-11T01:30:00Z`，相邻差严格 3 小时。响应同时给出 `input_dates` / `target_dates` 兼容别名，其内容也是完整 UTC datetime；日频日期字段仅保留历史协议，当前日频预测请求返回 409。

`prediction`、`reference`、`residual` 各有 24 个 `{field, minVal, maxVal, valid_cells}`，每个 `field` 为 `[240][480]`，整体形状各为 `[24,240,480]`，单位均为 DU；`residual = prediction − reference`。真实纬度从南到北，真实经度从西到东，三种场与 `grid.latitude` / `grid.longitude` 一致。选中通道或参考 TO3 的缺测显式报错，不补零。新 `earth_raw_utc_timeline_v1` checkpoint 要求完整输入 `window` 与目标 `horizon` 都属于同一任务分区，forecast origin 也必须属于该分区；官方 DLinear 和上传模型使用完全相同的可用起点集合。`origin_split` 表示完整输入加目标窗口所属的分区，合法窗口的 `input_split`、`target_split` 与其相同。跨分区窗口不会被标记后继续使用，而是返回 422 `earth_prediction_origin_out_of_range`。无 task split 的旧 checkpoint 继续使用已发布 manifest 的兼容行为。

历史预测的 `metrics.aggregation` 跟随 checkpoint 的指标契约：新 task-split/v3 产物使用 `user_forecast_origin_lead_cell_area_weighted`，旧 grid-uniform 产物继续使用 `user_forecast_origin_lead_grid_uniform`。总体、每个 lead 和累计时段均含六项指标与逐项单位/公式策略，使用与训练相同的物理公式。默认 24 步的 24/48/72 小时分别累计前 8/16/24 步全部格点；非默认 horizon 自动生成对应汇总，不平均逐步 RMSE/R²。这仅是本次历史预测窗口评价，与训练结果的完整测试集明确分开。

推理只读取一个 80 步跨度，每通道最多 8 个全网格步分块读取；只保留本次 56 步输入和 24 步参考，模型按 24×48 空间块执行，不物化两年数据或全部滑窗。响应仍包含完整全球数组。

### 地球多模型测试集比较

预测页先选择「地球 / 火星」，再选择「单模型 / 多模型」。地球单模型保留 UTC 历史回测；多模型复用模型选择、标签筛选、综合排名、总体柱状图、逐步曲线与参数矩阵。

`POST /api/earth/predict/training-models/compare` 接受 `{"task_ids":[123,124]}`，要求 2–32 个不同的正整数任务 ID。认证与任务访问规则沿用单模型，日频任务仍返回 `dataset_retired`。服务逐个严格加载 checkpoint、复核当前数据发布及保存的 split 范围，读取训练完成时保存的 test 指标；不训练、不重新推理、不读 Mars 数据。

仅当 dataset ID/版本/fingerprint、window/horizon、target/unit/aggregation、测试分区起止与实际窗口数量均一致时可比较；不一致返回 409 `earth_comparison_incompatible`。返回 `metric_source=verified_checkpoint_test_metrics`，每个 `items[]` 带任务、模型与数据身份，以及六项 `metrics.overall`、任务 horizon 行 `metrics.per_step`、累计时段指标、`split_meta` 与 `export_ref`。误差与百分比升序，R² 原值降序；旧 v1 缺项为空且排除该项排名，不按 0 处理。页面曲线横轴为提前小时，科研导出横轴保留 step，元数据记录 3 小时频率及指标契约。当前科研导出仅开放 RMSE/MAE/R²；Earth 不开放 SSIM；误差分布与 PFI 已在独立训练后诊断开放，暂不开放其科研导出。

训练页「用于预测」与「去模型比较」分别跳转 `mode=earth` / `mode=earth_compare`；旧 `earth`、`trained`、`trained_compare` 链接继续有效。行星、分析方式或账号切换会取消过期请求，Earth 比较结果不进入 Mars 缓存。

验证入口为 `tests/test_earth_3hourly_prediction_routes.py` 的 comparison 回归，涵盖指标来源、无重新推理、权限、发布身份与测试窗口一致性、DU 科研曲线导出。2026-10-08 已运行三小时路由与科研导出回归、前端测试及生产构建；浏览器合成 API 检查四种入口、任务隔离、DU 不重复换算、3–72 小时曲线与 390px 布局。后端 health、生产前端与数据代理均返回 200。测试使用合成发布及图表数据，不代表真实训练精度验收。

### 地球训练后诊断

训练结果页的 `action=test` 现在按行星分流：Earth 调用 `/api/earth/predict/diagnostics`，Mars 保留原有测试链路。`GET /api/earth/predict/diagnostics/context` 只对已完成、权重有效且身份可复核的三小时任务返回可用状态；日频旧任务、缺失或损坏 checkpoint 会给出禁用原因或稳定错误。诊断默认 4 个 test 窗口、最多 5,000 个散点、40 个残差分箱、固定种子 42、PFI 每通道 3 次；服务端上限为 8 窗口、20,000 点、100 箱和 5 次重复。

完整测试指标只消费 checkpoint 保存的 test 指标，缺项不由抽样结果补齐。诊断窗口在任务固定 test 分区内用种子化无放回抽样；模型以 24×48 空间块和批次流式推理，不物化完整测试集。TO3 输出反归一化为 DU，散点是预测/参考配对抽样并带 `y=x`，直方图为 `prediction-reference` 的 DU 残差，对选定窗口的全部 lead/grid 点流式累计；响应包含实际窗口数、UTC 时间范围、全球覆盖、有效点数和策略。

Earth PFI 按 checkpoint 实际输入通道计算，包括历史 TO3，不置换未来目标或真值。基线与置换共享相同窗口、空间块、lead 和评价口径；通道置换保留单窗口时间/空间结构，并在所有空间块复用同一来源映射。重要性是 `permuted RMSE - baseline RMSE`（DU），保留负值，返回每次 RMSE/增量、均值、标准差、种子和实际抽样数；少于两个窗口时明确不可计算。诊断缓存纳入任务/权重 SHA、数据指纹、test 划分、窗口/horizon、通道、normalization、算法版本、参数与种子，命中前重新复核身份；共享推理槽避免不同诊断或预测请求同时抢占 GPU。诊断图目前仅在 Earth 页面展示，科研导出继续消费已有结果快照且不重新推理。算法、精确范围、缓存及验证入口见[地球训练后诊断](earth-post-training-diagnostics.md)。新任务使用任务固定划分，旧三小时按发布 manifest，不迁移或重拟合。


### 身份、缓存与错误码

context、预测及缓存命中前均严格核对任务与 checkpoint 的数据 ID、版本、fingerprint、snapshot、窗口、三小时 UTC 中心规则、split、完整网格、通道、normalization 和权重形状。当前发布再次核对 SHA、时间轴和坐标；推理结束再次复核数据与 checkpoint，变化的结果不写入缓存。

`services/earth_prediction_cache.py` 是独立的进程内 LRU，最多 4 条、总大小上限 128 MiB；以只读 float32 数组保存三种场，不保存展开的 JSON 列表，不写数据库或磁盘，也不进入 Mars 缓存。响应给出 `cache_key`；其 canonical SHA 包含 planet、dataset_id、dataset_version、dataset_fingerprint、task_id、规范起点、全部 24 个 target timestamps 和 checkpoint SHA-256。不同进程不共享缓存，进程重启后清空。旧日频预测不增加缓存字段，也不使用此缓存。

| 情形 | HTTP / code |
| --- | --- |
| 任务未完成或没有可用产物路径 | 409 `earth_prediction_task_not_completed` |
| checkpoint 缺失、损坏、日频格式、权重/窗口/网格/时间规则或任务绑定不符 | 409 `invalid_earth_training_artifact` |
| 绑定发布被替换、缺失、损坏，或当前时间轴/坐标与绑定不符 | 409 `dataset_version_changed` |
| 起点不是 UTC ISO datetime | 422 `invalid_earth_prediction_origin` |
| 起点不在发布轴，或缺少完整 56 步输入/24 步参考 | 422 `earth_prediction_origin_out_of_range` |
| 所选输入或真实参考场缺测/不可读取 | 503 `earth_prediction_data_unavailable` |
| 当前数据包读取权限导致无法验证 | 503 `dataset_unavailable` |
| Earth 任务进入 Mars predict、metrics、PFI 或 compare | 409 `dataset_prediction_not_supported` |
| Earth 任务作为 Mars transfer 来源 | 409 `dataset_transfer_not_supported` |
| 访问他人任务 / 任务不存在 | 403 / 404 |

数据目录查询仍按注册表约定返回 200 和 availability；上述 409 是已绑定训练任务在回测复核时的错误。日频错误与原响应语义保留。

### 三小时回测验证

从 `AstraAtmos_backend/backend/` 执行，每个文件使用新的工作区英文临时目录：

```powershell
$earthPredictionTests = @(
  'tests/test_earth_3hourly_prediction_time.py',
  'tests/test_earth_3hourly_prediction_service.py',
  'tests/test_earth_3hourly_prediction_routes.py',
  'tests/test_earth_prediction_service.py',
  'tests/test_earth_prediction_routes.py',
  'tests/test_earth_training_isolation.py'
)
foreach ($earthPredictionTest in $earthPredictionTests) {
  $earthPredictionTemp = Join-Path $env:TEMP ('.earth-prediction-test-' + [guid]::NewGuid().ToString('N'))
  & "$env:CONDA_PREFIX\python.exe" -B -m pytest $earthPredictionTest `
    -q --basetemp "$earthPredictionTemp" -p no:cacheprovider
  if ($LASTEXITCODE -ne 0) { throw "Prediction regression failed: $earthPredictionTest" }
}
```

早期官方模型回测阶段使用合成 30 天、240 步、完整 240×480 网格和 NetCDF/严格 checkpoint 验证 24 步预测、逐元素参考/残差、DU 指标、惰性读取与空间块、缓存命中和身份变化拒绝；服务和路由成功用例均检查完整全球输出及 JSON 序列化，路由另覆盖认证、稳定错误码与 Mars 隔离。该阶段三小时服务、路由与时间契约 **113 项通过**；日频兼容、注册、checkpoint 与隔离回归 **202 项通过**，其中 checkpoint 在结构校验补强后 **86 项再次通过**。使用指定解释器及新的工作区英文临时目录，仅有既有 NumPy 和 FastAPI/Pydantic 提示。当时的真实 7 天包只有 56 步，不能完成 56→24 回测，该阶段未执行真实两年预测或三小时前端联调。本轮上传模型和浏览器检查范围见下文。

## 2026-10-07 数据构建与注册阶段历史验证

使用项目规定的 AstraAtmos 解释器，真实 `slv/rad` 7 天 smoke 已构建并独立验证。每个变量形状为 `[56,240,480]`，中心时间从 `2020-01-01T01:30:00Z` 到 `2020-01-07T22:30:00Z`，间隔严格三小时；五变量及掩码扫描通过，目标缺失率均为零。实际原始对照检查 2020-01-01、01-04、01-07 的 405 项，包含两极与周期经度接缝。float64 重网格的最大面积均值守恒误差为 `1.14e-13`，输出采用 float32 的舍入误差在 manifest 中单独记录。

相同请求再次执行成功校验并复用；NetCDF 和 manifest 的 SHA-256、修改时间均保持不变。新增构建/验证回归 25 项通过；既有日频构建、读取、注册和路由回归 55 项通过。测试日志存在既有 NumPy 二进制大小 RuntimeWarning，以及日频路由依赖的 FastAPI 弃用提示，没有测试失败。本次没有执行完整两年构建或训练。

后续注册阶段验证：三小时协议、目录及身份/路由测试共 158 项通过（其中三小时新增 99 项）；日频读取、注册、训练数据、checkpoint 与总览回归 135 项通过。扩展的任务身份、迁移与预测回归为 87 passed / 7 failed，失败均来自原有 `test_training_dataset_identity.py`：六项 SQLite fixture 缺少已存在的 `queued_at` 字段，一项仍预期拒绝已经支持的 Earth uploaded 模型。用接入前 HEAD 的四个数据集服务模块重跑原身份测试复现相同七项失败，未在本轮修改旧测试或任务数据库。实际环境变量读取和真实 smoke 的目录/详情 API 返回 200，两份旧日频包与新包同时为 available。

## 上传链路验证

2026-10-07 上传链路最终后端回归为 **435 passed、5 warnings，145.16 秒**，覆盖下列 17 个文件。新增上传契约测试含 80 项：严格 spec/类型/单位/轴、16 种通道组合、错误 dtype/形状/设备/梯度、具体 dataset_id 兼容性与稳定错误码、隔离超时、当时创建任务流程的实际参数 dry-run、源码冻结与队列恢复、合成训练/验证/测试指标、checkpoint 身份及权重隔离、缺失原源码时嵌入副本的摘要核对、完成门禁的独立 CPU 重载，以及完整全球历史回测和跨 split 拒绝。模板的 CPU 与本机可用 CUDA 设备检查均执行通过。此处为历史验证记录；当前任务创建已取消重复的实际配置 dry-run。

从 `AstraAtmos_backend/backend/` 执行。每次调用创建唯一临时根目录，将 TEMP/TMP 和 pytest basetemp 均放在工作区根目录下；不要复用既有目录或 checkpoint：

```powershell
$earthUploadTemp = Join-Path $env:TEMP ('.earth-upload-check-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $earthUploadTemp -ErrorAction Stop | Out-Null
$earthPreviousTemp = $env:TEMP
$earthPreviousTmp = $env:TMP
$earthUploadTests = @(
  'tests/test_earth_3hourly_uploaded.py',
  'tests/test_earth_3hourly_training_service.py', 'tests/test_earth_3hourly_artifact.py',
  'tests/test_earth_3hourly_training_contract.py', 'tests/test_earth_3hourly_training_runner.py',
  'tests/test_earth_3hourly_training_data.py', 'tests/test_earth_3hourly_prediction_service.py',
  'tests/test_earth_3hourly_prediction_time.py', 'tests/test_earth_3hourly_prediction_routes.py',
  'tests/test_user_model_downloads.py', 'tests/test_earth_uploaded_model.py',
  'tests/test_earth_uploaded_upload_flow.py', 'tests/test_earth_training_service.py',
  'tests/test_user_model_validator.py', 'tests/test_user_model_schema.py',
  'tests/test_uploaded_model_ls_inference.py', 'tests/test_earth_training_isolation.py'
)
try {
  $env:TEMP = $earthUploadTemp
  $env:TMP = $earthUploadTemp
  & "$env:CONDA_PREFIX\python.exe" -B -m pytest @earthUploadTests `
    -q --basetemp (Join-Path $earthUploadTemp 'pytest') -p no:cacheprovider
  if ($LASTEXITCODE -ne 0) { throw 'Earth upload regression failed' }
} finally {
  $env:TEMP = $earthPreviousTemp
  $env:TMP = $earthPreviousTmp
}
```

环境没有 pytest-asyncio；`tests/test_uploaded_training_contract.py` 和 `tests/test_user_model_service.py` 已另用固定解释器直接执行并通过，不能将其原生 async 函数当作普通同步 pytest 测试。Mars `test_uploaded_model_runner.py` 另有 **4 项既有失败**，在接手基线上复现：`test_prepare_tensors_builds_uploaded_runner_dataset_from_mcd_overview`、`test_prepare_tensors_aligns_ls_to_history_when_window_differs_from_horizon`、`test_prepare_tensors_rejects_partial_file_ls_as_unaligned_timeline`、`test_prepare_tensors_builds_uploaded_runner_dataset_from_raw_3h_mcd`。这些断言仍假设旧窗口数量/年份分区，与已有完整跨 MY 时间轴行为冲突；本轮没有改写它们或 Mars runner。

前端相关 5 文件 **45 passed**，生产构建通过（保留既有大 chunk 提示）。从 `frontend/` 运行 `node --test src/pages/ModelTrainingPage/earthThreeHourUploaded.test.js src/pages/ModelTrainingPage/earthTrainingConfig.test.js src/pages/ModelTrainingPage/earthTrainingDatasetStructure.test.js src/pages/ModelTrainingPage/trainingDefaults.test.js src/pages/ModelTrainingPage/uploadedModelParams.test.js`，然后 `npm run build`。Playwright 对构建产物检查了三小时上传模型的可用/不可用/未知提示与原因、开始按钮、错误数据集结论拒绝、TO3 必选、自定义参数键盘切换、1440px/390px 布局、三小时专用模板/说明，以及日频/Mars 原模板下载地址。移动端布尔参数隐藏 checkbox 的横向溢出已修复。浏览器使用 mock API，所有非 GET 请求均被阻止，未创建真实训练任务；不代表真实上传、训练及预测的浏览器全流程验收。

完整两年真实包只读集成 **1 passed，387.76 秒**。检查实际 descriptor、5848 步、5611 个 split 内历史起点及每个 split 的真实 80 步五变量 24×48 块，确认形状、float32 和有限性，读取前后包签名一致。可复现命令如下，先将 `$earthUploadPackage` 设为已配置并独立验证的外部完整发布目录。环境变量只作用于本次进程，不改写 `.env` 或开发默认值：

```powershell
$earthReadonlyTemp = Join-Path $env:TEMP ('.earth-upload-readonly-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $earthReadonlyTemp -ErrorAction Stop | Out-Null
$earthReadonlyPreviousTemp = $env:TEMP
$earthReadonlyPreviousTmp = $env:TMP
$earthReadonlyPreviousDir = $env:ASTRAATMOS_EARTH_MERRA2_3HOURLY_DIR
$earthReadonlyPreviousOptIn = $env:ASTRAATMOS_TEST_EARTH_3HOURLY_PACKAGE
try {
  $env:TEMP = $earthReadonlyTemp
  $env:TMP = $earthReadonlyTemp
  $env:ASTRAATMOS_EARTH_MERRA2_3HOURLY_DIR = $earthUploadPackage
  $env:ASTRAATMOS_TEST_EARTH_3HOURLY_PACKAGE = $earthUploadPackage
  & "$env:CONDA_PREFIX\python.exe" -B -m pytest `
    tests/test_earth_3hourly_uploaded_readonly.py -q -s `
    --basetemp (Join-Path $earthReadonlyTemp 'pytest') -p no:cacheprovider
  if ($LASTEXITCODE -ne 0) { throw 'Read-only Earth package integration failed' }
} finally {
  $env:TEMP = $earthReadonlyPreviousTemp
  $env:TMP = $earthReadonlyPreviousTmp
  $env:ASTRAATMOS_EARTH_MERRA2_3HOURLY_DIR = $earthReadonlyPreviousDir
  $env:ASTRAATMOS_TEST_EARTH_3HOURLY_PACKAGE = $earthReadonlyPreviousOptIn
}
```

运行时复查：后端 `/health`、前端页面及代理 `/api/datasets` 均为 HTTP 200。日频 v1/v2 与三小时 descriptor 均 `available`；三小时 ID/version/fingerprint 与上文完整包摘要一致，training profile 声明 `official/uploaded`、56→24。`default_earth_dataset_id` 保持 `earth_merra2_daily_v2`。README 及 docs 的 350 个相对文件链接检查通过，Git diff 空白检查通过。

本轮没有重建、迁移或修改真实包，没有执行完整真实生产训练，也没有真实任务的历史预测或精度验收。合成数据 smoke 仅验证执行与产物链路；只读集成仅验证数据读取契约。高分辨率预测的传输/浏览器内存、任意用户网络的 GPU 容量与收敛仍需按实际模型验证。

独立的三小时 **36×72（5°）** 训练发布、完整全球窗口与上传模型契约见 [5° 数据集](earth-merra2-5deg.md)；本文件的原 0.75° 契约继续兼容。
