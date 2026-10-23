# Earth MERRA-2 DLinear 训练与历史预测接入 Implementation Plan

> **2026-09-27 修订（本版为实施依据，替代此前正文）**：本计划已按当前 `earth_merra2_daily_v2` 真实数据契约与源码重写，并**扩大交付范围**：除官方 DLinear 网页训练外，**同一任务内交付从已完成 Earth 任务与 checkpoint 恢复的历史预测**（选择历史预测起点、查看未来 3 天逐日预测/参考/残差场与误差指标，单位 DU）。
>
> 此前版本正文中的 v1 假设**全部失效**，不得再作为实施依据：`earth_merra2_daily_v1` 身份、31×49、纬度 −60…60、经度 −120…120、4°/5° 步长、`coverage=regional`、`wrap_longitude=false`、8/15/8 纬带行数、“不周期拼接”，以及“预测仍关闭、Earth 预测一律 409”的阶段边界。
>
> 现行契约（以源码与实测为准）：默认数据集 `earth_merra2_daily_v2` / `v2`，全球 36×72、5°×5°，中心纬度 −87.5…87.5、经度 −177.5…177.5，单元边界 ±90°/±180°，`coverage=global`、`wrap_longitude=true`，731 个 UTC 日（2020-01-01…2021-12-31），五变量 `TO3/U10M/V10M/T2M/SWGDN`，划分 train 366 / validation 181 / test 184 天，7→3 窗口 357 / 172 / 175，归一化只拟合训练期。v1 原包与 API 保留为只读兼容身份，**不**把 v1 请求映射到 v2，也**不**把 Earth 请求映射到火星。

> **For agentic workers:** 默认顺序执行，不自行创建额外对话，不自动提交或推送；用户已有指令优先于流程建议。用户要求先修订本方案中与 v2 冲突的内容，再实施。

**Goal:** 在训练页选择 `earth_merra2_daily_v2`，用过去 7 天五个可选输入训练预测未来 3 天 TO3 的官方 DLinear；训练完成后在同一任务的「用于预测」进入预测页，选择有效历史预测起点，查看未来 3 天逐日预测场、参考场、残差场（DU）与误差指标。

**Architecture:** 数据集注册表负责发布身份与可用性，Earth 训练契约负责能力限制与严格参数校验；Earth 使用**独立的**数据准备与训练路径（`EarthOzoneWindows.from_release` + `models/training_scripts/earth_daily.py`），复用共享 `TrainingService` 的任务生命周期（进度、停止、日志、历史、标签）与现有 DLinear 工厂。训练产出一个包含权重、归一化、坐标、日期划分、完整模型配置与 DU 指标的版本化单文件 checkpoint；**父进程在标记完成前用 CPU 严格重载校验该产物**。预测由新增的 Earth 专用服务与路由从 checkpoints 恢复模型，按用户选择的历史预测起点读取此前 7 天输入，参考场取自同一发布数据集；Mars 推理、比较、PFI 与迁移路径明确拒绝 Earth 任务，绝不回落。

**Tech Stack:** FastAPI、Pydantic v2、SQLAlchemy/SQLite、NumPy/xarray/netCDF4、PyTorch；React 19、现有训练上下文与预测图表组件；pytest、Node 内置测试、Vite build、真实浏览器与服务联调验收。

---

## 0. 相对旧版的关键修订清单

| 旧版内容 | 修订后 |
| --- | --- |
| 数据集 `earth_merra2_daily_v1` | `earth_merra2_daily_v2`（v1 仅保留只读兼容，不映射） |
| 网格 31×49、纬 −60…60、经 −120…120 | 36×72、纬 −87.5…87.5（边界 ±90°）、经 −177.5…177.5（边界 ±180°） |
| `coverage=regional`、`wrap_longitude=false` | `coverage=global`、`wrap_longitude=true`，含极区与经度周期接缝 |
| 发布指纹 `74ce…d31` + v1 两个 SHA | v2 manifest `935c…702`、data `d280…396`，指纹由注册表计算得出 |
| 纬带 8/15/8 行假设 | 全部 36 行重新分配；全球地图按真实 cell bounds 绘制 |
| 输出 `Y=[B,3,1,31,49]` | `Y=[B,3,1,36,72]`，输入 `X=[B,7,C,36,72]` |
| 阶段边界：Earth 预测不开放、`trained_prediction=false` | **首期交付历史预测**：`trained_prediction=true`，Earth 预测路由按日期回测已开放 |
| 错误表 `dataset_prediction_not_supported` 覆盖全部 Earth 预测 | 仅覆盖**火星**推理/比较/PFI/迁移入口；Earth 自身的日期越界、指纹变化、权重损坏、无权限有各自错误码 |
| 无预测缓存身份要求 | 缓存键与比较入口必须区分星球与数据集身份；首期不要求 Earth/Mars 混合比较，也不要求无参考真值的未来外推 |

## 1. 交付范围与前置条件

项目根目录为 `AstraAtmos`。下文路径相对项目根目录；后端路径均以 `AstraAtmos_backend/backend/` 为前缀。本计划属于项目功能文档，保存在可追踪的 `docs/plans/`。

实施前先读父目录 `AGENTS.md`、[README](../../README.md) 的“快速接手 / 模块与代码入口 / 关键业务链路 / 当前功能边界 / README 维护约定”、[数据集注册协议](../dataset-registry.md)、[Earth 小包](../earth-compact-dataset.md)、[Earth 总览](../earth-overview.md)、[共用分析工作台](../earth-analysis-workbench.md)。接口接线以最新源码与已实现协议为准。禁止批量删除文件或目录。

### 必须交付

- 训练数据集选项与 `model_source=official/uploaded` 分开；Earth 首期仅支持官方 DLinear。
- 固定输入 7 天、输出 3 天；TO3 必选历史输入与唯一目标；U10M、V10M、T2M、SWGDN 可独立勾选，允许只用 TO3。
- 展示真实日期划分、样本数量（357/172/175）、全球 36×72 网格、各输入单位与训练集归一化说明。
- 复用现有启动、进度、日志、停止、历史记录、命名、私有标签与单条任务管理。
- 保存完整单文件 checkpoint；CPU 严格重载后产生一致预测及 DU 数值；训练完成前必须重载校验。
- 完成一次真实 v2 小包训练，取得有限验证与测试指标；刷新页面后记录仍可查询。
- 预测页从已完成 Earth 任务恢复模型，返回可选日期范围、3 天真实日期、真实经纬网格、预测/参考/残差场（DU）与指标。
- 日期越界、数据集指纹变化、权重损坏、任务无权限等返回明确错误，且**不**回退到火星预测。
- 后端、前端均阻止 Earth 任务进入 Mars 推理、比较、PFI 与迁移来源路径。

### 阶段边界（首期不做）

不做 Earth/Mars 混合比较、无参考真值的未来日期外推、SPHERE、ConvLSTM、上传模型、迁移学习、自选日期划分、持久性基线比较与 Earth 预测结果持久化缓存。训练结束后的固定测试集指标属于本阶段。

不改动火星数值单位、MY/Ls、数据读取与 checkpoint 格式，不修改全局地图网格，不为完成训练而下载完整 MERRA-2 或重新生成小包。

## 2. 已核实的代码事实（2026-09-27 快照）

| 入口 | 当前事实 | 处理 |
| --- | --- | --- |
| `services/dataset_identity.py` | `DATASET_IDS` 四项；`REGISTERED_DATASET_IDS`；`resolve_dataset_id` 拒绝客户端身份字段；`require_training_dataset()` 仅准许 `MARS_DATASET_IDS` | 保留 Mars runner 的拒绝行为；新增轻量纯识别 `is_earth_dataset_id` / `is_earth_training_task`，Earth 走独立分支 |
| `services/dataset_registry.py` | 固定 v1/v2 SHA；`get_earth_overview_snapshot(dataset_id, expected_fingerprint)`；`_earth_release` 按签名缓存一份 `VerifiedEarthRelease`；`build_training_binding()` 仅返回 Mars 绑定 | 抽出中性 `get_earth_snapshot()`（保留 wrapper 契约）、Earth 训练绑定、`training_profile` |
| `services/earth_dataset_metadata.py` | `VerifiedEarthRelease(metadata, signature, dates, latitude, longitude, fields)`，fields 只读 float32 全 5 通道；manifest 已在 `read_earth_release` 中逐项校验（含 train 统计量） | 训练直接从同一 release 取数，验证后不另开任意 NetCDF |
| `services/earth_dataset.py` | `EarthOzoneWindows(path, split, window, horizon)` 总是读 5 通道、每次自行拟合标准化 | 扩展 `from_release(...)`：通道选择、注入/拟合归一化、`input_channels`、`input_units`、`denormalize_ozone`；保留原 path 用法 |
| `services/training_channels.py` | Mars 通道名 U/V/D/S/T；`_training_dataset()` 调用 `require_training_dataset`；普通超参数全部转 CLI 参数 | Earth 先用独立白名单校验，再决定是否进入旧 Mars 规范化；Earth 通道不得被旧逻辑过滤为空 |
| `training_backbones/model_zoo.py` | `build_forecaster(architecture, input_channels, selected_channels, hidden_dims, height, width, window, horizon, use_sphere, architecture_params)`；DLinear backbone 内部 `feature_dim=5`，通道数不等时 `ChannelProjector` 提供可训练 1×1 卷积；forward 需要 `ls` 位置参数但 `use_phase_warp=False` 时不使用 | 已实测 C=1…5 前向/反向/参数形状（见验证记录）；保存完整 adapter state_dict 与构造参数 |
| `models/training_scripts/demo3.py` | 官方训练含 Mars 数据准备；末尾保存裸 state_dict | Earth 不经过该数据路径，不把新容器写给旧 Mars 加载器 |
| `services/model_artifacts.py` | `is_valid_model_weight_file()` 只检查存在且非空 | Earth 完成前增加契约校验与严格重载；非空垃圾文件不算成功 |
| `services/training_weight_service.py` | 上传权重校验只确认可被 Torch 读为 dict | 显式识别并拒绝 Earth checkpoint 容器进入 Mars 迁移 |
| `services/inference_service.py` | 存在公共预测上下文入口与旧 `get_test_results()` 直接加载路径 | 两处及 action 路由都要在 Mars 数据准备与缓存访问**之前**拦截 Earth |
| `schemas/datasets.py` | `DatasetDescriptor` 无训练 profile 字段 | 增加可空强类型 `training_profile` |
| `frontend/src/services/datasets.js` | `fetchDatasets({signal})` 返回 `{items}` | 训练页数据集列表复用该客户端，不在 `api.js` 建第二套目录客户端 |

源码可能被并行工作更新；实施时如已有等价实现，复用并统一命名，不创建第二套注册表或数据副本。

## 3. 固定的科学与数据契约（v2）

### 3.1 数据身份、网格与时间

| 项目 | 首期值 |
| --- | --- |
| dataset_id / version | `earth_merra2_daily_v2` / `v2` |
| 数据格式 schema | `astraatmos_earth_daily_v1`（文件协议名，非版本号） |
| 时间 | 2020-01-01 至 2021-12-31，连续 731 个 UTC 日；`proleptic_gregorian` |
| 网格 | `[36, 72]`，维度 `[lat, lon]`；中心纬 −87.5…87.5、经 −177.5…177.5，均递增 |
| 分辨率 | 5°×5°，全球 |
| 空间边界 | 单元边界 ±90°/±180°；`coverage=global`、`wrap_longitude=true` |
| 包内通道 | TO3、U10M、V10M、T2M、SWGDN |
| 物理单位 | DU、m s-1、m s-1、K、W m-2 |
| 目标 | TO3，反归一化后为 DU |
| 发布 SHA | manifest `935ca37bd3064772a370db6873b605e12b85c031281c2b34d8fbc49ab11f0702`、data `d280a17cb291e1568ccedd1638cb2fadcc5cb8d89bb8ed0d4d51987e8e181396` |
| 指纹 | 由 `build_dataset_fingerprint(dataset_id, version, manifest_sha, data_sha)` 在服务端计算，训练页与 checkpoint 均不维护第二份 |

### 3.2 日期划分与窗口

划分固定来自 manifest；页面只读展示，不提供随机划分或百分比滑块。

| 划分 | 日期范围（含首尾） | 天数 | 7→3 窗口数 | 首个目标区间 | 末个目标区间 |
| --- | --- | --- | --- | --- | --- |
| train | 2020-01-01…2020-12-31 | 366 | 357 | 2020-01-08…01-10 | 2020-12-29…12-31 |
| validation | 2021-01-01…2021-06-30 | 181 | 172 | 2021-01-08…01-10 | 2021-06-28…06-30 |
| test | 2021-07-01…2021-12-31 | 184 | 175 | 2021-07-08…07-10 | 2021-12-29…12-31 |

样本数公式：`days - window - horizon + 1`。输入与目标全部位于所属划分内，不借用前一划分末尾 7 天。输入截止日为 forecast origin，未来第 1 天为次日。所有辅助变量也只读取过去 7 天，不使用未来实测温度、风或辐射。

输出维度严格为 `X=[B,7,C,36,72]`、`Y=[B,3,1,36,72]`。三天同时输出，不做递归滚动。DataLoader 可打乱训练窗口；validation/test 不打乱。

### 3.3 通道与归一化

- 合法可选输入仅 `U10M`、`V10M`、`T2M`、`SWGDN`；缺省选中四个辅助输入；显式 `[]` 表示仅 TO3，不能被缺省逻辑替换。
- 请求顺序不决定模型通道顺序。规范顺序固定为 `TO3` 加按 `U10M,V10M,T2M,SWGDN` 排列的选中变量；重复或未知通道返回 422。
- 每个有效通道在 **366 个训练日的全部 36×72 网格**上拟合一个均值与标准差；float64 累加，保存 float32 有效数值，`ddof=0`。
- 标准差小于 `1e-6` 时有效缩放值设为 1，记录 `constant_channel_mask`；不把常量通道除成 NaN。
- validation/test 与 checkpoint 恢复复用保存值，不重算；目标使用输入 TO3 对应的均值与有效标准差。
- 不做 MinMax、对数、空间插值、缺值补零或预测负值裁剪。包内已有 NaN/Inf/非法填充值时沿用包校验失败行为。
- 保存 `method=per_channel_standard`、`fit_split=train`、`fit_date_start=2020-01-01`、`fit_date_end=2020-12-31`、`ddof=0`、`epsilon=1e-6`、`channel_order`、`mean`、`scale`、`constant_channel_mask`、`target_channel_index=0`。
- 实测训练期统计量（manifest 与重算一致）：TO3 mean 286.762064 / scale 47.314837；U10M −0.001669 / 4.970813；V10M 0.220144 / 4.115977；T2M 278.888111 / 21.438948；SWGDN 158.234657 / 106.847826。

### 3.4 训练与指标

首期固定 Adam 与标准化空间 MSE，所有网格等权；不把该损失或误差称作面积加权、全球平均。网页默认 epochs=10、batch_size=8、learning_rate=0.001、seed=11、early_stopping_patience=0、linear_hidden_layers=2。0 表示不早停，但仍保存最佳验证权重。

只用 validation MSE 选择 best epoch：严格降低才更新，相同值保留较早 epoch。恢复 best state 后才执行一次最终 test。训练与验证均按误差元素总数聚合，不对大小不同的 batch 均值直接平均。

结果保留 validation/test 的总体与第 1/2/3 天 RMSE、MAE；RMSE/MAE 单位为 DU，MSE 如保留须为 DU²。指标从反归一化后的预测/参考值计算，float64 聚合。保存 `aggregation=forecast_origin_lead_grid_uniform` 与各划分 `window_count`，不表述为独立日样本数。

### 3.5 历史预测契约（首期）

- 预测起点必须是发布数据集中存在的日期，且该日期之前有 7 天输入数据、之后有 3 天参考数据：可选范围固定为 `2020-01-08 … 2021-12-28`（相对 731 日轴，索引 7…727）。服务端返回该范围与全部可选日期，前端只从其中选择。
- 三天目标日期为 `origin+1, origin+2, origin+3`，参考场取自同一发布数据集的同名日期。
- 输入窗口为 `origin-6 … origin`（含 origin），仅使用 TO3 与用户训练时选中的辅助通道，顺序取自 checkpoint 的 `input_channel_order`。
- 输出：`prediction_du`、`reference_du`、`residual_du = prediction - reference`，形状 `[3, 36, 72]`，每场 3 个逐日场；单位为 DU。
- 指标：总体与逐日 RMSE、MAE（DU），由本次 3 天窗口 float64 聚合，`aggregation=user_forecast_origin_lead_grid_uniform`，并标明 `reference_available=true`。
- 不提供无参考真值的未来外推；起点越界返回 422 `earth_prediction_origin_out_of_range`。

## 4. 请求、能力与错误契约

### 4.1 注册表训练/预测 profile

在 `schemas/datasets.py` 为描述符增加可空的强类型 `training_profile`。Mars 保持 null；Earth 返回：

```json
{
  "profile_id": "earth_daily_dlinear_v1",
  "model_architectures": ["dlinear"],
  "model_sources": ["official"],
  "target": "TO3",
  "target_unit": "DU",
  "window": 7,
  "horizon": 3,
  "step_unit": "day",
  "optional_channels": ["U10M", "V10M", "T2M", "SWGDN"],
  "default_selected_channels": ["U10M", "V10M", "T2M", "SWGDN"],
  "supports_sphere": false,
  "supports_transfer_learning": false,
  "supported_planet": "earth"
}
```

`capabilities.training` 与 `capabilities.trained_prediction` 在任务服务、runner、完成校验与预测路由全部就绪后才置 `true`；可用性继续由 `availability` 单独表达。客户端按钮还必须检查 `availability=available`。不可用时维持 profile，但不给伪造的日期、坐标或可训练状态。

### 4.2 沿用现有启动接口

`POST /api/training/start`（顶层 `dataset_id` 与 legacy `hyperparameters.training_dataset` 均有效，冲突沿用拒绝规则）：

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

`model_script` 为兼容现有客户端保留；服务端按 dataset/model_source 选择 `earth_daily.py` 并写入任务，不按客户端脚本名决定 Earth 执行代码。

Earth 参数用独立 Pydantic 严格模型或等效校验，拒绝 bool 充当 int、非整数、非有限学习率。范围：epochs 1…1000、batch_size 1…64、learning_rate 0＜lr≤1、seed 0…2³²−1、patience 0…200、linear_hidden_layers 1…4。window/horizon 必须为 7/3；不合法值不静默夹到合法范围。

允许字段只含示例超参数、可选 `model_source="official"` 与 `transfer_learning=false`。未知字段、内部 `_` 字段、上传/迁移路径、设备路径、自定义划分、归一化参数、目标替换全部拒绝。服务端自身的 `_data_source` 在校验后加入。

### 4.3 错误分类

`DatasetRequestError` 沿用结构：`detail={code,message}`，有可用性原因时附公开 reason，不泄漏路径。

| 情形 | HTTP / code | 副作用 |
| --- | --- | --- |
| 非 DLinear、uploaded、SPHERE=true、transfer_learning=true、uploaded_model_id 非空 | 409 `dataset_training_configuration_not_supported` | 不建任务、不加载用户模型、不启动进程 |
| 错误窗口、通道、范围、字段类型或额外字段 | 422 `invalid_earth_training_parameters` | 同上 |
| 包缺失、哈希不符、内容非法 | 503 `dataset_unavailable` | 同上；UI 显示原因与刷新入口 |
| 父进程绑定后、子进程启动前包身份改变 | 子进程失败，任务 `failed`，`dataset_version_changed` 或 `dataset_unavailable` | 不生成成功产物，不回退到火星 |
| Earth 任务进入 Mars 推理 / 比较 / PFI / action=test / 迁移 | 409 `dataset_prediction_not_supported` / `dataset_transfer_not_supported` | 不读 Mars 数据、不进入旧预测缓存 |
| checkpoint 不完整、不匹配或不能严格重载 | 任务 `failed`，`invalid_earth_training_artifact` | 不把非空权重当成功 |
| Earth 预测起点越界 / 非日期 / 缺失 | 422 `earth_prediction_origin_out_of_range` / `invalid_earth_prediction_origin` | 不读数据、不建缓存 |
| Earth 预测时数据集指纹变化 | 409 `dataset_version_changed` | 不返回旧指纹结果 |
| Earth checkpoint 损坏或与任务身份不符 | 409 `invalid_earth_training_artifact` | 不加载、不返回部分结果 |
| 任务不属于当前用户 / 无法访问 | 403 / 404 沿用现有权限契约 | 不泄漏任务是否存在以外的信息 |
| 对未完成或失败的 Earth 任务请求预测 | 409 `earth_prediction_task_not_completed` | 不加载权重 |

未知 dataset、双入口冲突、客户端身份字段、用户认证与标签权限保持第一阶段契约。不把所有 `ValueError` 都改成 422；只转换 Earth 参数与预测起点异常。

## 5. 数据、进程与产物接口

### 5.1 复用已验证快照

注册表增加 `get_earth_snapshot(dataset_id, expected_fingerprint=None)`：返回 `VerifiedEarthRelease`，只从受控目录与发布哈希读取。省略 expected 时取得当前受校验 release；给定 expected 时严格匹配。未知 ID、非 Earth、不匹配及不可用分别有稳定错误。现有 `get_earth_overview_snapshot()` 保持签名与错误契约，内部调用通用方法，避免破坏总览测试。

父服务使用 `app.state.dataset_registry`，不为每个训练/预测请求复制五场原始数组。训练子进程有独立地址空间，各持一份本进程的已验证 release。归一化不就地修改只读 fields，避免总览展示标准化值。

`build_training_binding()` 增加 Earth 分支，从同一 release 构造身份列：

- id/version/fingerprint 来自 registry，`dataset_identity_status=verified`。
- `dataset_snapshot` 为 JSON，含 planet、schema、两种 SHA、time、grid、variables、channel_order、splits，及 `binding_basis=server_registry`、`version_status=verified`。
- snapshot 表示所用发布版本，不承载会变化的训练进度，不替代 checkpoint 中的训练契约。
- Mars 分支继续旧逻辑；旧的 `require_training_dataset()` 不得把 Earth 意外放入 `demo3.py` 或 `user_model_runner.py`。

开启训练后，`EARTH_APPLICATION_LIMITATIONS` 只保留物理范围说明与“Earth 未来外推/混合比较尚未接通”一类**当前**能力说明；manifest 中“Training and prediction remain disabled”这类构建期说明通过 `_STALE_LIMITATION_MARKERS` 过滤，**不修改已发布 manifest 字节与其固定 SHA**。

### 5.2 数据窗口 API

扩展现有 `EarthOzoneWindows`，保留 `EarthOzoneWindows(path, split=..., window=..., horizon=...)` 的独立脚本行为。新增契约：

```python
train = EarthOzoneWindows.from_release(
    release, split="train", window=7, horizon=3,
    selected_channels=["U10M", "T2M"], normalization=None,
)
validation = EarthOzoneWindows.from_release(
    release, split="validation", window=7, horizon=3,
    selected_channels=["U10M", "T2M"], normalization=train.normalization,
)
test = EarthOzoneWindows.from_release(
    release, split="test", window=7, horizon=3,
    selected_channels=["U10M", "T2M"], normalization=train.normalization,
)
```

只有 train 且 `normalization=None` 时拟合统计量；validation/test 无统计量时报错。给定 normalization 时严格验证通道顺序、长度、有限值、正 scale 与拟合范围，只应用不重拟合。`train.normalization` 为可 JSON 序列化的独立字典。

类同时提供 `input_channels`（规范顺序）、`input_units`、`dates`（所选划分每日坐标）、`lat`、`lon`、`target_channel_index`、原有 `mean`/`std` 兼容属性与 `denormalize_ozone(values)`（保持 Tensor 或 ndarray 类型，不隐式转 CPU）。数据与标签返回 float32，切片复制避免调用方修改快照。

只构建所需标准化数据，不预先复制 704 个完整窗口，不把所有网格点预展开成百万条样本。

### 5.3 服务端训练 spec 与子进程

新增内部 schema `earth_training_spec_v1`，字段固定为 `schema`、`task_id`、`dataset_binding`（五列，snapshot 在内存中为对象）、`hyperparameters`（规范且无 `_` 字段）。完成任务 flush 后由服务端构造。

用环境变量 `ASTRAATMOS_EARTH_TRAINING_SPEC` 传入该 JSON，避免把长 JSON、路径或私有身份字段塞进现有 hyperparameter CLI。Earth 进程只接收 `--output_path`；不再无差别 `build_hyperparameter_args()`。

进程沿用服务端受控 `ASTRAATMOS_EARTH_MERRA2_DIR`；父服务与子进程必须来自同一配置；相对路径由 backend 根目录解析，不依赖 Popen 的 `cwd=MODELS_DIR`。测试可通过构造函数注入临时 registry，生产不接受客户端指定期望哈希。

runner 解析环境 spec 后新建本进程 registry，校验 task id、规范参数与发布身份，再取得同次校验快照训练。环境 spec 不含客户端提供的目录；runner 缺 spec 时不得自行套用默认 Mars 数据。

进程规范：单进程 DataLoader，`num_workers=0`，随机种子覆盖 random/NumPy/Torch，训练 sampler 使用确定 generator。`device` 由服务端自动选择 CUDA/CPU 并记录实际值；自动化测试可在内部 `run_training(..., device="cpu")` 注入 CPU，不向网页开放任意执行参数。

### 5.4 单文件 checkpoint

保持现有 `build_task_output_path(task_id, name)` 的 `.pth` 路径与唯一 task id。一个文件含完整契约，避免新增旁文件给重命名、复制、删除带来一致性问题。

容器只保存 tensor 与 `weights_only=True` 可读取的普通 Python 基本类型；NumPy 数组、datetime、Path、模型对象先转换为列表或字符串。

| 字段 | 必存内容 |
| --- | --- |
| artifact_schema | `astraatmos_earth_forecast_checkpoint_v1` |
| model_state_dict | 完整 ForecasterAdapter，包括 projector 与 backbone 的 CPU tensors |
| model_config | architecture=dlinear、implementation_id=`astraatmos_gridpoint_dlinear_v1`、input_channels、selected_channels、hidden_dims、height/width=36/72、window/horizon=7/3、use_sphere=false、architecture_params |
| dataset_binding | 与任务五列一致的 identity，snapshot 为对象，含发布 SHA、日期与完整经纬数组 |
| training_contract | target=TO3、target_unit=DU、input_channel_order、input_units、tensor_layout=`BTCHW`、step_unit=day、window/horizon、strict_split_windows=true、split_window_counts |
| normalization | 第 3.3 节规定的完整统计量及 fit 日期；TO3 统计值索引明确 |
| run | task_id、seed、optimizer=Adam、loss=`normalized_mse_grid_uniform`、实际 hyperparameters、best_epoch、epochs_completed、best_validation_loss、stopped_early、run_complete=true、UTC 创建时间、Python/Torch 版本、device |
| metrics | schema=`earth_training_metrics_v1`、target/unit、aggregation、validation/test 各自 window_count、overall、by_lead |

`model_config` 必须直接足以调用 model_zoo 工厂：`hidden_dims` 保存 `[64,64,64]` 以保持完整调用契约，但页面不把它展示为 DLinear 生效参数；DLinear 的结构参数只有 `linear_hidden_layers`。投影尺寸由实际输入通道数决定，不硬编码为 5。

`by_lead` 为按 lead_day 1、2、3 排列的列表，每项至少 `{lead_day, rmse, mae}`；overall 至少 `{rmse, mae}`。所有指标有限；标准化 validation loss 与 DU 指标分开标明。

训练期间把最佳 state 深拷贝到 CPU 内存；恢复 best state、完成验证/测试后写同目录明确临时文件，对临时文件严格重载及同输入输出比对，全部成功才 `os.replace()` 原子替换为最终文件。停止/异常/OOM 不发布最终产物，不复用以前任务的输出文件；用户已停止时完成回调不得覆盖成 completed（保持现有 failed + “Stopped by user” 约定）。

新增 `load_earth_training_artifact(path, expected_binding=None, expected_hyperparameters=None, expected_task_id=None)`：验证上述字段、shape/窗口/通道/坐标与 snapshot 一致性、归一化与指标合法性，构造模型并 `load_state_dict(..., strict=True)`。另提供 `create_earth_forecaster(input_channel_order, linear_hidden_layers=2)` 与 `earth_forward(model, inputs)`（屏蔽零时间占位的旧签名）。

`TrainingService` 子进程退出码为 0 后在线程中执行此校验，成功才写 completed；Earth 数据库 metrics 直接复制 `checkpoint.metrics`，不从 Mars 正则日志凑另一套指标。`model_available` 仍是轻量字段，列表接口不批量加载 Torch；新增 `trained_prediction_supported`（Earth 为 true）与旧字段语义分离。

### 5.5 Earth 历史预测服务

新增 `services/earth_prediction_service.py`：

- `get_earth_prediction_context(task, registry)`：校验任务为已完成 Earth 任务、`model_available`、产物存在；返回可选日期范围（`start=2020-01-08`、`end=2021-12-28`、`count`、`dates` 列表）、三天相对偏移、变量与单位、网格（36×72 与完整经纬数组）、split 划分与 checkpoint 中记录的 `input_channel_order`、`linear_hidden_layers`、指标摘要。
- `run_earth_prediction(task, origin, registry, device=None)`：严格重载 checkpoint（校验 dataset fingerprint 与任务身份一致）→ 从 release 取 `origin-6…origin` 输入并按保存的归一化标准化 → 模型 eval + `torch.no_grad()` → 反归一化为 DU → 参考场取 `origin+1…origin+3` → 返回 `prediction_du`、`reference_du`、`residual_du`（`[3,36,72]` 列表）、`dates`、`grid`、`metrics`。
- 指纹不一致返回 409 `dataset_version_changed`；checkpoint 损坏返回 409 `invalid_earth_training_artifact`；未完成任务返回 409 `earth_prediction_task_not_completed`。
- 首期不写持久化预测缓存，也不创建 Mars 缓存条目；返回体显式带 `planet="earth"`、`dataset_id`、`dataset_version`、`dataset_fingerprint`、`target_unit="DU"`，使前端缓存键与比较入口可区分身份。

## 6. 文件职责与任务划分

| 动作 | 文件 | 职责 |
| --- | --- | --- |
| 新建 | `services/earth_training_contract.py` | Earth 训练/预测 profile、严格参数校验、通道规范、内部 spec 构造与校验 |
| 修改 | `services/dataset_registry.py`、`schemas/datasets.py` | 中性 snapshot、Earth 训练绑定、profile、能力开关 |
| 修改 | `services/dataset_identity.py` | 纯识别函数（Earth ID / Earth 任务），不引入 torch 或数据库依赖 |
| 修改 | `services/earth_dataset.py` | 快照窗口、通道选择、可重用归一化、`from_release` |
| 新建 | `training_backbones/earth_daily_model.py` | DLinear 统一构造与 Earth forward |
| 新建 | `services/earth_training_artifact.py` | 原子保存、结构验证、严格重载、DU 指标聚合 |
| 新建 | `models/training_scripts/earth_daily.py` | 训练循环与 CLI 入口 |
| 新建 | `services/earth_prediction_service.py` | 历史预测上下文与执行、指标、错误分类 |
| 修改 | `services/training_service.py`、`services/training_channels.py` | Earth 分派、子进程 spec、完成校验、停止竞态 |
| 修改 | `routers/training.py`、`schemas/training.py` | 错误透传、任务能力、action 保护 |
| 修改 | `routers/predict.py`、`schemas/predict.py` | Earth 预测接口与响应模型；Mars 路径拦截 |
| 修改 | `services/inference_service.py` | Mars 推理、比较、PFI 的 Earth 拒绝分支 |
| 修改 | `services/training_weight_service.py` | 上传 Earth checkpoint 的明确不兼容报告 |
| 新建 | `frontend/src/pages/ModelTrainingPage/earthTrainingConfig.js` | Earth 配置、通道、提交构造纯函数 |
| 新建 | `frontend/src/pages/ModelTrainingPage/EarthTrainingDatasetPanel.jsx` | 日期划分、单位、固定窗口与样本数说明 |
| 新建 | `frontend/src/pages/PredictPage/earthPredictModel.js` | Earth 预测模式、日期起点、缓存键与错误文案纯函数 |
| 新建 | `frontend/src/pages/PredictPage/EarthPredictPanel.jsx` | 日期起点选择、三天场展示、DU 指标 |
| 修改 | `frontend/src/pages/ModelTrainingPage.jsx`、`ExperimentConfigWorkspace.jsx`、`experimentCenterModel.js`、`trainingParamSanitizers.js` | 数据集选项、Earth 配置、Mars 表单快照与恢复 |
| 修改 | `frontend/src/pages/PredictPage.jsx`、`predictModelModes.js`、`trainedModelSelection.js`、`predictRequestCoordinator.js` | Earth 模式接线、任务筛选、场景切换清理 |
| 修改 | `frontend/src/services/api.js` | `datasetId` 传递、Earth 预测接口、结构化错误解析 |
| 修改 | `frontend/src/i18n/zh.js`、`frontend/src/i18n/en.js` | 新增标签、错误与单位说明 |
| 新建 | `docs/earth-training.md` | 实现后的对外契约、启动与验证说明（含预测章节） |

无需新增数据库表：身份、hyperparameters、metrics 与 output_model_path 已有。新增 profile 为 API 可选字段；保留旧客户端所需字段。

### Task 1：锁定输入契约与注册表绑定

**文件：** 新建 `services/earth_training_contract.py`、`tests/test_earth_training_contract.py`；修改 `services/dataset_registry.py`、`services/dataset_identity.py`、`schemas/datasets.py`、`tests/test_dataset_registry.py`、`tests/test_training_dataset_identity.py`。

- [ ] 先写 `normalize_earth_training_hyperparameters(hypers)` 的测试：固定窗口、范围、空辅助通道、规范顺序、未知/重复通道、禁止类型强转；失败统一抛 `DatasetRequestError`。
- [ ] 建立 `require_earth_training_configuration(model_source, uploaded_model_id, hypers)`：先检测 SPHERE/架构/迁移/上传不支持，再做参数模型验证。模型来源在通用“非法值回退 official”之前校验。
- [ ] 定义 `EARTH_TRAINING_PROFILE`，以深拷贝放入 descriptor；增加 `DatasetTrainingProfile` schema，Mars 描述符 profile=null。
- [ ] 抽出 `get_earth_snapshot()`，保留总览 wrapper 及其错误契约；Earth binding 从同次 release 创建，status=verified。
- [ ] 保留 Mars runner、上传 runner 的 `require_training_dataset()` 拒绝 Earth 行为；更新原来“所有 API Earth 请求必定 409”的测试，区分新的 Earth 服务路径与未开放的旧 runner。
- [ ] 验证绑定快照不会被调用者修改污染 registry；包缺失/变化时不返回上次可用身份。

核心断言示例：

```python
import pytest
from services.dataset_identity import DatasetRequestError
from services.earth_training_contract import normalize_earth_training_hyperparameters

def test_channel_order_and_ozone_only():
    one = normalize_earth_training_hyperparameters({"selected_channels": []})
    assert one["selected_channels"] == []
    assert (one["window"], one["horizon"]) == (7, 3)
    two = normalize_earth_training_hyperparameters({"selected_channels": ["SWGDN", "U10M"]})
    assert two["selected_channels"] == ["U10M", "SWGDN"]

@pytest.mark.parametrize("bad", [
    {"window": 3}, {"window": True}, {"epochs": 1.5},
    {"selected_channels": ["U"]}, {"selected_channels": ["U10M", "U10M"]},
    {"learning_rate": float("nan")}, {"_data_source": "personal"},
])
def test_invalid_parameters_are_rejected(bad):
    with pytest.raises(DatasetRequestError) as error:
        normalize_earth_training_hyperparameters(bad)
    assert error.value.status_code == 422
```

### Task 2：实现快照窗口与训练集归一化

**文件：** 修改 `services/earth_dataset.py`；新建 `tests/test_earth_training_data.py`；复用 `tests/conftest.py` 的 `earth_release`、`earth_spatial_release`。

- [ ] 为 `from_release()` 增加 357/172/175 数量与首尾输入/目标日期断言；断言每个样本最后输入日与第一目标日相差一天。
- [ ] 增加有空间变化的 fixture 测试，区分纬经交换、通道错位与南北翻转；不能只用空间常数场验证网格。
- [ ] 实现规范通道选择、纯训练日统计拟合与 normalization 注入；validation/test 缺省统计量时明确失败。
- [ ] 保留独立 path 构造行为，内部共用窗口/统计逻辑；不破坏已有可变 window/horizon 的读取器单元测试（固定 7/3 由网页训练契约承担）。
- [ ] 用“修改 validation/test 数值后训练统计完全不变”的测试证明无泄漏；另测常量通道 scale=1、非有限值失败。
- [ ] 对 ndarray 与 Tensor 测试 DU round-trip，统计数组顺序与选中输入严格对应。

```python
import numpy as np
from services.dataset_registry import DatasetRegistry
from services.earth_dataset import EarthOzoneWindows

def test_full_release_windows_and_physical_target(earth_spatial_release):
    registry = DatasetRegistry(**earth_spatial_release)
    release = registry.get_earth_snapshot("earth_merra2_daily_v2")
    train = EarthOzoneWindows.from_release(
        release, split="train", window=7, horizon=3,
        selected_channels=["SWGDN", "U10M"], normalization=None,
    )
    test = EarthOzoneWindows.from_release(
        release, split="test", window=7, horizon=3,
        selected_channels=["U10M", "SWGDN"], normalization=train.normalization,
    )
    assert len(train) == 357 and len(test) == 175
    assert train.input_channels == ["TO3", "U10M", "SWGDN"]
    x, y = test[0]
    assert x.shape == (7, 3, 36, 72)
    assert y.shape == (3, 1, 36, 72)
    start = int(np.flatnonzero(release.dates == np.datetime64("2021-07-08"))[0])
    np.testing.assert_allclose(
        test.denormalize_ozone(y[:, 0]), release.fields["TO3"][start:start + 3],
        rtol=1e-6, atol=1e-4,
    )
```

### Task 3：封装 DLinear 并验证 36×72

**文件：** 新建 `training_backbones/earth_daily_model.py`、`tests/test_earth_dlinear.py`。只在发现必要接口问题时修改 `model_zoo.py`。

- [ ] 实现 `create_earth_forecaster(input_channel_order, linear_hidden_layers=2)` 与 `earth_forward(model, inputs)`；先验证通道顺序属于已定义规范且 TO3 为首项。
- [ ] 测试 C=1…5 的前向、反向与一次 Adam 更新；输出 shape 与 y 完全一致，全部值有限。
- [ ] 测试不同 batch size，`use_sphere` 必定 false；传真实日期无需先变成 Ls。
- [ ] 对有 projector 的 C=1…4 检查参数进入 optimizer、state_dict 与重载，不只测五通道 Identity 情况。

```python
import torch
from training_backbones.model_zoo import build_forecaster

def create_earth_forecaster(input_channel_order, linear_hidden_layers=2):
    return build_forecaster(
        architecture="dlinear", input_channels=len(input_channel_order),
        selected_channels=list(input_channel_order[1:]), hidden_dims=[64, 64, 64],
        height=36, width=72, window=7, horizon=3, use_sphere=False,
        architecture_params={"linear_hidden_layers": linear_hidden_layers},
    )

def earth_forward(model, inputs):
    time_placeholder = inputs.new_zeros((inputs.shape[0], inputs.shape[1]))
    output = model(inputs, time_placeholder)
    expected = (inputs.shape[0], 3, 1, 36, 72)
    if tuple(output.shape) != expected:
        raise ValueError(f"Unexpected Earth output shape: {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise ValueError("Non-finite Earth model output")
    return output
```

零占位仅兼容旧签名，不保存成 Earth 季节特征。运行：`Invoke-EarthTrainingTests tests/test_earth_dlinear.py`。

### Task 4：先做 checkpoint 与 DU 指标契约

**文件：** 新建 `services/earth_training_artifact.py`、`tests/test_earth_training_artifact.py`。

- [ ] 定义第 5.4 节各字段的 schema 校验与 `load_earth_training_artifact()`；以不完整非空文件、错误 dataset/task/channel/grid、损坏 state_dict 为失败样例。
- [ ] 实现 `compute_earth_metrics(predictions_du, targets_du)`，参数 `[N,3,1,H,W]`，输出 `{overall,by_lead}`；shape 不符/非有限值报错。
- [ ] 对生产训练支持逐 batch 平方误差、绝对误差与元素计数累加，避免为指标保留所有预测；最终结果与数组版一致。
- [ ] 实现原子保存：临时文件位于最终路径同目录，严格重载与输出比对通过后才替换；只操作本任务明确路径，不扫描清理目录。
- [ ] 测 CPU round-trip：`torch.load(weights_only=True)` 成功、严格重载后同输入的标准化输出一致、DU 输出一致；零辅助通道与全部通道均覆盖。
- [ ] 测重载使用保存的 normalization，即使当前默认参数改变也不重新拟合或改变通道顺序。

```python
import numpy as np
import pytest
from services.earth_training_artifact import compute_earth_metrics

def test_metrics_are_physical_and_split_by_lead():
    reference = np.full((2, 3, 1, 36, 72), 250.0)
    prediction = reference + np.array([1.0, 2.0, 3.0])[None, :, None, None, None]
    result = compute_earth_metrics(prediction, reference)
    assert result["overall"]["mae"] == pytest.approx(2.0)
    assert result["overall"]["rmse"] == pytest.approx(np.sqrt(14.0 / 3.0))
    assert [row["lead_day"] for row in result["by_lead"]] == [1, 2, 3]
    assert [row["rmse"] for row in result["by_lead"]] == pytest.approx([1, 2, 3])
```

### Task 5：完成独立 Earth runner

**文件：** 新建 `models/training_scripts/earth_daily.py`、`tests/test_earth_training_runner.py`。

- [ ] 入口按 backend 根目录配置导入路径；只解析 `--output_path`，从 `ASTRAATMOS_EARTH_TRAINING_SPEC` 读取内部 spec，缺少或非法则非零退出。
- [ ] 可测试核心为 `run_training(spec, output_path, registry, device=None)`，CLI 使用真实 registry；测试可传 fixture registry。
- [ ] 验证 spec 与实际 snapshot 身份后构建三份窗口，创建 DLinear、Adam、MSELoss；限制随机数与 DataLoader 行为。
- [ ] 实现训练、验证、按 validation 选 best、patience、恢复 best、最终 validation/test DU 评估。最优权重须 `.detach().cpu().clone()`，不引用活模型参数。
- [ ] 日志沿用现有进度格式，普通十进制输出并 flush；epoch 完成后保留 99.x% 的进行中状态，只有产物校验成功才到 100%。
- [ ] 输出内部日志说明日期、通道、样本量、device、best epoch；UI 仅显示理解训练所需信息。
- [ ] 全部 loss/梯度/参数与指标做有限值检查；失败抛清晰错误并不保存完整标记产物。
- [ ] 按第 5.4 节写临时 artifact，再 CPU 重载抽取一个 test batch 验证一致性，通过后发布最终文件；成功打印完成说明并退出 0。

日志必须兼容：

```text
Epoch 1/10 Batch 1/45 Loss=0.123456
Epoch 1/10 Loss=0.110000 Val Loss=0.120000
Earth best epoch: 1
Earth metrics unit: DU
```

45 是 357 个训练窗口、batch_size=8 时的 batch 数；不得硬编码，按 loader 长度输出。不得用测试 loss 填充 Val Loss。最佳权重测试用受控验证序列 `[0.8, 0.4, 0.6]`：best_epoch=2，保存 epoch 2 参数；patience=1 时第三轮触发早停，仍评估最佳参数。另测末 batch 不足时 loss 元素加权、test 不参与优化/早停。

### Task 6：接入共享任务服务与完成状态

**文件：** 修改 `services/training_service.py`、`services/training_channels.py`、`routers/training.py`、`schemas/training.py`；新建 `tests/test_earth_training_service.py`、`tests/test_earth_training_routes.py`。

- [ ] 创建任务之前解析 dataset_id、配置支持性、参数、包可用性与身份；标签/名称沿用现有事务检查。错误路径断言数据库无新增记录、Popen 未调用、上传模型加载器未调用。
- [ ] Earth 使用服务端脚本 `earth_daily.py`；`normalize_training_hyperparameters()` 先按 dataset 分派 Earth 纯校验，再处理旧 Mars，防止清空地球通道。
- [ ] 任务 flush 后创建内部 spec，在 `_run_training_subprocess()` 新增可选 `earth_training_spec=None` 参数；其他调用方保持默认值，修正测试 subclass/stub 签名。
- [ ] Earth 的 args 仅脚本与 output_path；spec 用 env 注入，身份字段不落入普通超参数 CLI。实际 output_path 与 task id 路径一致。
- [ ] 退出码 0 后用 `asyncio.to_thread()` 严格验证 artifact 与绑定/规范参数/task id，读取 metrics 后原子更新任务终态；失败写清晰 error_code。
- [ ] 增加停止/完成竞态回归：停止标记不可被末轮日志、待处理 progress 或完成回调覆盖；保留现有 failed + “Stopped by user” 约定。
- [ ] Earth 错误优先于 ValueError 捕获，返回结构化 detail；schema 增加 computed `trained_prediction_supported`（Earth=true）。
- [ ] 接通后启用 registry Earth `training` capability；`trained_prediction` 在 Task 10 预测路由就绪后再置 true。

测试至少覆盖：1 epoch 成功、非零进程退出、退出 0 但无文件、非空垃圾文件、有效文件身份不符、stop 后退出 0、tag 权限拒绝、名称冲突、身份字段伪造。数据库用临时 SQLite，日志与产物都放 pytest tmp_path，不污染真实用户记录。

### Task 7：封住跨场景推理与迁移入口

**文件：** 修改 `services/inference_service.py`、`routers/predict.py`、`routers/training.py`、`services/training_service.py`、`services/training_weight_service.py`；新建 `tests/test_earth_training_isolation.py`。

- [ ] 增加纯任务识别函数 `is_earth_training_task(task)`：优先读 task.dataset_id，再读合法的 legacy `hyperparameters.training_dataset`；明确 Earth 时禁止回退 Mars，损坏/未知身份按既有未知身份策略处理。
- [ ] 在 `_prepare_task_prediction_context()` 完成用户访问检查后、准备数据/缓存前拒绝 Earth。
- [ ] 在 `get_test_results()` 直接路径增加同一保护；action=test 路由在 `prepare_task_inference_data_env()` 之前拒绝。
- [ ] 搜索所有以 task id 调用的推理/metrics/compare/error-distribution/PFI 分支，保证覆盖；compare 先对全部任务完成访问与场景预检，再读任何缓存或开始计算。混入 Earth 时整个请求拒绝，不先算完 Mars 再发现 Earth，也不静默丢掉它只比较其余模型。
- [ ] `routers/predict.py` 先捕获 `DatasetRequestError` 再捕获 `ValueError`，使拒绝返回 409 与稳定 code，不变成 400 或 500。
- [ ] Mars 迁移读取来源 task 后、加载文件前拒绝 Earth 来源；`TrainingWeightService._validate_weight_file()` 识别 `artifact_schema=astraatmos_earth_forecast_checkpoint_v1` 时返回 ok=false 与明确不兼容 errors，上传记录为 invalid。不抽取 state_dict 伪装 Mars；保留原先支持的 Mars 裸权重路径。
- [ ] 不创建 Earth 预测缓存；测试把 Mars loader、cache lookup、weight loader 改为“一调用就失败”，证明保护发生在这些动作之前。

```python
from services.dataset_identity import DatasetRequestError, is_earth_training_task

def require_mars_prediction_task(task):
    if is_earth_training_task(task):
        raise DatasetRequestError(
            "dataset_prediction_not_supported",
            "Earth tasks use the Earth historical prediction endpoint",
            status_code=409,
        )
```

该 helper 可定义在 `services/inference_service.py` 供路由复用；`is_earth_training_task()` 定义在轻量 identity 模块，不导入 Torch/数据库。历史缺字段 Mars task 保持旧行为，不因 `dataset_version=null` 拒绝全部旧模型。

### Task 8：训练页数据集与 Earth 配置

**文件：** 新建 `earthTrainingConfig.js`、`EarthTrainingDatasetPanel.jsx` 及相邻 `.test.js`；修改 `ModelTrainingPage.jsx`、`ExperimentConfigWorkspace.jsx`、`experimentCenterModel.js`、`trainingParamSanitizers.js`、`frontend/src/services/api.js` 与 API 测试。

- [ ] 复用 `frontend/src/services/datasets.js` 的 `fetchDatasets({signal})`，读取返回对象的 `.items`；不在 `api.js` 建第二个目录客户端，避免训练页硬编码包日期与 fingerprint。
- [ ] 训练数据集列表基于 registry，把 `earth_merra2_daily_v2` 作为第三个可选项与两个 Mars 选项并列列出（沿用当前“选项行直接列出”的既有布局，不改成原生下拉）；默认仍为 Mars。
- [ ] Earth 缺失/invalid 时选项仍可见，附具体不可用原因，按钮禁用并支持重试。Mars `availability=unverified` 不能被新逻辑一律禁用。
- [ ] 进入 Earth 时保存当前 Mars 表单快照，切 official/DLinear、7/3、SPHERE=false、transfer=false，清除有效提交中的 `uploaded_model_id` 与迁移源；回 Mars 恢复 Mars 配置，避免 Earth 通道进入 Mars 请求。
- [ ] Earth 面板显示 TO3 必选、四个辅助变量与单位，真实日期三分区与 357/172/175 样本数，全球 36×72 与 5° 分辨率说明；文案明确“过去 7 天”“未来 3 天”。
- [ ] 提供 epochs/batch/lr/seed/patience/linear_hidden_layers，范围对应后端；隐藏不适用的循环网络隐藏维度，Earth DLinear 不展示 Mars SPHERE 或上传/迁移表单。
- [ ] 构造 Earth 请求走独立纯函数 `buildEarthTrainingHyperparameters(form)`；只发送白名单字段，规范通道顺序；未知 dataset 不得被 sanitizer 静默改成 OpenMARS。
- [ ] `startTrainingTask()` options 增加 `datasetId` 并序列化 dataset_id，保留 legacy key 同步；解析 detail.code/message，兼容字符串 detail 与非 JSON 响应。
- [ ] 元数据请求加入取消/序号保护；账号退出、场景切换与较慢响应不能把 Earth 配置覆盖到 Mars。提交时校验当前表单场景，不依赖尚未生效的 React setState。
- [ ] 中英文、深浅主题、窄屏布局跟随已有样式，不引入新组件框架。

```javascript
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildEarthTrainingHyperparameters } from './earthTrainingConfig.js';

test('Earth payload fixes dates and preserves ozone-only selection', () => {
  const payload = buildEarthTrainingHyperparameters({ selectedChannels: [] });
  assert.equal(payload.training_dataset, 'earth_merra2_daily_v2');
  assert.equal(payload.model_architecture, 'dlinear');
  assert.equal(payload.window, 7);
  assert.equal(payload.horizon, 3);
  assert.equal(payload.use_sphere, false);
  assert.deepEqual(payload.selected_channels, []);
  assert.equal(Object.hasOwn(payload, 'dataset_fingerprint'), false);
  assert.equal(Object.hasOwn(payload, 'transfer_source_task_id'), false);
});
```

运行（frontend 目录）：

```powershell
node --test src/pages/ModelTrainingPage/earthTrainingConfig.test.js
node --test src/pages/ModelTrainingPage/trainingParamSanitizers.test.js src/services/api.trainingTags.test.js
```

新增 API Earth 测试文件命名为 `src/services/api.earthTraining.test.js`，断言顶层 dataset_id、legacy 同步、409/422 message 展示与旧调用兼容。结构测试只能辅助定位，不代替真实点击验收。

### Task 9：训练历史、指标与「用于预测」入口

**文件：** 修改 `ModelTrainingPage.jsx`、`ExperimentResultPanel.jsx`、`trainingHyperparameterFormatting.js`、`TrainingHistory.jsx` 及对应测试。

- [ ] 历史卡片增加 Earth/Mars、数据集显示名、version、7 天→3 天；详情显示规范输入通道、各单位、日期划分、best epoch 与 DU 指标。`dataset_id` 与 `model_source` 分别显示，Earth 不是“自定义模型”。
- [ ] metrics 按 schema 分支解析；Earth 展示 validation/test 的总体/逐日 RMSE/MAE；旧 Mars flat metrics 仍正常显示。训练/验证 Loss 曲线标注为标准化 MSE，不标 DU。
- [ ] Earth「用于预测」按钮在任务 completed 且 `model_available` 时可用，写入现有 `TRAINING_TASK_HANDOFF_KEY` 并携带 Earth 场景标记跳转预测页；Earth 的「去模型比较」明确禁用并说明混合比较尚未开放。
- [ ] Earth 正常使用日志、重命名、标签与现有单条管理；执行验收时不批量删除任务/文件。
- [ ] 标签/搜索筛选对 Earth 任务同样生效；切换筛选不改变后台运行任务或当前日志订阅。
- [ ] 新建 `src/pages/ModelTrainingPage/earthTrainingHistory.test.js`，覆盖不同场景 metrics、禁用动作与旧历史 JSON 兼容。

### Task 10：预测页 Earth 历史预测

**文件：** 新建 `frontend/src/pages/PredictPage/earthPredictModel.js`、`EarthPredictPanel.jsx` 及 `.test.js`；修改 `PredictPage.jsx`、`predictModelModes.js`、`trainedModelSelection.js`、`predictRequestCoordinator.js`、`stores/predictCache.js`、`frontend/src/services/api.js`、i18n。

- [ ] 新增 Earth 预测模式（与 `trained` / `trained_compare` 并列的第三模式），从训练页 handoff 或模型选择进入；Mars 两种模式保持原行为。
- [ ] 服务端返回的可选日期范围驱动起点选择（日期输入或可选列表），不硬编码训练/测试划分；越界时禁用提交并显示服务端 code。
- [ ] 展示三天逐日日期、预测场 / 参考场 / 残差场（DU）与总体+逐日 RMSE/MAE；沿用现有场展示与指标表组件习惯，不新建图表框架。
- [ ] 切换任务/星球时清理过期结果（清空场、指标与错误），避免显示上一任务数据；缓存键必须包含 `planet`、`dataset_id`、`dataset_version`、`dataset_fingerprint`、`training_task_id` 与起点日期。
- [ ] Earth 模式不显示 MY/Ls 控件，单位固定 DU；Mars 继续使用 MY/Ls 与原有单位。
- [ ] 结构测试覆盖：模式定义、日期范围到起点的映射、缓存键身份区分、越界与错误码文案、切换任务清理。

```powershell
node --test src/pages/PredictPage/earthPredictModel.test.js
node --test src/pages/PredictPage/trainedModelSelection.test.js
npm run build
```

### Task 11：真实数据联调与文档交付

**文件：** 新建 `docs/earth-training.md`；同步 `README.md`、`docs/earth-compact-dataset.md`、`docs/dataset-registry.md`、`docs/earth-overview.md`；记录本计划实际验收状态。

- [ ] 执行第 7 节回归，逐条记录实际命令与结果；先修复失败再继续。
- [ ] 启动应用并按 AGENTS.md 验证 `GET /health`、前端首页与代理 `/api/datasets` 均 200。
- [ ] 浏览器进入 `#/training`，选择 `earth_merra2_daily_v2`，确认 registry 日期/单位/样本数（357/172/175）显示正确。
- [ ] 通过真实网页启动一次训练（建议 epochs=1、batch_size=8、全部辅助变量），观察 pending→running→completed、日志与 loss 曲线；记录 task id、参数、best epoch、窗口数与 metrics，禁止虚报精度。
- [ ] 检查任务 DB 五列与 checkpoint 一致；CPU 加载真实产物，对一个 test batch 重载前后比对：标准化输出 `rtol=1e-5, atol=1e-6`，DU 输出 `rtol=1e-5, atol=1e-4`。
- [ ] 从该任务进入预测页，选择历史预测起点，核对返回三天日期、36×72 网格、非空预测场与有限指标；记录实际预测起点与指标。另测越界日期返回 422、指纹变化与损坏权重返回 409。
- [ ] 刷新页面确认训练记录保留；测试重命名与已有/新建标签，不改变 checkpoint 身份。另启动一次任务验证停止，保留日志说明结果。
- [ ] 切换回 Mars，检查旧表单、历史与预测选择不受 Earth 污染；检查 Earth 不出现在 Mars 预测/比较/迁移选项，直接请求对应接口返回规定的 409。
- [ ] 从已完成的地球总览查看某天 TO3 后进入训练，确认使用相同数据 ID 与版本；训练后再读同日同点 TO3，确认总览仍为原始 DU，未被训练归一化污染。
- [ ] 补测单位隔离：固定 dataset fingerprint、日期与经纬网格点，等待加载完成并断言点位值为有限数值；改变全局火星臭氧/温度单位后重新选择同日同点，确认单位仍为 DU 且数值一致。占位 `--`、加载中或不同网格点不算通过。
- [ ] 同步文档并明确区分已开放（Earth 官方 DLinear 7→3 训练、历史起点预测、DU 指标）与仍未开放（Earth/Mars 混合比较、未来外推、地球其他模型、迁移、SPHERE、Earth 预测持久化缓存）。
- [ ] 检查 diff、文档链接与新增文件追踪状态；交付改动文件、测试、实际 task id、指标、预测日期与剩余限制，不自动提交/推送。

## 7. 验证命令与验收矩阵

先确认解释器：`$env:CONDA_PREFIX\python.exe`（含 torch、netCDF4）。不要默认系统 Python 就是训练环境。

后端工作目录：`AstraAtmos_backend/backend/`。每次调用使用新的纯英文临时路径，不复用已存在的 basetemp，也不执行批量清理；返回非零时立即报告失败。

```powershell
function Invoke-EarthTrainingTests {
    param([Parameter(Mandatory = $true)][string[]]$TestFiles)
    $earthTrainingTemp = Join-Path $env:TEMP ('.earth-training-test-' + [guid]::NewGuid().ToString('N'))
    & "$env:CONDA_PREFIX\python.exe" -m pytest @TestFiles -q --basetemp "$earthTrainingTemp"
    if ($LASTEXITCODE -ne 0) { throw "Earth training verification failed: $TestFiles" }
}

Invoke-EarthTrainingTests tests/test_earth_training_contract.py
Invoke-EarthTrainingTests tests/test_earth_training_data.py
Invoke-EarthTrainingTests tests/test_earth_dlinear.py
Invoke-EarthTrainingTests tests/test_earth_training_artifact.py
Invoke-EarthTrainingTests tests/test_earth_training_runner.py
Invoke-EarthTrainingTests tests/test_earth_training_service.py
Invoke-EarthTrainingTests tests/test_earth_training_routes.py
Invoke-EarthTrainingTests tests/test_earth_prediction_service.py
Invoke-EarthTrainingTests tests/test_earth_prediction_routes.py
Invoke-EarthTrainingTests tests/test_earth_training_isolation.py
Invoke-EarthTrainingTests tests/test_dataset_registry.py
Invoke-EarthTrainingTests tests/test_dataset_routes.py
Invoke-EarthTrainingTests tests/test_earth_dataset.py
Invoke-EarthTrainingTests tests/test_earth_overview_service.py
Invoke-EarthTrainingTests tests/test_earth_overview_routes.py
Invoke-EarthTrainingTests tests/test_training_dataset_identity.py
Invoke-EarthTrainingTests tests/test_training_channel_contract.py
Invoke-EarthTrainingTests tests/test_official_training_runner.py
Invoke-EarthTrainingTests tests/test_uploaded_training_contract.py
Invoke-EarthTrainingTests tests/test_trained_model_predict_contract.py
Invoke-EarthTrainingTests tests/test_training_model_artifacts.py
Invoke-EarthTrainingTests tests/test_training_tags.py
Invoke-EarthTrainingTests tests/test_transfer_learning_strategy.py
Invoke-EarthTrainingTests tests/test_inference_legacy_official_weights.py
Invoke-EarthTrainingTests tests/test_prediction_analysis_cache_identity.py
Invoke-EarthTrainingTests tests/test_prediction_horizon_contract.py
```

项目部分测试替换 `sys.modules`，以上默认逐文件独立进程，避免模块污染；不删除失败断言。总览两个文件必须回归，确认取数 wrapper、不可变原始数组与 capability 没被本阶段破坏。缺外部 MCD 数据的既有失败要单独说明缺失路径，不伪造数据、不删除测试。

前端工作目录：`frontend/`：

```powershell
node --test src/pages/ModelTrainingPage/earthTrainingConfig.test.js src/services/api.earthTraining.test.js
node --test src/pages/ModelTrainingPage/earthTrainingHistory.test.js
node --test src/pages/PredictPage/earthPredictModel.test.js
node --test src/pages/PredictPage/trainedModelSelection.test.js src/pages/PredictPage/predictModelModes.js
node --test src/pages/ModelTrainingPage/trainingParamSanitizers.test.js src/pages/ModelTrainingPage/experimentConfigStructure.test.js
node --test src/services/datasets.test.js
npm run build
```

根目录：`git diff --check`。新文档可能尚未跟踪，额外检查新文件尾空格、链接与代码块。

| 验收维度 | 必须观察到的结果 |
| --- | --- |
| 数据 | 731 日、36×72、规范纬经数组、TO3 DU；无 MY/Ls |
| 通道 | 只臭氧、单辅助、多辅助、全部辅助均可建模；canonical 顺序可恢复 |
| 划分 | 357/172/175，窗口不跨边界；未来辅助场未输入模型 |
| 归一化 | 训练日拟合；修改 validation/test 不改变统计；恢复只读保存统计 |
| 模型 | C=1…5 在 36×72 前反向可行；adapter/projector 权重一起保存 |
| 选择与评估 | best 由 validation 决定；test 仅最终使用；DU 总体与逐日指标有限 |
| 产物 | CPU `weights_only` 严格恢复；错误身份/shape/state/nonfinite 拒绝；非空垃圾不算成功 |
| 任务 | 真实任务可创建、跟踪、停止、刷新后保留；停止竞态不变成成功 |
| 预测 | 历史起点返回三天日期、36×72 真实网格、非空预测/参考/残差（DU）与有限指标 |
| 权限与错误 | 身份由服务端生成；越界 422；指纹变化/损坏权重 409；无客户端路径或哈希绕过 |
| 场景隔离 | Earth 不能通过 UI/API/旧 handoff/Mars transfer/compare 使用 Mars 推理与缓存 |
| 回归 | Mars 数据读取、训练、预测与上传模型仍按原契约工作；总览未被改坏 |

## 8. 后续阶段留下的稳定接口

后续阶段直接读取 `astraatmos_earth_forecast_checkpoint_v1`，通过保存的 `model_config` / `input_channel_order` / `normalization` / grid / date splits 重建模型。未来比较身份至少包含 `planet`、`dataset_id`、`version`、`fingerprint`、`target`、`horizon`、测试起点集合与指标 aggregation；不同模型可使用不同辅助输入，但比较的参考目标与日期集合必须一致。持久性基线与无参考真值外推也必须复用同样窗口、目标日期与单位。

若未来添加模型、季节特征、日期划分或更换归一化，需提升 profile/implementation/artifact 版本中实际改变的契约，不覆盖 v1 语义，也不能修改固定 ID 对应的数据内容来伪装同一版本。

## 9. 执行对话可直接使用的任务说明

```text
请按 docs/plans/2026-09-23-earth-dlinear-training.md（2026-09-27 修订版）实施 Earth MERRA-2 v2 的官方 DLinear 网页训练与历史预测。

先读 AGENTS.md 与 README 接手章节，检查 Git 状态并保留既有未提交修改。以当前 earth_merra2_daily_v2 的真实数据契约与源码为准：731 个 UTC 日、全球 36×72、五变量、357/172/175 窗口、归一化只拟合训练期。v1 身份保留只读兼容，不映射到 v2，也不把 Earth 回退到火星。

完成训练页 Earth 数据集选择、固定过去 7 天预测未来 3 天、TO3 必选与四个可选辅助输入；复用任务进度、日志、停止、历史与标签；使用独立 Earth runner 与现有 DLinear 工厂；保存可严格恢复的单文件 checkpoint（数据版本/指纹、完整模型配置与权重、通道顺序、真实坐标、日期划分、归一化参数、DU 指标），并在标记完成前用 CPU 严格重载校验。

随后接通预测页 Earth 模式：从已完成 Earth 任务与其 checkpoint 恢复模型，按用户选择的历史预测起点读取此前 7 天输入，返回随后 3 天的预测场、参考场与残差场（DU）及误差指标，并返回可选日期范围、三天日期与真实经纬网格。日期越界、指纹变化、权重损坏、任务无权限返回明确错误，且绝不回退到火星预测。前端沿用现有预测页模型选择与场展示习惯，Earth 用日期与 DU，Mars 继续 MY/Ls 与原有单位；切换任务时清理过期结果；缓存键区分星球与数据集身份。

补充后端与前端测试（数据切窗与边界、训练集归一化、checkpoint 重载、历史日期预测、权限与错误、Mars 回归、前端模式切换），用真实 v2 小包完成至少一次训练与预测端到端验收，并核对预测场非空、日期与网格正确、指标有限。后端只用 $env:CONDA_PREFIX\python.exe；前端源码变更后在 frontend 执行 npm run build。需要启动应用时按 AGENTS.md 验证 /health、前端首页与代理 /api/datasets 均 200。

同任务更新 README 与专题文档，准确区分已开放与未开放能力；报告修改文件、实际执行的测试与联调结果、训练任务 ID、预测日期与剩余限制。不要自动提交、推送、创建额外对话或批量删除文件。
```

编制验证范围：源码与协议核对、计划内容与链接检查，以及 DLinear C=1…5 在 36×72 上的前向/反向/参数构建实测。计划中的训练、测试与网页验收均需实施对话实际执行，不能作为已通过的证据。
