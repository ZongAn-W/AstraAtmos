# Earth 三小时上传模型 v1

需要一次输入完整全球图时使用独立的[全图 v2 说明与模板](earth-3hourly-fullgrid-model.md)。v1 仍固定 24×48；v2 固定 240×480、batch 1–64（默认 1，可运行大小取决于模型与显存），训练/预测/诊断按各自冻结契约运行，不把旧权重自动升级。

本模板仅适用于 `earth_merra2_3hourly_v1`，独立于日频 `earth_merra2` 声明及 Mars 模板。下载 [Python 模板](earth-3hourly-uploaded-model-template.py)，导出 `MODEL_SPEC` 和 `build_model(config)`。三小时声明的 schema 为 `astraatmos_earth_3hourly_uploaded_model_v1`，声明字段必须完整、类型和值必须与模板一致；不接受旧 Earth 标签自动升级。需要共享地形门控、季节坐标和 ConvLSTM/SimVP 结构时，可下载[四位置共享门控模板](earth-3hourly-four-positions-shared-no-after-decoder-model.py)；该模板仍使用 v1 的 24×48 空间块。

## 数据与调用

官方与上传模型共用反归一化 TO3 的六项评价指标及 v3 保存/完成门禁：MSE（DU²）、RMSE/MAE（DU）、R²（无量纲）、MAPE/SMAPE（%）。新训练总体、每步和累计时段按发布 `lat_bounds` 做球面 cell-area 加权；旧三小时 v1/v2 checkpoint 保持只读兼容、缺项不补写。公式、单位和验证入口见[地球评价指标](earth-evaluation-metrics.md)。

服务端 registry 提供 UTC 三小时数据，任务配置输入/输出窗口（默认 56→24）和 240×480 全球网格。目标始终为 TO3（DU），辅助变量可选 U10M/V10M（`m s-1`）、T2M（K）、SWGDN（`W m-2`）。`input_channels/input_units/auxiliary_inputs` 声明的是支持的通道全集，训练请求可选任意辅助变量子集。

**全球网格不是单次模型调用的空间尺寸。** 训练和预测沿用服务端 24×48 不重叠空间分块，共 100 块，再拼成全球输出。块间没有 halo、坐标、经纬度或额外位置输入；需要跨块空间上下文的模型不适用本版本。模型可以声明**每窗口一个标量的季节坐标**（见下文 `derived_inputs`）：它是纯时间的 `[B,window]` 张量，同一窗口的所有块拿到相同数值，因此不引入跨块空间上下文，也不改变分块契约。

训练内部缓存按空间块连续存储，缓存物理布局不改变这里的 BTCHW float32 调用及完整空间覆盖。CUDA 加载使用页锁定内存和非阻塞传输，梯度有限性检查仍覆盖全部参数并保留异常拦截。布局、批量读取的身份复核和旧缓存兼容规则见[三小时训练缓存](earth-merra2-3hourly.md)。

| 项目 | 实际契约 |
| --- | --- |
| 调用 | 按声明固定为 `model(x)`、`model(x, ls)`、`model(x, topography)` 或 `model(x, ls, topography)`。返回单个 `torch.Tensor` |
| 输入 | `[B,window,C,24,48]`，BTCHW；`window` 为任务配置的 1–240 个三小时时间步，C 为 1 到 5 |
| 通道轴 | 轴 2；顺序为 TO3，然后按 U10M、V10M、T2M、SWGDN 顺序保留所选变量 |
| 输出 | `[B,horizon,1,24,48]`，轴 2 仅 TO3；`horizon` 为任务配置的 1–240 个三小时时间步 |
| 数值 | 输入和输出均为有限 float32；是按训练集统计量标准化后的数值，输出按保存的 TO3 统计量还原为 DU |
| 模型初始化 | dry-run、实际训练构建和 checkpoint 重建共用 CPU float32 初始化，转换模型浮点参数/缓冲区；runner 再迁移到所选 CPU/CUDA 设备 |
| 设备 | 输出必须与输入同设备，模型不得自行固定设备 |
| eval 批次 | 同一样本的预测不能依赖 batch 大小、其他样本、排列顺序或重复调用状态，eval 不能修改参数或缓冲区；训练模式可使用正常 BatchNorm，eval 使用其固定运行统计量 |
| 训练 | 必须有可训练参数；前向和 backward 必须成功并产生有限梯度 |
| 缺失 | 选中输入及 TO3 目标存在缺失时拒绝窗口/任务，不填零、不插值 |
| split | 新任务按完整原始 UTC 时间轴和请求比例连续划分；训练、验证、测试窗口及上传历史回测均不能跨任务分区；旧任务沿用 manifest |
| normalization | 仅当前任务 train 区间拟合人口均值/标准差，ddof=0、epsilon=1e-6；预测严格使用 checkpoint 保存值 |

`build_model(config)` 必须返回 `torch.nn.Module`。服务器传入：`dataset_id`、`contract_schema`、任务 `window/horizon`（默认 56/24）、`height=24`、`width=48`、`global_grid_shape=[240,480]`、`spatial_tile_shape=[24,48]`、`in_channels`、`selected_channels`（含 TO3）、`target_channel='TO3'`，以及校验后的自定义参数。参数 schema 沿用通用 int/float/bool/select 形式，但不得覆盖以上字段、dataset identity、路径、设备或 normalization。

禁止在 spec 或参数中指定数据目录、dataset version、fingerprint、snapshot。数据绑定由服务器创建；上传代码只处理给定张量。

## 季节坐标 `derived_inputs`

模型可以在 `MODEL_SPEC` 顶层声明服务器提供的季节坐标，以及独立于 MERRA-2 发布包的 ETOPO 静态地形：

```python
MODEL_SPEC = {
    "name": "...", "parameters": {...},
    "datasets": {"earth_merra2_3hourly_v1": {...}},
    "derived_inputs": {"ls": {"required": True, "shape": ["batch", "window"],
                              "dtype": "float32", "unit": "degree"}},
}
def forward(self, x, ls):   # 声明后前向必须是两个参数
    ...
```

`ls` 是太阳视黄经（0° 春分、90° 北夏至、180° 秋分、270° 北冬至），与火星 `Ls` 同一定义、同一关系和同一单位，由窗口的 UTC 时间戳计算：发布包本身没有这个变量。取值落在 `[0, 360)`，形状恰为 `[batch, window]`（`window` 是输入窗口长度，不含 horizon），不做标准化；窗口内通常只变化约 7°，信息内容是"这个窗口属于哪个季节"，不是日内时刻。

规则：

- 声明必须**逐字段完全相同**，只支持 `ls` 与 `topography`。写错单位（如 `radian`）、形状、dtype、`required`，增加未知字段或资源路径，都会在上传校验时被拒绝。
- 声明与实现必须一致：`forward` 必须按固定顺序显式接收对应参数名；缺参、多参、`*args` 或不同参数名会被拒绝。未声明的旧模型保留原有调用。
- **声明后模型永远不会在没有坐标时被调用。** 训练、历史回测与诊断都由发布轴上同一窗口的时间戳重新计算，因此同一个窗口不同分块拿到相同数值。
- 未声明的模型行为完全不变：前向仍是 `model(x)`，checkpoint 里不出现相关字段，旧权重与旧上传包继续可用。
- **生产者版本被冻结**：声明了 `ls` 的 checkpoint 会同时记录 `season_algorithm`（当前为 `solar_longitude_meeus_low_precision_v1`）。该字段缺失、未知或与契约不一致时拒绝加载——坐标是算出来的，公式变了就不能拿旧模型配新数值。
- Mars 的顶层 `auxiliary_inputs`（从数据文件读 `Ls`、MOLA 地形）在地球仍然被拒绝，错误信息会指向 `derived_inputs`。Earth feed 内同名字段仅表示可选气象通道。地球地形使用 ETOPO，不经过 MOLA 路径，也没有可读的 `Ls` 列。

### 静态地形 `topography`

```python
"derived_inputs": {
    "topography": {"required": True, "shape": ["batch", 1, "height", "width"],
                   "dtype": "float32", "unit": "meter"},
}

def forward(self, x, topography):
    ...
```

同时声明 `ls` 与 `topography` 时，固定调用 `model(x, ls, topography)`，与字典书写顺序无关。`topography` 是有限 float32 `[B,1,H,W]`，与 x 同设备、没有时间轴，不增加气象通道 C、不参与 TO3 等标准化。服务端传递物理米值；模型可以显式自行缩放。模型拿到独立副本，原地修改不会污染静态资产、其他样本或下一次调用。

地形先严格对齐活动发布的完整 240×480 坐标及 cell bounds，再逐样本按真实空间切片取得 24×48；同一批不同位置使用各自地形，shuffle 和最后一个批次保留该对应关系。全图 v2 使用完整场。PFI 只置换气象通道，地形始终对应接收样本的位置，不自动成为气象 PFI 通道。

上传、重新校验、16 种气象通道组合、eval 独立性和 backward 探针均使用已配置资产，并在 v1 探针中覆盖不同空间块；不会以零场代替缺失资产。只有声明地形的模型要求资源存在，旧模型和应用启动不依赖该资源。训练任务与 checkpoint 冻结声明及资产内容身份；严格重载比较当前地形、目标网格及处理算法，资产缺失、替换或身份不完整明确拒绝。旧 checkpoint 不补写地形字段，`ls` 的 `season_algorithm` 继续独立冻结；官方 DLinear 不自动接收地形。

可上传示例：[v1 仅地形模型](earth-3hourly-topography-model-template.py)、[v2 季节坐标与地形模型](earth-3hourly-fullgrid-topography-model-template.py)、[v1 四位置共享门控模型](earth-3hourly-four-positions-shared-no-after-decoder-model.py)。这些示例在前向计算中消费地形；共享门控模板还消费服务器派生的季节坐标。原始无地形模板继续可用。配置 `ASTRAATMOS_EARTH_TOPOGRAPHY_PATH`、来源、离线构建、海洋及冰盖策略、EGM2008 垂直基准、摘要覆盖范围与验证命令见 [Earth 静态地形资产](earth-topography-asset.md)。该能力是模型输入，不代表三维球面已实现真实地形起伏，也不证明预测精度提高。

## 校验和错误

v1 分块上传和手动重新校验优先使用模型声明支持的短窗口 2→1；输入或输出未声明该值时，分别使用对应声明列表的首项（例如仅支持 56→24 的模型仍校验 56→24）。完整 24×48 空间块、模型默认参数、16 种通道组合与全部前向、梯度及 eval 独立性检查保留。报告保存实际探针的 `window`、`horizon` 和 `validation_scope=declared_channels_probe_windows`，通过校验只证明该探针配置；实际训练继续使用任务选择的窗口，不由上传探针覆盖。该策略避免可变窗口的循环/门控模型在全部通道上重复长序列检查而超时。显式 `earth_probe` 仍按传入的窗口、batch 和参数执行。

上传沿用仓库的单文件大小、UTF-8、AST 导入/调用检查和隔离进程超时边界。该检查不是操作系统级不可信代码沙箱，部署时仍需遵循既有任务执行权限边界。

上传和手动重新校验的模型 dry-run 默认从 120 秒通用隔离进程预算开始（全图 v2 在合法声明确认后扩展，见下段），可通过后端 `.env` 的 `USER_MODEL_VALIDATION_TIMEOUT_SECONDS` 设置正整数秒数；修改配置后重启后端。创建任务不重复执行实际配置 dry-run。到期仍强制终止子进程，全部通道、前向、梯度及 eval 批次一致性检查保留。超时报告顶层 `code` 为 `uploaded_model_validation_timeout`，兼容性查询返回 `status=unknown`、`compatible=false` 和同一错误码，页面显示“校验超时，请重试”。上传包在重新校验成功前仍为 `validation_status=invalid`，禁止创建训练任务。旧报告只有 `User model validation timed out after ... seconds` 文本时也识别为超时，无需改写数据库。该预算不影响下文 checkpoint 完成门禁的独立 30 秒重载检查。

全图 v2 上传与重新校验在隔离子进程完成导入、数据声明规范化及参数 schema 检查后，通知父进程使用通用预算与 `USER_MODEL_FULL_GRID_VALIDATION_TIMEOUT_SECONDS`（默认 900 秒）的较大值；总截止时间从进程启动计算，通知不重置计时，不在父进程执行上传源码。导入卡死、非法声明、v1 分块及 Mars 不获得扩展。显式 `earth_probe` 仍检查实际窗口、batch、通道和模型参数，全图使用同一预算；显式指定的验证器超时（含 `None`）保持原义。上传与重新校验在后台线程等待隔离进程，其他 API 可继续响应。任务创建复用报告，不运行这些探针。探针超时表示当前配置尚未完成验证，不是模型已经证实不兼容。隔离进程先读取完整报告再等待退出，避免大报告堵塞 Windows 管道；无结果、异常退出和真正超时继续拒绝，详见[全图 v2](earth-3hourly-fullgrid-model.md)。

上传 dry-run 检查全部 16 种通道组合、B=1/2、eval 前向、train 前向及 backward。在 backward 前后均用固定的不同样本核对单独与合批、交换顺序、更换其他成员和重复调用的输出，覆盖 B=2/3/默认8，采用 float32 容差 `rtol=1e-5, atol=1e-5`。checkpoint 重载保留并校验保存的训练 batch，但运行时独立性探针使用单独的有界预算：v1 的合批大小固定为 B=2/3，全图 v2 为 B=1/2，不随训练 batch 放大；因此只在 B>8 才依赖其他样本的模型可能通过这些探针，不能据此声称所有 batch 均已证明独立。探针拒绝 eval 修改参数或缓冲区，并在结束时恢复原值与各模块模式。依赖 batch 均值、`BatchNorm(track_running_stats=False)`、首次调用校准或 eval 计数缓冲区的模型会被拒绝。成功报告保存 `eval_batch_policy=earth_eval_sample_independent_v1`；旧报告没有该证据时显示 unknown，需要重新校验。创建任务复用该报告，不再次执行实际所选通道、batch 和自定义参数的隔离探针。有限探针不能证明任意源码对全部输入都独立，CPU dry-run 通过也不代表任意任务配置、GPU 资源预算或模型收敛已验收。

兼容性接口 `GET /api/user-models/{id}/earth-compatibility?dataset_id=earth_merra2_3hourly_v1` 返回具体数据集的 `status`、`compatible`、`code` 和 `reasons`：`available` 可用，`unavailable` 明确不兼容，`unknown` 未获得有效执行结论。未知、超时、失败均不能创建训练任务。日频的原接口默认值保留。

| 错误码 | 含义 |
| --- | --- |
| `uploaded_model_not_earth_3hourly_compatible` | 未声明本三小时 feed |
| `uploaded_model_contract_invalid` | spec、保留参数、保存配置或执行契约不合法 |
| `uploaded_model_compatibility_unknown` | 无有效结论或隔离进程异常退出 |
| `uploaded_model_validation_timeout` | 模型校验隔离进程超过配置时间上限；未知结论，可重新校验 |
| `uploaded_model_earth_3hourly_dry_run_failed` | 构建、输出类型/形状/有限性或梯度校验失败 |
| `invalid_earth_training_parameters` | 请求参数未通过 schema 或窗口范围与形状校验 |
| `uploaded_model_missing` / `uploaded_model_tampered` | 源码缺失或摘要不匹配 |
| `uploaded_model_invalid` | 上传包整体校验失败，不能进入训练 |
| `earth_prediction_origin_out_of_range` | 没有完整输入/真值或完整窗口跨 split |

模型工厂返回 double/half 浮点参数时，服务器在上述三处统一转为 float32；这不改变输出检查。返回形状不符、tuple/dict 输出、float64 输出、NaN/Inf、错误设备或无梯度仍拒绝。运行阶段也检查有限 float32 和输出形状。

## 任务与 Checkpoint

创建任务先核对归属、valid 状态、保存报告中具体 dataset/schema 的兼容证据、源码 SHA-256、参数 schema 和合法 batch 范围，再固定包 ID、版本、源码、契约 schema、自定义参数及服务器数据身份并排队；不额外执行模型 dry-run。重启队列后仍使用同一份源码。实际训练继续检查输出类型/形状/有限 float32 和全部梯度有限性。三小时 checkpoint schema 为 `astraatmos_earth_forecast_checkpoint_3hourly_v1`；上传模型 implementation 为 `astraatmos_earth_3hourly_uploaded_runner_v1`，同时保存独立模型契约版本、全球网格、空间块、通道/单位、步长、任务划分策略/请求比例/实际索引与 UTC 边界/步数及窗口数和 normalization；新策略缺失或不一致时拒绝，旧任务不迁移。

重载核对任务与数据集身份、嵌入源码摘要、声明、参数 schema 和 build config，再以 `strict=True` 加载 state dict，并对加载实际权重后的模型重复 eval 批次一致性检查。旧 checkpoint 不改写，但实际模型违反此检查时拒绝发布、诊断或回测。runner 在发布前检查重载与前向一致性；任务完成门禁另调度 30 秒超时的独立 CPU 进程复查权重及前向，超时或失败不能标记 completed。原源码文件不可用时使用摘要匹配的嵌入副本，并在预测 context 报告原文件状态。日频及 Mars checkpoint 不能通过三小时 schema/identity 检查。

实现与验收范围见 [三小时专题](earth-merra2-3hourly.md)。合成数据训练、回测 smoke 仅证明代码链路；完整真实包的只读检查不等同于真实数据训练、模型精度或真实任务历史回测验收。


## 任务级比例

上传模型和官方 DLinear 共用[任务级划分策略](earth-task-splits.md)。请求比例均为有限数值、大于 0 且总和为 1，默认 0.7/0.2/0.1；每区至少 window+horizon 步。数据身份仍绑定原发布，normalization 与缓存按数据、实际 train 区间、通道顺序及方法/布局版本共享；当前任务和 checkpoint 继续严格保存自己的完整划分。模型自定义参数、batch、学习率和轮次不改变全时间轴缓存内容。复用证明、完整性与失效规则见[训练准备缓存](earth-preparation-cache.md)。队列重启复用固定划分；多模型比较要求相同 test UTC 范围及窗口口径。

独立的三小时 **36×72（5°）** 训练发布、完整全球窗口与上传模型契约见 [5° 数据集](earth-merra2-5deg.md)；本文件的原 0.75° 契约继续兼容。
