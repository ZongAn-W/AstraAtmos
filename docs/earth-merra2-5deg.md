# Earth 三小时 5° 训练数据集

`earth_merra2_3hourly_5deg_v1` 是独立的全球 **36×72、5°** 发布。它由已验证的 `earth_merra2_3hourly_v1`（240×480、0.75°）逐时间点做球面单元面积重叠加权得到。时间不聚合：2020–2021 的 5848 个三小时时间点、01:30 至 22:30 UTC 中心标签、时间边界、五变量单位与缺失掩码全部保留。任一有正重叠面积的源单元缺失会使目标单元缺失，不填零、不对残余有效面积重新归一化。

5° 格点中心为纬度 −87.5…87.5、经度 −177.5…177.5，单元边界覆盖全球。240/36 不是整数，构建器计算实际边界重叠；纬度权重为 `sin(north)-sin(south)`，经度权重为重叠角宽。该数据适合降低空间计算量，但不能保留原有小尺度结构；格点数减少约 44.44 倍不代表模型训练必然等比例加速。

## 构建与配置

工作目录为 `AstraAtmos_backend/backend`，使用项目指定的 AstraAtmos Conda 解释器。用外部目录变量指定源包与新包：

```powershell
# $sourceRelease / $lowresRelease 均为 Git checkout 外的完整绝对路径。
& "$env:CONDA_PREFIX\python.exe" scripts/build_earth_merra2_5deg.py `
  --source-dir $sourceRelease --output-dir $lowresRelease `
  --verification-cache $verificationCache
```

构建器先校验源发布，保留源发布 SHA-256、指纹、原始 SLV/RAD 清单和时间规则；新 manifest 具有独立 ID、数据文件 SHA-256 与发布指纹。全部变量按小块扫描。输出不覆盖已有目录；已有完整包只有与当前源发布指纹和网格一致且校验通过时才复用，损坏/中断产物保留并拒绝复用，需使用新目录。

后端 `.env` 设置 `ASTRAATMOS_EARTH_MERRA2_3HOURLY_5DEG_DIR` 为新发布目录，默认位置为 `data/earth/merra2_3hourly_5deg_v1/`。该目录须包含 `manifest.json` 与 `earth_merra2_3hourly.nc`。缺包显示不可用，不代用原高分辨率包或日频包。默认数据总览仍使用原三小时 0.75° 发布；5° 条目仅开放训练、历史回测、模型比较和测试诊断，`capabilities.web_overview=false`。

## 训练与上传模型

训练页选择「地球 MERRA-2 · 3 小时 · 36 × 72（5°）」。官方 DLinear 和明确声明低分辨率契约的上传模型均使用完整全球窗口；不切成 24×48 块。新任务按原始 UTC 时间轴连续划分（默认 70/20/10），仅训练分区拟合归一化；窗口与目标不得跨分区，窗口/步长范围仍为 1–240，默认 56→24。目标为 TO3（DU），辅助通道依次为 U10M、V10M、T2M、SWGDN。

上传示例：[5° SimVP 模板](earth-3hourly-5deg-model-template.py)。独立声明：

- 数据集 key：`earth_merra2_3hourly_5deg_v1`
- schema：`astraatmos_earth_3hourly_5deg_model_v1`
- `grid=[[36,72]]`、`spatial_tile_shape=[36,72]`
- 输入 `[B,window,C,36,72]`，输出 `[B,horizon,1,36,72]`，float32、同设备
- batch 1–64；上传探针默认 batch 1、优先 2→1，检查全部 16 种通道组合、梯度和 eval 样本独立性

使用共享地形门控网络时，可上传 [36×72 共享门控优化模板](earth-3hourly-5deg-shared-topography-model-template.py)。它保持原网络的层、参数名/尺寸及参数默认值，在一次前向内复用地形与季节编码，并通过样本分段和激活重算降低大 CUDA batch 显存；优化不缩短时间窗口，不改一次参数更新的 batch。旧上传包固定源码，不自动获得模板内部改动。runner 的混合精度、预取及验证边界见 [Earth 训练提速](earth-training-performance.md)。

做纯 SimVP 对照时，使用 [T11 纯 SimVP 对照模板](earth-3hourly-5deg-t11-pure-simvp-model.py)，桌面交付文件名为 `09_pure_simvp_earth_36x72_T11_control.py`。该文件从 T11 保留四个 SimVP 核心模块的实现和参数：SpatialEncoder 的 3×3 stride=2 / 3×3 编码、TemporalTranslator 的 1×1 历史折叠投影、2 个 TemporalInceptionBlock 和 SpatialDecoder 的反卷积；删除 ConvLSTM、共享门控、Ls 与地形声明。模型直接接收 `[B,window,5,36,72]`，输出 `[B,horizon,1,36,72]`，不需要地形资产。为匹配 T11，只调整上传模型选择和自定义参数中的四项 SimVP 参数；不要把 T11 的 `convlstm_hidden_dim`、`gate_hidden_dim` 或 `initial_gate_strength` 带入纯 SimVP 配置。两者的第一层输入通道不同，其余 SimVP 核心层逐字一致。

仅去掉 ConvLSTM、保留门控时，使用 [T11 门控 SimVP 对照模板](earth-3hourly-5deg-t11-no-convlstm-model.py)，桌面文件名为 `10_gated_simvp_no_convlstm_earth_36x72_T11_control.py`。其基线为 task90 冻结的 T11 源码，SHA-256 记录在文件头。季节/地形编码、共享门控、四个 SimVP 核心类及 SharedGateTranslator 定义均保留；删除 ConvLSTMCell、ConvLSTMEncoder 和 `convlstm_hidden_dim` 参数，空间编码器直接接收观测通道，不新增 5→16 适配层。

默认两个时间块仍有五次共享门控调用：输入连续两次 → 空间编码后 → 两个时间块之间 → 空间解码前。连续两次分别对应原 ConvLSTM 前后的位置，保留原调用次数；它们均作用于当前五通道观测，而非原来的五通道/16通道组合。前三次使用逐历史帧季节编码，后两次使用历史季节编码均值。解码后直接输出，无额外门控。若增加时间块，仍在每两个时间块之间调用一次门控。共享条件编码复用、CUDA 批次分段和激活重算沿用 T11。

该对照的默认参数为 `spatial_hidden_dim=64`、`temporal_hidden_dim=128`、`num_temporal_blocks=2`、`dropout=0.1`、`gate_hidden_dim=32`、`initial_gate_strength=1.0`；仍要求 `ls` 和匹配的 5° ETOPO 地形。五通道 20→20 时总参数为 580,258，较 T11 少 12,160 个 ConvLSTM 参数和 6,336 个第一层卷积参数。比较时固定五通道、20→20、batch32、20 epochs、seed11、学习率 0.001、早停5和70/15/15划分，仅切换新上传模型及上述六项自定义参数。第一层输入尺寸和初始化顺序随递归层删除而改变，因此相同 seed 不保证保留层初始权重逐位一致；此对照保留门控调用规则，仍需多 seed 实验评估训练效果。新文件必须作为独立模型上传，旧 T11 checkpoint 不能严格加载到该模型。

其余 feed 字段与原三小时契约一致。每个上传文件只声明一个三小时空间 feed。旧 24×48/v2 240×480 模型需要修改声明与实现，再上传或校验为新模型；已有高分辨率任务/产物不迁移，跨分辨率模型不能混合比较。

可选 `derived_inputs.ls` 仍按真实窗口 UTC 时间计算。`topography` 使用独立的 `earth_etopo2022_surface_5deg.nc`，按新发布的实际坐标与 cell bounds 从 ETOPO/GSHHG 原始源重新构建；运行时严格校验资产/网格身份，张量为 `[B,1,36,72]`，单位米，不随时间复制为数据通道。默认资产在 `data/assets/`，可通过 `ASTRAATMOS_EARTH_TOPOGRAPHY_5DEG_PATH` 指定。构建方式见 [地形专题](earth-topography-asset.md)，传 `--target-release $lowresRelease/earth_merra2_3hourly.nc --output data/assets/earth_etopo2022_surface_5deg.nc`。不声明地形的模型不依赖该资产。

checkpoint 冻结新发布身份、36×72 网格、窗口、通道、task split 与归一化；完整测试集指标、回测和 PFI 使用 5° 单元球面面积权重。缓存按数据指纹和布局隔离，高分辨率缓存不能复用到本数据集。

## 验证范围

`tests/test_earth_5deg_dataset.py` 覆盖非整数比例重网格、独立单元权重对照、面积均值守恒、缺失传播、发布与缓存证明复用，以及官方/上传地形模型的真实优化、严格 checkpoint 重载、历史预测和诊断。前端 `earth5Degree.test.js` 验证数据集选择、窗口/通道请求和空间契约隔离。生成完整两年发布与短训练验证不代表长期收敛或科学预测精度。

T11 去 ConvLSTM 对照的回归入口为 `tests/test_earth_t11_no_convlstm.py`，覆盖保留类定义与默认参数、五个门控位置、与绕过递归层且调整第一层输入的原 T11 前向/梯度对照、季节/地形依赖、单样本独立性、state-dict 严格重建，以及 20→20、batch32 的 CUDA 混合精度/激活重算和一次参数更新。它不证明真实数据长期收敛或精度收益。

2026-10-17，该对照通过上述 5 项回归，含本机 CUDA 的 batch32、20→20 合成参数更新；生产隔离上传验证器使用实际 5° ETOPO 资产，以默认 2→1 窗口通过全部 16 种通道组合的输出、梯度与 eval 独立性检查。本次仅交付模板并执行上述验证，未注册上传包、创建训练任务或执行真实数据收敛实验。

2026-10-16 本机已构建完整两年发布（NetCDF 215,388,392 字节，dataset fingerprint `4f87873269107381dd982b858070c31f8a2f440736bf769bd5fdb124f7e25fc2`），五变量缺失率均为 0；另独立抽查 3 个时间点 × 9 格 × 5 变量共 135 个值及全球面积均值，最大单格绝对存储误差为 2.81×10⁻⁵，最大面积均值误差为 2.43×10⁻⁷。真实两年五通道 DLinear 使用 7→1、batch 32、1 epoch CPU 完成训练、严格重载、末时刻历史回测与抽样诊断/PFI；其作用是验证链路，不作为精度结论。相关后端两组回归分别通过 236 与 178 项，新增队列恢复与下载模板隔离校验用例另通过；前端相关 72 项及生产构建通过。旧日频 `test_earth_training_artifact.py` 的 40 项在当前分支基线仍因 `dataset_retired` 失败，本次没有重新开放日频入口。
