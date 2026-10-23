# Earth 训练提速

实现入口为 [Earth runner](../AstraAtmos_backend/backend/models/training_scripts/earth_daily.py)、[窗口读取](../AstraAtmos_backend/backend/services/earth_dataset.py)和 [36×72 共享门控优化模板](earth-3hourly-5deg-shared-topography-model-template.py)。优化保持请求的 window/horizon、变量、batch 和网络参数尺寸；不会把 20→20 改成短窗口。CPU 继续使用 FP32，Mars runner 不变。

## Runner

- CUDA 默认 autocast：支持 BF16 时优先 BF16，否则使用 FP16 和动态梯度缩放。损失和面积加权累加使用 FP32，参数与保存权重仍为 FP32。FP16 梯度溢出时跳过该次更新并降低 scale，连续 8 次溢出或整轮没有成功更新会失败，不发布无更新产物。
- CUDA 训练启用 TF32 和 cuDNN 固定 shape 算法调优。逐轮 validation 使用相同 autocast，早停按该验证损失选择最佳权重；最终完整 validation/test 关闭 autocast，恢复运行前的 TF32/矩阵精度，关闭 cuDNN 调优并启用确定性卷积，随后执行严格保存、重载和逐位前向一致性检查。同一份权重的非确定性 CUDA 卷积也可能存在末位差异，因此最终验收固定算法而不放宽比较。正常或异常退出均恢复原 CUDA 设置。
- 每个分区预计算一条只读季节坐标数组，窗口读取复制所需片段，季节算法与时间戳不变。ETOPO 张量按 batch 准备，保留各样本隔离和真实空间切片。
- normalized memmap 在 CUDA 上估算每批输入与目标数据不超过 64 MiB 时，由单个后台线程预取下一批。只提前一批，样本顺序、源与缓存身份检查、读取错误传播均保留。使用线程避免 Windows 多进程复制受保护映射。原 240×480 的大批次保持同步加载，固定页内存及非阻塞 GPU 传输沿用既有设置。

后端 `.env` 可配置以下变量，新的训练子进程生效：

| 变量 | 默认 | 含义 |
| --- | --- | --- |
| `ASTRAATMOS_EARTH_AMP` | `1` | `0` / `false` / `no` / `off` 关闭 autocast，TF32 和 cuDNN 调优仍启用 |
| `ASTRAATMOS_EARTH_PREFETCH` | `1` | 同样的关闭值禁用批次预取 |

checkpoint 的 `run.acceleration` 记录 AMP dtype、TF32 和 cuDNN 设置，`run.amp_skipped_optimizer_steps` 记录溢出跳步数，`run.saved_metrics_precision=float32` 和 `run.saved_metrics_cudnn_deterministic` 标明最终评价精度与卷积设置。外部上传契约仍要求 float32 输入输出；仅 CUDA autocast 调用允许模型内部返回 FP16/BF16，再在契约边界转为 FP32，CPU/常规调用仍拒绝错误 dtype。NaN/Inf、损失、梯度与契约检查保留。

AMP、TF32、算法调优和浮点加法顺序可能改变训练数值轨迹。网络结构相同不等于逐位相同或长期精度相同；关闭 AMP 也不代表关闭 TF32。最终指标精度与训练/早停精度分别记录，不能以单步吞吐证明收敛。

## 共享地形门控模板

优化模板保留原模型的全部层、参数名、shape、参数默认值和共享门控位置；地形编码、季节编码及同空间尺寸的地形仿射映射仅在一次前向内复用，梯度不 detach，不跨 batch 缓存计算图。

当 CUDA 上 `batch × window >= 256` 时，内部按至多 `max(1, 128 // window)` 个样本分段，训练用 PyTorch 非重入 activation checkpoint 重算中间结果，并保留 Dropout 重算的随机状态；eval 同样分段但不重算。该网络使用每样本 GroupNorm，没有跨样本 BatchNorm，因此可分段执行。输出拼接后仍对原 batch 计算损失并只执行一次优化器更新。单样本时间轴和完整全球空间网格不拆分；长窗口或很宽隐藏层仍可能超出显存。

在上传面板将优化 `.py` 作为新文件上传，再用它创建新实验。已经上传的包与旧 checkpoint 固定原源码，只能获得 runner 优化，不能自动获得模板内部缓存与分段。现有训练不支持断点续训；修改源码可能触发开发服务器 `--reload`，长期训练应在不带 `--reload` 的后端运行。

若要把 T11 的门控/ConvLSTM 贡献作为消融项，可使用 [纯 SimVP T11 对照模板](earth-3hourly-5deg-t11-pure-simvp-model.py)。它只保留 T11 的空间编码器、时间 Inception 模块、时间投影和解码器，直接输入五个观测通道，不接收 `ls` 或地形；四项 SimVP 参数和 T11 相同。第一层从 T11 的 16 个 ConvLSTM 特征输入改为五个观测输入，因此该层参数量必然不同，后续 SimVP 层可逐字对齐。对照训练应固定 T11 的数据集、通道、20→20、batch、epochs、seed、学习率、早停和划分。

要单独去掉 ConvLSTM 并保留季节/地形门控，使用 [T11 门控 SimVP 对照模板](earth-3hourly-5deg-t11-no-convlstm-model.py)。它保留条件编码复用、样本分段、激活重算和默认两时间块下的五次共享门控；移除递归层后，原 ConvLSTM 前后的门控连续执行，第一层空间编码改为直接处理观测通道。默认五通道 20→20 为 580,258 个参数，结构、上传参数与比较边界见 [5° 模型说明](earth-merra2-5deg.md#训练与上传模型)。

## 实测与验证

2026-10-17，本机 RTX 4060 Laptop 8 GB、PyTorch 2.6.0+cu124，同五通道、36×72、20→20、原隐藏层默认值与 Adam。每种配置执行 1 次预热和 3 次计时，测量 GPU 同步后的前向、损失、反向和参数更新；输入为合成数据，不包含发布验证、加载、完整 validation/test 或 checkpoint 写盘。

| 配置 | batch | 平均单步 | PyTorch 峰值 allocated |
| --- | ---: | ---: | ---: |
| 原模板 FP32 | 4 | 129.5 ms | 1804 MiB |
| 编码复用 + BF16/TF32 | 4 | 75.2 ms | 1337 MiB |
| 编码复用 + BF16/TF32 + 内部分段/重算 | 32 | 783.3 ms | 2029 MiB |

batch 4 的同配置比较约 1.72 倍吞吐，allocated 减少约 25.9%。batch 32 只测量最终优化配置，不能据此给出相对原版的加速倍数。allocated 不包含全部驱动/桌面显存，以上不是整轮耗时或任意显卡的保证。基准原始报告保存在工作区 `test-runs/earth_acceleration_benchmark_20261017.json` 与 `earth_acceleration_batch32_20261017.json`，不随 Git 分发。

回归入口为 `tests/test_earth_training_acceleration.py`：覆盖季节值等价、地形样本隔离、预取顺序/异常/内存门限、输出 dtype、原模板与编码复用的 FP32 输出/梯度、CUDA 重算与 Dropout 随机状态、FP16 溢出恢复，以及 CUDA 短训练后的最终 FP32 评价和严格重载。隔离上传校验另对实际交付文件执行；吞吐基准本身不验收完整 epoch 或长期收敛。

同日实际任务 90 使用该优化模板在完整两年 5° 发布上完成 20 轮 GPU 训练，仍为五通道、20→20、batch 32，最佳 epoch 15。训练与完整测试评价已结束，但父进程最终 CPU 重载超过旧 30 秒预算而报失败；后续完整复核、备份和恢复记录见 [重载超时恢复](earth-metrics-task-recovery.md#重载超时恢复)。新的全图重载预算覆盖 36×72 契约，原保存 batch、strict state-dict、eval 独立性及输出逐位一致门禁保留。这一次训练及其保存指标不等于独立科学精度或长期稳定性验收。
