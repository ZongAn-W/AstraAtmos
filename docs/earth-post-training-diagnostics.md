# 地球训练后诊断

Earth 三小时官方 DLinear 与独立 v1 上传模型共用诊断服务。训练结果页「模型测试」和预测页「地球训练后诊断」使用同一接口及默认参数；Earth 不进入 Mars 的 Ls、地形、数据准备或 PFI。日频任务继续停用。

预测页的地球与火星单模型工作台共用指标与诊断展示组件：`PredictMetrics`、`ErrorDistributionChart`、`PermutationImportanceChart` 和 `PredictionChartCard`。`predictionPresentation.js` 的 Earth/Mars 展示适配分别提供指标定义、评价范围、单位转换、PFI 轴标签与导出能力；通用图表不判断行星。`EarthDiagnosticPanel` 保留独立手动触发、抽样控件和现有身份/请求协调 hook，只负责地球诊断装配及范围说明；训练结果页自动诊断行为继续由 `autoLoad` 显式配置。

地球单模型页面本身不添加行星专属外壳：标题、说明段、上下文折叠块、起点说明行与参考场说明段已移除，指标区只保留当前窗口与完整测试集两个共用卡片，其中完整测试集卡片的分区行由 checkpoint 的 `split_policy`/`split_ranges` 映射成与火星相同的 `split_meta` 形状后渲染。

该面板的文案全部走 `predict.earthDiagnostics.*`（zh/en 同名键），不再在组件内写中英文内联字符串；标题层级使用共用的 `prediction-chart-title` / `prediction-chart-subtitle`，面板外壳使用共用的 `prediction-panel`，卡片内嵌的指标块以 `flattened` 传入 `PredictMetrics` 以避免二次套框。PFI 卡片的标题、评价范围与帮助文本仍由共用的 `PermutationImportanceChart` 渲染，面板不再重复输出一份；单通道不可计算的原因显示在明细表行内，整组不可计算时才输出兜底说明。触发方式保持独立手动：进入页面只读一次 `/diagnostics/context` 核验可用性，不发送计算请求。

完整组件职责、UTC/Ls 起点、全屏地图/球面及桌面/移动端回归见[共用单模型预测工作台](shared-single-model-prediction.md)。

## 入口与权限

- `GET /api/earth/predict/diagnostics/context?training_task_id=123`：复核已完成状态、checkpoint、数据及任务固定配置，供按钮禁用和原因展示。
- `POST /api/earth/predict/diagnostics`：计算或读取已验证的诊断缓存。
- `POST /api/training/tasks/123/action?action=test`：按任务行星分流。Earth 返回相同诊断载荷，Mars 保留原测试行为。

接口沿用账号认证和任务归属规则，管理员可访问其他账号任务。请求排队进入计算槽及返回前再次读取任务；权重或配置变化拒绝发布结果。checkpoint 严格验证 task ID、产物 SHA、window/horizon、实际通道、归一化、模型源码摘要及数据发布身份。上传源码丢失时沿用现有已验证嵌入副本兼容规则，响应返回警告；副本哈希或实际权重不一致则拒绝。客户端不能指定数据路径、指纹、normalization 或真值。

```json
{
  "training_task_id": 123,
  "sample_windows": 4,
  "scatter_points": 5000,
  "histogram_bins": 40,
  "seed": 42,
  "pfi_repeats": 3,
  "include_pfi": true
}
```

窗口 1–8，散点 1–20,000，分箱 5–100，PFI 重复 1–5，种子为 0–2³²−1 整数。默认 4 窗口、5,000 点、40 箱、seed 42、3 次重复；实际窗口不足时返回实际数量。页面可调整窗口数和重复次数。

## 评价范围与旧任务

`full_test_metrics` 只读取有效 checkpoint 已保存的完整 `metrics.splits.test`。六项指标采用[DU 指标契约](earth-evaluation-metrics.md)及 checkpoint 的真实 aggregation/metric policy；新 v3 为球面 cell-area 加权，旧三小时 v1/v2 保持 grid-uniform。旧三小时 v1 只有 RMSE/MAE 时其他项显示“未提供”，不以训练 loss、validation、单次历史预测或诊断抽样补值。契约缺损的 checkpoint 按现有规则拒绝；服务不会伪造完整测试集指标。

`scope` 与 `sample_metrics` 描述本次抽样诊断。从任务的 test 分区生成完整 window+horizon 窗口，按 NumPy `default_rng(seed)` 无放回选择索引并按时间排序。新任务读取已固定的 `task_split`；无该策略的旧三小时任务读取 manifest 发布分区。归一化只复用该任务 checkpoint 的 train 统计量，不重新拟合。所选窗口的全部输出按 checkpoint policy 计算；新 v3 使用发布 `lat_bounds` 的 cell-area 权重，旧 grid-uniform checkpoint 继续逐格等权。重复出现的目标时间/格点按预测起点分别计入，响应中的 `sample_metrics.aggregation` 与 `sample_metrics.metric_policy` 是诊断展示的权威口径。

响应给出实际/可用窗口数、所选索引、每窗输入与目标 UTC 范围、起点范围、网格大小、全球覆盖、经纬度范围、有效及预期评价点数、种子与策略。默认 56→24、4 窗口时，诊断累计 `4×24×240×480 = 11,059,200` 个预测/参考对；这些计数保留在服务端统计中，浏览器只接收有上限的散点与直方图。

## 散点与残差直方图

所有推理输出先反归一化为 TO3 DU。残差固定为 `prediction-reference`：正值表示高估。散点 `reference`/`prediction` 严格配对，坐标单位 DU，附 `y=x` 线；流式遍历按空间块、批次、窗口、lead、纬度、经度排序，在等大小流区间内用固定种子各选一个点，最多返回请求上限。它是进一步的点位抽样；附带指标属于所选窗口全部格点，不是散点子样本，也不是完整 test。

直方图对所选窗口所有 lead/grid 残差做两遍流式处理：第一遍累计物理指标、散点和残差极值/摘要，第二遍按第一遍极值确定的均匀边界累计样本计数。直方图的 `counts` 始终是残差样本数，不是面积加权分布；面积加权 RMSE/MAE 等只通过 `sample_metrics` 表达。恒定残差以值±0.5 DU 分箱。返回边界、计数、均值、总体标准差、最小/最大和有效点数；两遍残差摘要哈希必须一致，计数之和必须等于声明范围。非有限输入、参考、输出或残差、无有效窗口、身份变化均明确报错，不能过滤后声称完整。

## Earth PFI

每个实际模型输入通道逐项置换，包含历史 TO3；未来目标和参考真值不置换。每次以随机循环排列构造无固定点的窗口映射，将某通道的整段时间×空间场替换成另一窗口同通道。映射一次生成，所有空间块和通道复用，避免块间来源不一致。基线与每项置换使用相同窗口、空间格点、全部 lead、同一面积权重矩阵和 DU RMSE 公式；响应中的 baseline/permutation RMSE 与 `sample_metrics.metric_policy` 使用同一口径。

主要重要性为 `permuted_rmse - baseline_rmse`，单位 DU，保留负值。返回基线、每次置换 RMSE/增量、均值、总体标准差（ddof=0）、窗口映射、实际窗口/点数及 seed。少于两个有效窗口时 PFI 不可计算；某通道在所有抽样窗口完全相同时，该项返回不可计算原因，不能将无法置换解释成模型未使用。变化的未使用通道可得到接近零的重要性。PFI 仅代表当前模型在本抽样范围下的敏感性，不代表因果贡献。

## 内存、协调与缓存

每次只读取当前 24×48 空间块中的有限窗口，每通道最多 8 个时间步分块读取，推理 batch 为 2；PFI 的来源窗口在当前块内保留，最多 8 个。不加载完整 test 预测/真值体积、不重新物化全部滑窗、不重新训练。默认 4 窗口、5 通道、3 次 PFI 时，在 100 个空间块上约执行 `100×2×(2+5×3)=3,400` 次批推理调用；首次计算包含直方图第二遍，成本取决于网络与 NetCDF 压缩读取。

上传模型 eval 必须逐样本独立：同样窗口在不同 batch 大小、排列和其他成员下保持相同输出。准入与 checkpoint 实际权重重载都执行固定探针校验，允许浮点容差；依赖 batch 统计量的模型拒绝。由此训练完整评估、本诊断的 B=2 和单起点回测的 B=1 采用相同逐样本预测定义。细节见[上传模型契约](earth-3hourly-uploaded-model.md)。

现有 Mars 分析缓存的计算入口与 Earth 预测/诊断共用进程级推理槽；不同重请求串行执行，相同 Earth 诊断后续请求复用结果。HTTP 请求取消后已开始的工作仍完成并保有计算槽，不能提前放行导致两个 GPU 推理重叠。此槽仅协调当前后端进程的推理，训练子进程与多 worker 不共享它。

Earth 诊断缓存是进程内 LRU（8 条 / 16 MiB），键至少包含 task ID、checkpoint SHA、数据绑定与指纹、实际 test 划分、window/horizon、实际通道、normalization、模型身份/源码摘要、算法版本、空间块/批次策略、抽样参数和 seed。每次命中前仍加载严格 checkpoint 并复核当前发布；进程重启清空，多 worker 不共享。前端取消旧请求并清空旧结果，使用请求 token 与任务/条件身份拒绝过期响应；条件切换和失败均有重新核验/重试入口。

## 科研导出与验证边界

本组散点、残差直方图与 Earth PFI 暂未开放科研图导出。现有 Earth 三联图和多模型 test 曲线仍只消费账号私有的结果快照，导出不重新推理。若后续开放诊断图导出，应消费本次诊断快照，保留抽样范围与 DU PFI 定义；不能套用 Mars 的 ΔR²。

后端回归入口：`tests/test_earth_diagnostics.py`（手算/确定性统计）、`tests/test_earth_diagnostics_routes.py`（权限、分流、参数与协调）、`tests/test_earth_diagnostics_integration.py`（严格产物/数据流）。前端为 `earthDiagnosticsModel.test.js`、`predictionPresentation.test.js` 与 `services/earthDiagnostics.test.js`；展示适配回归覆盖 Earth 六项 DU 指标、Mars 误差显示换算、当前窗口/完整 test/抽样诊断范围、服务端散点配对和直方图计数、ΔRMSE/ΔR² 负值与导出能力。生产构建使用 `npm run build`。从后端目录用项目指定的 工作区约定的 Conda 解释器执行 pytest，并使用仓库外新的英文 `--basetemp`。验证使用合成数据及模型，真实两年训练和真实模型精度验收不在此范围。
