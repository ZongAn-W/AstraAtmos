# 环境变量预测

平台以一次实验的“完整历史输入 → 一个未来目标”为任务定义。地球和火星均可选择目标变量；目标不必包含在历史输入中。例如地球可使用 TO3、U10M、V10M、SWGDN 预测 T2M，火星可使用 O3、U、V、D、S 预测 T。输入和参考目标保持现有时间、空间对齐与分区规则，不把未来输入或未来目标送入模型。

## 可用变量与数据来源

| 行星 | 标识 | 含义 | 原生输出单位 |
| --- | --- | --- | --- |
| 地球 | TO3 | 总柱臭氧 | DU |
| 地球 | T2M | 2 m 温度 | K |
| 地球 | U10M / V10M | 10 m 纬向 / 经向风 | m s-1 |
| 地球 | SWGDN | 地表向下短波辐射 | W m-2 |
| 火星 | O3 | 臭氧 | μm-atm |
| 火星 | T | 数据集温度场 | K |
| 火星 | U / V | 纬向 / 经向风 | m/s |
| 火星 | D | 沙尘光学厚度 | 1 |
| 火星 | S | 太阳下行辐射通量 | W/m² |

地球使用两个已注册的 MERRA-2 三小时发布（原生 0.75° 与独立 5° 发布），变量依照发布包与掩码读取。火星 `mcd_overview` 使用服务器 MCD；`openmars_mcd` 中臭氧来自 OpenMARS，其他变量来自对齐至 OpenMARS Ls 的 MCD，不能将它们描述为新增 OpenMARS 观测。这里的“可选变量”仅指现有服务器数据集，上传总览文件仍不会自动成为训练源。

## 在实验中心使用

1. 选择行星与服务器数据集。
2. 选择一个预测目标和至少一个输入变量。输入可包含目标历史值，也可仅包含其他环境变量。
3. 选择兼容模型，配置输入 / 输出窗口和数据划分，开始实验。
4. 完成后使用“用于预测”。预测页从训练任务读取目标与单位，不能临时把臭氧模型改成温度模型。

同一目标的不同输入组合可以作为不同实验比较；不同目标的误差值具有不同物理意义，服务端拒绝混合排名。温度输出与误差使用 K，风、辐射及沙尘保留各自原生单位，只有火星 O3 可以沿用臭氧单位设置换算成 DU。

## 请求与持久化契约

`POST /api/training/start` 在 `hyperparameters` 中支持：

```json
{
  "training_dataset": "earth_merra2_3hourly_5deg_v1",
  "model_architecture": "dlinear",
  "target_variable": "T2M",
  "input_variables": ["TO3", "U10M", "V10M", "SWGDN"],
  "window": 56,
  "horizon": 24
}
```

上例是 `hyperparameters` 片段；完整请求还需实验名称、模型来源等既有字段。火星将标识替换为 `target_variable="T"`、`input_variables=["O3","U","V","D","S"]`，并使用火星数据集和兼容模型。

- `target_variable` 是唯一输出变量，省略时地球为 TO3、火星为 O3。
- `input_variables` 是完整输入列表，至少一项，校验合法性并规范为数据集通道顺序，不自动插入目标。未知变量、重复变量和空输入在训练前拒绝。
- 旧 `selected_channels` 仍表示臭氧以外的辅助输入；未传 `input_variables` 时保留“臭氧 + 辅助输入”的旧行为。旧 Mars 任务若通道字段缺失或无效，任务描述符会沿用推理端规则从训练脚本后缀恢复通道；新客户端应提交完整列表，不根据 `selected_channels` 猜测输入张量。
- 任务、checkpoint 和缓存固定目标、输入顺序、归一化、数据身份与窗口；目标统计与输入统计独立，预测使用训练保存的统计反归一化。
- 预测、指标、误差分布、PFI、模型测试及比较响应携带 `target_variable`、`target_unit`、`input_variables`；前端按任务身份校验三者后才接受结果。跨目标比较拒绝，比较项的输入顺序也必须与各自任务一致；非臭氧指标与科研导出保留目标原生单位，不应用臭氧换算。
- Mars `/api/predict/run` 的顶层变量元数据与 `model_info` 保持一致，包括命中缓存的结果。路由必须保留服务返回的三个字段；回归验证需经过实际 HTTP 响应组装与序列化，单独构造响应 Schema 不能覆盖路由丢字段的问题。

旧臭氧 checkpoint 保持臭氧语义，不能仅修改名称或超参数将其视为其他目标。任务迁移要求源、目标实验具有相同变量身份；裸权重没有目标身份，非臭氧任务需要自带匹配元数据的 checkpoint。

## 模型能力与验证边界

官方模型和上传模型仍遵守各自输入 / 输出形状契约，输出通道数为 1。Earth 上传模型必须具备对应数据集、变量组合的兼容声明与验证证据；旧 Earth 臭氧上传契约不能自动视为支持温度或独立输入。具体声明见[三小时上传模型](earth-3hourly-uploaded-model.md)。Mars 沿用基本前向契约校验，`valid` 不证明任意输入与目标组合均能训练；实际运行仍检查输出形状、梯度与完成产物，上传代码需能读取并正确处理真实任务配置。

Earth 新上传模型可在既有 `MODEL_SPEC.datasets[dataset_id]` 声明中增加：

```python
"supported_targets": ["TO3", "T2M"],
"supports_independent_inputs": True,
"input_variable_sets": [
    ["TO3", "U10M", "V10M", "SWGDN"],
    ["U10M", "V10M", "SWGDN"],
],
```

这表示验证两种目标与两种完整输入组合，共四种配置。每个输入集合按 `TO3,U10M,V10M,T2M,SWGDN` 的原始通道顺序排列。省略新声明则仍只允许 TO3 目标和含 TO3 的输入；开启独立输入而未列集合时枚举全部非空输入子集。为控制上传验证资源，一次声明最多 32 种“目标 × 输入集合”配置，超出时需用 `input_variable_sets` 缩小范围。重新校验后的报告在 `earth_datasets[dataset_id].forecast_configurations` 中保存成功验证的配置；训练请求必须命中其中一项，不能仅凭输出形状相同通过。具有上述新声明的 Earth 模型在 `build_model(config)` 中收到真实 `target_variable`、完整 `input_variables` 与 `out_channels=1`，同时保留原有 `target_channel` / `selected_channels` 键。旧声明维持原始构造配置以兼容历史 checkpoint；模型代码仍需支持声明的配置。

本功能提供任务配置、训练与回测基础设施，不保证任意输入组合包含足够预测信息。长期训练效果、真实数据泛化精度、无参考真值的未来外推、同时输出多个目标和用户自行注册新变量，均不在本次功能范围。

代码入口：后端 [Mars 变量规则](../AstraAtmos_backend/backend/services/mars_variables.py)、[Earth 训练契约](../AstraAtmos_backend/backend/services/earth_training_contract.py)；前端实验配置和预测展示共用变量元数据。训练、预测与诊断的既有数据身份、时间分区和账号权限仍然适用。

## 验证入口

- [Earth 变量链路测试](../AstraAtmos_backend/backend/tests/test_earth_variable_forecasts.py)：官方与上传模型的独立输入训练、严格重载、参考场、Kelvin 诊断、上传证明和队列恢复；包含未选 TO3 缺测时的目标统计隔离。
- [Mars checkpoint 测试](../AstraAtmos_backend/backend/tests/test_mars_checkpoint.py)：官方与上传模型的温度训练 / 重载 / 预测、不同目标的权重迁移门禁、数据来源与缓存隔离。
- [任务身份测试](../AstraAtmos_backend/backend/tests/test_environment_training_identity.py)、[任务列表 HTTP 序列化测试](../AstraAtmos_backend/backend/tests/test_training_variable_serialization_routes.py)、[预测 HTTP 契约测试](../AstraAtmos_backend/backend/tests/test_trained_model_predict_contract.py)及[科研导出测试](../AstraAtmos_backend/backend/tests/test_research_export.py)：旧任务兼容、目标与完整输入往返、响应模型序列化、跨目标比较、原生单位与臭氧转换隔离。
- 前端变量规则位于 [environmentVariables.js](../frontend/src/utils/environmentVariables.js)，模型测试、地球诊断、单模型响应与比较身份的 Node 测试覆盖变量身份校验和单位展示。浏览器已验证两行星目标与输入可独立选择、草稿保留和窄屏布局；这些检查不构成模型预测精度验收。
