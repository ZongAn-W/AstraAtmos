# 共用单模型预测工作台

地球与火星单模型预测由同一套组件装配。`EarthPredictPanel` 保留地球上下文、逐步指标和诊断入口，场图布局、查看模式、时间步、指标卡片、诊断图表外壳及全屏交互使用火星已有组件的共用实现。Earth 与 Mars 继续调用各自服务，不改变训练划分、checkpoint、指标公式或上传模型契约。

## 组件职责

| 入口或组件 | 共用职责与保留差异 |
| --- | --- |
| [PredictPage.jsx](../frontend/src/pages/PredictPage.jsx) | 行星/单模型/多模型模式、账号作用域、任务选择、请求取消与结果身份；Earth/Mars 请求状态分别保留 |
| [SingleModelWorkbench.jsx](../frontend/src/pages/PredictPage/SingleModelWorkbench.jsx) | 单模型结果装配、三联图/单图状态、时间步和全屏状态；不决定行星 API 或物理单位 |
| [PredictSidebar.jsx](../frontend/src/pages/PredictPage/PredictSidebar.jsx) | 共用模型下拉、标签筛选、参数摘要、起点/步长区域、运行与错误重试；起点控件由 adapter 提供 |
| [PredictDisplay.jsx](../frontend/src/pages/PredictPage/PredictDisplay.jsx) | 参考/预测/残差顺序、三联图与单图切换、时间步、当前步三联图导出与全屏按钮 |
| [PredictMetrics.jsx](../frontend/src/pages/PredictPage/PredictMetrics.jsx) | 按配置显示指标定义、单位、改善方向、缺项、精度和评价范围 |
| [ErrorDistributionChart.jsx](../frontend/src/pages/PredictPage/ErrorDistributionChart.jsx) | 散点、参考/预测分布及残差直方图的共用卡片与主题；只绘制配置提供的图种和数据 |
| [PermutationImportanceChart.jsx](../frontend/src/pages/PredictPage/PermutationImportanceChart.jsx) | 共用 PFI 卡片、零线、正负值、标签与可用误差条；定义/单位/范围由配置提供 |
| [PredictFullscreenHUD.jsx](../frontend/src/pages/PredictPage/PredictFullscreenHUD.jsx) | 共用全屏外壳、统计读数、真实坐标二维展开和纬向剖面、Escape/焦点/页面滚动恢复 |
| [PredictionChartCard.jsx](../frontend/src/pages/PredictPage/PredictionChartCard.jsx)、[PredictStatus.jsx](../frontend/src/pages/PredictPage/PredictStatus.jsx) | 统一诊断图表标题、范围、操作、加载/错误状态与重试外壳 |
| [EarthPredictPanel.jsx](../frontend/src/pages/PredictPage/EarthPredictPanel.jsx) | Earth 上下文与模型身份、结果适配、完整测试集指标配置和诊断装配；不添加任何引言面板，区块顺序与 Mars 一致 |
| [EarthDiagnosticPanel.jsx](../frontend/src/pages/PredictPage/EarthDiagnosticPanel.jsx) | Earth 独立手动诊断、抽样参数、实际窗口范围及 PFI 明细；指标和图表引用共用组件 |

样式由 [singleModelWorkbench.css](../frontend/src/pages/PredictPage/singleModelWorkbench.css)、[predictionDisplay.css](../frontend/src/pages/PredictPage/predictionDisplay.css) 和 [predictionCharts.css](../frontend/src/pages/PredictPage/predictionCharts.css) 提供。桌面模型侧栏与结果并排，窄屏上下排列；三联图在空间不足时纵向显示，时间步区和详细指标表分别滚动，全屏在移动端纵向布局。主题、色带、单位偏好和数值精度来自现有设置；Earth 的目标单位始终固定 DU。

深色预测分析页复用首页 `--bg-gradient` 与既有 `StarField`，侧栏、准备面板、输入和图表保留实色；浅色主题不显示星点，首页鼠标视差与地球交互不延伸到预测页。

无有效场结果且未运行预测时，共用展示区以 `prediction-display--empty` 显示约 140px、12px 圆角的单层引导面板，不渲染三张空结果卡片和空图导出/切换工具。有可选模型时，入口定位并打开侧栏已有模型选择器；没有可选模型时，入口进入现有 `#/training`。加载模型期间入口显示加载状态，禁用的预测按钮旁说明原因并使用 `aria-describedby` 关联，不自动选择模型或发送预测请求。Earth 进入单模型页只读取训练任务目录；预测上下文、checkpoint/数据身份校验和诊断可用性检查延迟到用户点击“开始预测”后，避免页面进入即触发重型准备流程。

无模型、无结果、无指标数据时不展示空指标或诊断占位；指标区域展示但尚未取得数据时显示 `--`。Earth 实际指标响应中的缺失项仍显示“未提供”，避免把缺失值当作零。真实结果恢复参考/预测/残差图表及原有全屏、时间步和导出。

**单模型页面不再有行星专属外壳。** 标题、说明段、`TO3 · DU` 单位胶囊、`模型与数据集上下文` 折叠块、结果上方的起点说明行和参考场说明段都已移除，主操作沿用共用的 `🚀 开始预测`，起点范围提示只保留越界告警。Earth 侧栏也不再提供 Mars 没有的「刷新起点范围」次级按钮；上下文读取失败时由既有 `PredictStatus` 的「重试」触发同一次运行（内部会按需重新加载上下文）。Earth 保留「逐步与累计窗口指标」折叠表与手动诊断面板，前者是 Mars 单模型没有的附加明细。

## 行星适配层

适配边界集中在三个入口，共用展示组件消费配置或回调，不在布局内部逐项判断行星。

- [singleModelAdapters.js](../frontend/src/pages/PredictPage/singleModelAdapters.js)：起点控件描述、请求转换、响应转换与身份核对、时间步标签、真实网格、单位转换和能力声明。
- [PredictionPlanetAdapter.jsx](../frontend/src/pages/PredictPage/PredictionPlanetAdapter.jsx)：装配起点控件和具体场渲染；Earth 使用 `EarthMap2D`，Mars 使用已有 `FieldCanvas` 与明确指定 `planet="mars"` 的 `SphericalFieldCanvas`。
- [predictionPresentation.js](../frontend/src/pages/PredictPage/predictionPresentation.js)：纯展示配置及数据转换，提供指标定义、评价范围、诊断图种、PFI 标签、单位与科研导出能力。原始 API 结果和快照身份不因显示单位改变。

[predictionFieldGrid.js](../frontend/src/pages/PredictPage/predictionFieldGrid.js) 校验场与真实经纬度的行列关系，保留源顺序和非均匀轴，不生成猜测坐标；[predictionDisplayModel.js](../frontend/src/pages/PredictPage/predictionDisplayModel.js) 提供共用色阶、时间步、全屏读数和对话框交互纯函数。Mars 原有网格模块保留兼容入口。

| 契约 | Earth | Mars |
| --- | --- | --- |
| 模型与服务 | 已完成且权重有效的三小时官方 DLinear / 独立契约上传任务；`/api/earth/predict/*` | 既有已训练模型；`/api/predict/*` |
| 起点 | UTC `datetime-local`，输入转明确 UTC ISO 时间，须在服务端可选起点中 | Ls 滑块，保留当前可选范围与实际输出 Ls 标签 |
| 运行窗口 | 请求只带任务与起点，每次完整运行任务 horizon | 请求步长 1–30，且不得超过训练 horizon |
| 展示步 | 提前小时 + 对应 UTC 时间；仅改变查看内容 | 步序 + 服务端实际 Ls；仅改变查看内容 |
| 坐标与单位 | `grid.latitude/longitude` 与场行列一致；TO3 原始 DU，无 Mars 换算 | 各帧真实 `lat/lon`；μm-atm / DU 按现有显示偏好换算 |
| 指标 | MSE（DU²）、RMSE/MAE（DU）、R²（无量纲）、MAPE/SMAPE（%）；旧三小时缺项显示未提供；完整测试集分区行与 Mars 同格式 | RMSE/MAE、SSIM、R²，保留现有公式与聚合信息 |
| 诊断触发 | 独立手动测试分区诊断；默认 4 窗口、5,000 对散点、40 分箱、seed 42、PFI 3 次 | 保留现有预测流程中的测试集误差分布与抽样 PFI |
| PFI | 置换后 RMSE − 基线 RMSE，即 ΔRMSE（DU）；保留负值与不可计算原因 | 基线 R² − 置换后 R²，即 ΔR²；保留负值 |
| 全屏主视图 | 地球二维地图；不加载火星纹理、Ls 标签或火星单位 | 既有火星球面；共用真实坐标读数、二维展开与纬向剖面 |
| 科研导出 | 当前步三联图可用；诊断散点/残差直方图/PFI 不开放 | 当前步三联图、当前步普通散点与抽样 PFI 按已有快照可用 |

参考和预测共用当前步物理色阶，残差以零为中心对称；恒定场和全零残差有明确非退化范围。设置中的色带改变仅作用于物理场，残差保留固定发散色带。地球场值、误差与指标不会因全局 Mars 单位偏好再次缩放。

## 评价范围与导出

页面明确区分三种范围：

| 范围 | 数据来源与页面含义 |
| --- | --- |
| 当前预测窗口 | 选定历史起点的一次完整输出；Earth `result.metrics` 是此窗口六项指标，当前步场/普通散点也属于此窗口 |
| 完整测试集 | Earth 读取有效 checkpoint 保存的 `metrics.splits.test`，并在同一卡片下显示该块的指标口径与分区行；Mars 使用现有完整测试集指标与误差直方图，不用单次窗口或抽样指标替代 |
| 抽样诊断 | Earth 独立 test 窗口诊断覆盖所选窗口全部 lead/grid，散点再按上限取点；Mars PFI 最多 40 个抽样测试窗口，独立于页面 Ls |

Earth 工作台同时提供本次窗口指标和上下文 checkpoint 的完整 test 指标；诊断结果另列完整 test 与诊断样本指标，缺项保持未提供。Mars 在相同条件重复运行时保留已缓存的完整 test 指标，不用回包中的当前窗口指标替代。窗口数、实际 UTC 范围、格点计数和种子沿用服务器证据；某通道不可计算的 PFI 不能当作零。详情见[地球评价指标](earth-evaluation-metrics.md)与[地球训练后诊断](earth-post-training-diagnostics.md)。

科研导出仍使用账号私有的已有 `export_ref`，不重新训练或推理。当前步选择只更改零基 `step`，不改变原始窗口快照身份；引用缺失或页面正在加载时按钮禁用，服务器继续复核引用、权限、任务、模型/数据身份及有效期。Earth 诊断未开放的图种没有可用导出按钮；多模型图的原有导出范围未扩展。文件格式、资源上限和 MATLAB 包见[科研图导出](research-figure-export.md)。

### 完整测试集分区行

两种行星在完整测试集卡片下都显示同一格式的分区行，由共用的 `getSplitLabel` 渲染：

- Mars 直接读取回包 `metrics.split_meta`（`published_manifest_splits` 带 `test_range`，或任务比例）。
- Earth 的 checkpoint 把绑定存在 `metrics.split_policy` / `split_ranges`（或 `split_ratios` / `task_split.ranges`），字段名是 `date_start`/`date_end`；[earthPredictModel.js](../frontend/src/pages/PredictPage/earthPredictModel.js) 的 `buildEarthSplitLabelSource` 在展示层把它整理成同一 `split_meta` 形状：`published_manifest_splits` 显示「发布测试分区」，任务自定义划分（`earth_task_split_v1`）显示「任务自定义测试分区」，并带 `window_count`。

边界或策略读不到时该行不显示，不回退成猜测的比例行，也不伪造窗口数；`getSplitLabel` 仍保留比例与历史兼容分支供其他调用方使用。

## 身份、取消与缓存

Earth/Mars API、数据准备与服务端缓存保持各自业务路径；组件共用不让 Earth 请求进入 Mars 推理。Earth 仍使用独立进程内预测缓存及诊断缓存，Mars 仍使用既有账号作用域内存缓存和服务端分析缓存。

切换行星、任务、起点或账号会取消旧请求、清空旧结果及全屏。Mars 请求协调器校验通道 token、当前条件与账号 scope；Earth 预测回包同时匹配冻结任务、起点、完整 window/horizon、数据版本/指纹、模型身份及真实坐标，诊断使用独立请求 guard。即使底层请求忽略 abort，晚到结果也不能覆盖当前身份。服务器缓存命中前继续检查实际权重、数据及任务契约，权限不依赖浏览器缓存。

## 验证入口与边界

前端从 `frontend/` 执行 `node --test` 和 `npm run build`。适配与交互纯函数入口为 `singleModelAdapters.test.js`、`predictionPresentation.test.js`、`predictionDisplayModel.test.js`，并保留 `earthPredictModel.test.js`、`earthDiagnosticsModel.test.js`、`predictRequestCoordinator.test.js` 和既有账号/缓存隔离回归。

合成浏览器入口为 [browser-check.js](../scripts/audit/shared-prediction/browser-check.js)。该脚本只在隔离浏览器页面中替换 API，模型和数据均标为合成；不创建训练任务或启动真实推理。从仓库根目录、已有独立 Playwright CLI 会话中执行：

```powershell
npx --package @playwright/cli playwright-cli -s=shared-prediction run-code --filename scripts/audit/shared-prediction/browser-check.js
```

页面加载生产构建，通过真实组件检查 1440px 桌面和 390px 移动端、深浅主题、4/6 位精度、色带、三联图/单图、时间步、地球地图/火星球面全屏、PFI 与评价范围、任务/行星切换、过期响应拒绝、错误重试及退出登录清理；还检查支持的三联图导出入口、Earth 完整测试集分区行、Mars 重复运行保留完整测试集指标。脚本中的小网格只验证数据配对和交互，不模拟完整 240×480 长 horizon 的传输、解析与绘图成本。

后端从 `AstraAtmos_backend/backend/` 使用工作区规定的 工作区约定的 Conda 解释器逐文件执行 `-m pytest`，每次指定仓库外新的英文 `--basetemp`；未标记 async 的已训练模型契约加 `--asyncio-mode=auto`。本轮实际结果如下：

| 文件 | 实际结果 |
| --- | --- |
| `test_earth_3hourly_prediction_time.py` | 34 passed |
| `test_earth_3hourly_prediction_service.py` | 55 passed |
| `test_earth_3hourly_prediction_routes.py` | 27 passed |
| `test_earth_diagnostics.py` | 7 passed |
| `test_earth_diagnostics_routes.py` | 13 passed |
| `test_prediction_horizon_contract.py` | 15 passed |
| `test_trained_model_predict_contract.py`（asyncio auto） | 11 passed |
| `test_earth_diagnostics_integration.py` | 15 passed、2 skipped（官方模型分支不适用上传源码检查） |
| `test_prediction_analysis_cache_identity.py` | 3 passed |
| `test_mars_prediction_coordinates.py` | 49 passed、7 failed |

2026-10-09 本轮后端共 229 passed、2 skipped、7 failed。七项 Mars 坐标失败属于已记录范围：旧官方裸权重两项数值差异，NaN 坐标四项错误消息正则不符，masked 坐标一项提前被完整有限数据校验拒绝；不计入通过结论。重构前端全量回归 **807 passed**；后续空态占位修正运行相关展示/适配测试 **17 passed**，`npm run build` 成功；使用最新生产构建的合成浏览器 **42 项检查通过、无页面运行时错误**，包括无模型时 Earth/Mars 同为 `--` 和未运行时隐藏参考场说明。HTTP 409 的合成错误用于验证重试，浏览器网络错误日志属于预期。生产静态服务已更新并重启，后端 `/health`、前端页面和数据代理复验均为 200。

本轮未执行真实模型训练或真实预测精度验收，未验收完整分辨率长 horizon 的浏览器性能，未执行 MATLAB 重建脚本。全分辨率 Earth JSON 传输/内存和进程内缓存、多 worker 等既有边界仍适用。生产更新须先构建，再由 `scripts/serve-prod.mjs` 使用 `ROOT=frontend/dist`、`PORT=5173`、`API_PORT=8000` 运行；按工作区约定复核后端 `/health`、前端根页面和 `/api/datasets` 代理，三项 HTTP 200 才算运行验证。

## 2026-10-13 单模型页面外壳对齐本轮记录

移除 Earth 单模型页面的行星专属外壳（标题、说明段、`TO3 · DU` 胶囊、上下文折叠块、起点说明行、参考场说明段）、删除 Earth 独有的「刷新起点范围」按钮，并让完整测试集卡片显示与 Mars 同格式的分区行。前端 `node --test` 全量 **846 passed、2 failed**；两项失败（`experimentInspectorStructure.test.js` 的「访客 / 登出不再被悄悄切回官方模型」与 `trainingDefaults.test.js` 的「new experiments apply current defaults while copied configurations keep task values」）在未改动的 `f71071e` 树上同样失败，属于既有断言与训练页实现脱节，与本次改动无关。`npm run build` 成功。

浏览器侧用一个独立合成驱动核对：设备为 1440px 与 390px，Earth 与 Mars 各一张截图，**18 项检查通过、0 运行时错误**，覆盖「不再出现生成预测按钮」「不再出现刷新起点范围」「无引言标题」「上下文块/起点说明行/参考场说明均不存在」「保留逐步与累计窗口指标表」「完整测试集分区行等于 checkpoint 绑定」「三联图顺序与六项指标不变」「390px 无横向溢出」「Mars 仍为开始预测」。仓库自带的 `scripts/audit/shared-prediction/browser-check.js` 在本机缺少其期望的 Chromium 修订（缓存为 1243，CLI 要求 1247），且其语言切换步骤在未改动的树上同样超时，因此本轮以独立驱动为准；该脚本的 Earth 固定数据与断言已同步更新（补 `latitude_range`/`longitude_range`、发布分区绑定与分区行断言）。

运行验证：后端 `run_backend.py`（工作区约定的 Conda 解释器）+ `scripts/serve-prod.mjs`（`ROOT=frontend/dist`、`PORT=5173`、`API_PORT=8000`）在服务中，`/health`、`http://127.0.0.1:5173/`、`/api/datasets` 三项 HTTP 200；服务进程与端口属于实时状态，接手时需重新检查。本轮未改动后端代码，未执行真实训练或真实预测精度验收。

同一轮收口了 Earth 手动诊断面板（触发方式按既有约定保持手动）：`EarthDiagnosticPanel` 的文案迁到 `predict.earthDiagnostics.*`（zh/en 同名键），标题层级改用共用的 `prediction-chart-title` / `prediction-chart-subtitle`，面板外壳改用共用的 `prediction-panel`，未再使用的 `.earth-diagnostics-head`、`.earth-diagnostic-metrics`、`.earth-diagnostic-chart-grid`、`.earth-diagnostic-pfi-plot` 样式删除；`PredictMetrics` 新增 `flattened` 选项，避免它自带的面板外壳在诊断卡片内二次套框；PFI 的标题/范围/帮助文本只由共用卡片输出一次。该面板第二个独立合成驱动 **20 项检查通过、0 运行时错误**：进入页面只发 `GET /diagnostics/context`（不发计算请求）、计算按钮与默认参数文案来自 i18n、主标题/副标题使用共用类、无 `h4` 残留、四个子标题与 PFI 表头来自 i18n、`flattened` 后卡片内无嵌套面板、PFI 标题只出现一次、完整测试集与抽样诊断指标块仍分离、诊断科研导出仍未开放、切到英文全部文案随 i18n 变化且不触发重算。
