# 预测分析科研图导出

第一版入口在预测分析页面各图旁的「导出科研图」。仅消费已返回的预测与分析结果；导出端点不会训练、调用模型或在结果缺失时补算。

地球与火星单模型现由[共用预测工作台](shared-single-model-prediction.md)装配导出入口：`PredictDisplay` 使用当前步结果快照，`ErrorDistributionChart` 与 `PermutationImportanceChart` 通过 `predictionPresentation.js` 的配置决定能力。共用外壳不新增导出图种；Earth 当前步三联图可用，Earth 抽样诊断散点、残差直方图及 ΔRMSE PFI 继续禁用科研导出。`singleModelAdapters.js` 保留行星、时间、网格与物理单位契约，切换任务/起点/账号后旧快照不作为当前结果使用。

## 支持图表与统计口径

| 图表 | 实际绘图数据 | 范围 |
| --- | --- | --- |
| Mars 三联图 | 原始参考、预测、预测减参考；真实 `lat/lon` | 当前所选预测步，保留请求 Ls、实际输入/目标 Ls、窗口起点 MY/样本索引（已有响应提供时） |
| Earth 三联图 | 三小时原始 DU 场、真实经纬度、UTC 时间戳 | 当前所选步，保留 `origin_split`、起点与目标时间；不使用 Mars 单位换算；日频数据集已停用 |
| Mars 参考/预测散点 | 当前步全部格点，按纬度行、经度列顺序展开 | 当前所选预测步；普通散点，无密度估计、无抽样；等比例坐标、`y=x` |
| Mars 多模型逐步指标曲线 | 各模型已有 `metrics.per_step` | 各模型绑定的完整测试集；RMSE/MAE/R² 合并测试像素，SSIM 按现有服务聚合；不同数据集或 split 不应当作相同实验 |
| Earth 多模型逐步指标曲线 | 严格验证 checkpoint 保存的 test `by_lead`，由比较 API 整理为 `metrics.per_step` | 相同三小时发布、测试分区与窗口数的完整测试集；支持 RMSE/MAE（DU）和 R²（无量纲），步数按任务 horizon，每步 3 小时；导出透传 `schema`、`aggregation`、`metric_units` 与 `metric_policy`，明确区分 cell-area-weighted 与历史 grid-uniform；不重新推理，禁止 Earth/Mars 混合曲线 |
| Mars 单模型 PFI | 已有 `items.importance`、基线 R² | 最多 40 个均匀索引抽样测试窗口；每特征一次随机置换，ΔR²=基线−置换后 R²，保留负值 |

PFI 不依赖页面 Ls 起点，也不等于当前预测窗口或完整测试集逐格评价。新计算响应附 `sampling`（实际样本数、测试窗口数、抽样索引、置换方式、未固定 seed）；旧缓存缺少该信息时明确标记 `legacy_unknown`，不编造样本数。导出数据只能重画已有 PFI 值，不能复现随机置换实验。

Earth 训练后诊断的散点、残差直方图和 Earth PFI 当前只在页面展示，不新增导出类型；未来开放导出时必须消费诊断结果快照，不能在导出请求中重新推理。页面诊断使用固定 test 分区的有限窗口和明确种子，完整 test 指标与抽样指标分开标注。

共用工作台中的三联图和单图顺序为参考、预测、残差；两者的当前步三联图按钮仍导出该步全部三场。Earth 展示步选择只切换查看内容，每次运行完整任务 horizon，导出引用保持原始完整窗口身份及零基 step。地球值保持 DU，目标标签为提前小时与 UTC；Mars 请求保持原步长上限、Ls 标签及可选显示单位。Earth 地图放大与 Mars 球面放大均不产生新的科研快照。

旧误差分布 API 的 `scatter.density` 为兼容字段，全部为 1，不能作为密度估计。页面已改用当前步全部格点普通散点；误差直方图和其 RMSE/MAE 仍是完整测试集，单独标明范围。Mars 页面与导出统一采用 `1 μm-atm = 0.1 DU`，RMSE/MAE、直方图横坐标与场图读数同步转换；R²、SSIM、ΔR² 不转换。Earth 原有 DU API 契约不变。

Earth 页面与比较新增 MSE/MAPE/SMAPE，但现有导出 schema 只接受 RMSE/MAE/R²/SSIM，因此新增三项不开放导出按钮。旧模型仍可导出已有 RMSE/MAE；R² 导出只纳入逐步值完整的模型，至少两项。Earth 快照元数据透传指标 schema、逐项单位、aggregation 与公式/空间权重策略，旧快照无需补写字段；不自动训练或推理。完整公式及旧产物规则见[地球评价指标](earth-evaluation-metrics.md)。

## 导出面板与科研规范

- 单栏 85 mm（地图三联图上下排列，默认高 215 mm）、双栏 180 mm（三联图并排，默认高 60 mm）与自定义宽高；其他图可切换 85 mm 或 180 mm。
- 中英文、自动/无衬线/衬线字体、6–16 pt、0.3–3 pt 线宽、白底、色带、标题与三联图 `(a)/(b)/(c)`。复用全局格式、DPI、字号、标题、语言、单位和色带偏好，本次覆盖不写回全局设置；旧偏好的 150 DPI 提升为 300 DPI。
- PDF/SVG 的文字、坐标轴和曲线保留矢量元素；场图局部栅格化，超过 3000 点的散点局部栅格化。SVG 保留 `<text>`，接收端需要相同字体；PDF 嵌入 TrueType 子集。
- PNG 为 300/600 DPI，按毫米计算像素，不使用紧边界裁剪改变物理尺寸。预览为同一绘图函数的 120 DPI PNG，保持相同布局、坐标、单位及色阶。
- 当前步参考与预测共用原始数值极值色阶，残差以零为中心对称。全零残差用 ±1 源单位显示；恒定场用明确小范围扩展避免退化色阶。
- 经纬度与矩阵行列原样配对。平面格点用真实中心的相邻中点和端点半格距计算单元边界，`pcolormesh` 平面填色，不做平滑、插值、面积加权或重新排序。缺轴、重复轴、非单调轴、非有限值、三场轴不一致、残差方向错误均拒绝。
- 多模型同时使用颜色和线型，图例放在坐标区外；PFI 有零线、负值不截断。
- 多模型曲线在导出面板中单独选择本图的 2–8 个模型，保留页面原有对比选择。对比超过 8 个模型时初始不选中任何模型，等待用户明确选择，不静默截取前 8 个；未选足 2 个时不请求预览、不能下载。选满 8 个后先取消已有选择才能加入其他模型；预览、文件及 MATLAB 数据包均使用同一组引用与原有模型顺序。
- 后端从已安装字体选择 Arial/DejaVu Sans 或 Times New Roman/DejaVu Serif，并以 Microsoft YaHei/Noto CJK 等回退中文。逐字检查字体覆盖与图框裁切，布局失败提示增大尺寸或减小字号。生产 Linux 部署须提供可用 CJK 字体；未安装时中文导出明确报错，英文仍可用。
- 原生 `dialog` 支持键盘焦点约束、Escape、关闭后恢复焦点；390 px 窄屏按预览/参数纵向排列。预览加载、错误和下载状态有明确提示，过期预览不能下载，关闭/条件切换取消请求。

## API、身份与资源边界

入口：`GET /api/predict/research-export/capabilities`，`POST /api/predict/research-export/preview`，`POST /api/predict/research-export/download`。全部使用现有 Bearer 认证。

预测/PFI 响应新增 `export_ref`，比较接口在每个 `items[].metrics.export_ref` 返回。引用是服务器生成的随机快照 ID 及其完整条件（行星、任务、分析、起点、horizon、变量）；不包含 JWT 或服务器路径。请求传引用、`kind`、零基 `step`、`metric`、`options`、`bundle`。Schema 位于 [research_export.py](../AstraAtmos_backend/backend/schemas/research_export.py)，字段明确白名单并拒绝多余参数；不接受数组、文件路径、Python/MATLAB 代码。

按钮在结果仍在加载或缺少导出引用时禁用；切换条件后旧结果不作为可导出结果。Mars `/predict/run` 必须在手工整理响应时透传 `export_ref`，`/predict/metrics` 的响应 Schema 也须保留该字段，否则页面有图但没有导出引用。旧页面缓存需重新获取一次当前条件的结果；切回页面可能继续恢复浏览器内存中的旧缓存。用户再次点击「开始预测／开始对比」时，缺少引用的 PFI／比较结果会重新取回，不因条件相同直接跳过请求。已有引用过期、被淘汰或服务重启不会直接令按钮变灰，而是在预览/下载时返回明确错误。

服务器保留账号私有的结果快照，2 小时 TTL、128 MiB 总上限、最多 32 个快照，LRU 淘汰；应用重启失效。引用必须与快照条件逐项一致。渲染前后都复核快照有效性、任务权限、完成状态、模型文件与当前数据身份；Mars 使用既有 checkpoint/数据绑定及产物指纹，Earth 使用既有发布指纹与 checkpoint 校验。产物在结果获取期间变化时不给可用导出引用。旧页面缓存无引用、过期/已淘汰快照、模型或数据变化都要求重新获取结果，导出端点不会自动推理。

身份缺失的历史 Mars 模型仍遵循已有 legacy 规则，导出元信息保留该限制，不宣称历史训练数据身份已验证。多 worker 的快照不共享，首版部署应使用单个后端 worker；多实例需要粘性会话或后续共享快照存储。

最多 8 个曲线模型，每场最多 100 万格点，散点最多 25 万点且不静默降采样；毫米与 DPI 组合最多 2400 万像素。进程最多两个导出请求，Matplotlib 绘图串行保护全局字体/rc 状态。浏览器中止请求后，已启动的原生绘图继续占用名额至完成，避免中止预览绕过成本限制。忙碌返回 429，条件/身份变化返回 409，权限返回 401/403，参数/布局无效返回 422；文件响应禁止缓存。

参数错误在面板中仅展示字段和简短原因，不回显 Pydantic 的 `input` 或整批结果引用。旧页面将 31 个比较模型直接提交时会收到 422 `sources: too_long`，与字体、尺寸、DPI 无关；更新后的面板在请求前限制本图模型数量，更多模型需要分组导出。

## MATLAB 数据与脚本包

勾选「下载数据与 MATLAB 脚本」得到 ZIP：

- `figure.pdf/svg/png`：最终科研图。
- `figure.mat`：`figure_data` 保存实际使用的矩阵、坐标中心及单元边界、单位、共享/对称色阶、散点/曲线/PFI 数值、色带 RGB、字体、布局与 `metadata_json`。保存显示单位下的数值，重画脚本不再次换算。
- `rebuild.m`：MATLAB R2020b+ 重建对应类型图，便于调整字体、图例与布局；仅加载同目录 `figure.mat`，没有模型执行或动态代码。
- `metadata.json`：模型名/任务 ID、数据集/可用指纹、起点/目标日期或 Ls、步、window/horizon、范围、split/抽样信息、变量、单位、参数、残差定义与建议图注；MAT 内的 JSON 与该文件一致。`value_unit` 明确 R²/SSIM/ΔR² 无量纲，`unit` 保留场数据的单位设置。白名单不导出超参数全文、JWT 或服务器绝对路径。
- `README.txt`：使用说明与建议图注，保留各模型独立统计口径。

MATLAB 重建保持数据、坐标、色阶与图类型；MATLAB 字体、版本和布局引擎可能影响精确排版。**本轮未安装 MATLAB，未执行 `.m` 脚本**；MAT 文件已用 SciPy 回读检查。单个 `.m` 不是 MATLAB `.fig`。

## 验证与复现

从后端目录使用工作区规定的 工作区约定的 Conda 解释器执行 `-m pytest tests/test_research_export.py`，临时目录指定仓库外的新英文路径。该文件覆盖换算、共同/对称色阶、坐标顺序、残差方向、坏数据拒绝、权限与条件、过期、白名单、尺寸/DPI、SVG 真文字与局部栅格化、PDF MediaBox、MAT 回读和无敏感路径。

`tests/test_earth_3hourly_prediction_routes.py` 的已有合成 24 步响应增加第 24 步科研导出验证，明确禁止导出再次推理，逐项核对 MAT 的 DU 值、纬度与 UTC 目标时间。前端纯函数测试为 `researchExportModel.test.js`；构建从 `frontend/` 执行 `npm run build`。

共用单模型工作台的本轮验证入口另见[验证入口与边界](shared-single-model-prediction.md#验证入口与边界)：适配与展示回归核对 Earth DU、Mars 显示换算、当前窗口/完整 test/抽样诊断、ΔRMSE/ΔR² 负值及科研导出能力；[合成浏览器脚本](../scripts/audit/shared-prediction/browser-check.js)核对桌面/390px、主题/精度/色带、时间步/全屏、旧结果与错误恢复。该脚本替换预测 API，只验证页面和能力约束；实际科研文件格式与布局的验证仍以本文既有文件测试为准，不将页面合成预览视为 MATLAB 或真实精度验收。

布局样例生成器：[generate-fixtures.py](../scripts/audit/research-export/generate-fixtures.py)，以同一解释器从仓库根目录运行，参数为仓库外输出目录。可加 `--curve-models 8` 检查单图模型数量上限；默认 3 条合成曲线。生成中英文、单双栏、全部四类图的 PNG/SVG/PDF 与 MATLAB 包；全部为明确标记的合成布局数据，不能作为模型效果证据。

2026-10-08 实际检查：导出契约 16 passed、既有预测缓存集成 4 passed、Mars 路由 11 passed、Earth 日频 21 passed、三小时路由 24 passed；前端全量 747 passed 与生产构建通过。Poppler 渲染 PDF、Edge 渲染 SVG、PNG 视觉检查；浏览器合成 API 夹具检查窄屏、偏好覆盖、错误/禁用、下载事件、PFI/比较入口、Tab 双向焦点循环和 Escape 焦点恢复。浏览器夹具使用固定合成预览，只验证交互；图像布局/单位/色阶一致性由真实绘图函数与文件测试验证。后端重载后首次目录请求曾超时，完成初始化后复验 `/health`、前端与 `/api/datasets` 代理均为 200。未对真实模型重新训练或推理。

已有 `test_mars_prediction_coordinates.py` 为 49 passed / 7 failed；加载当前提交的原始 `inference_service.py` 重跑得到相同 7 项失败，涉及 legacy 官方模型数值与缺测坐标错误契约，不作为本轮通过结论。项目环境另有既有 NumPy/NetCDF 二进制兼容提示及框架弃用提示；构建有既有 bundle 体积警告。

同日按钮禁用问题复查：真实 FastAPI 响应流程复现了 `/predict/run` 手工响应丢失引用、`/predict/metrics` Schema 过滤引用两处遗漏，现已修正；新增测试从预测 HTTP 响应取得引用后下载三联图与散点，并禁止导出再次调用预测。本次导出及缓存测试 22 passed（18+4），Mars 路由 11 passed；前端全量 754 passed、生产构建通过，运行健康/前端/目录代理复验均 200。缺引用的旧 PFI/比较缓存可通过用户再次点击开始按钮取回。

同日 31 模型导出失败复查：新增面板模型选择与简明参数错误提示，导出契约测试 20 passed（含 31 引用在绘图前被拒绝、2/8 模型真实绘图及 MAT 曲线数量），前端全量 759 passed、生产构建通过。Edge 合成 API 夹具验证 31 模型比较保留、2/8 模型引用选择、预览/下载引用一致、无效选择不请求预览、简明 422、390 px 无横向溢出、Tab 焦点循环与 Escape 恢复。8 曲线合成样例经 Poppler 渲染 PDF、Edge 渲染 SVG、PNG 视觉检查，图例无重叠或裁切；PDF/SVG 为 180×100 mm，PNG 为 2125×1181 像素、300 DPI。健康与前端复验 200；目录代理的 PowerShell 请求曾超时，直接后端及随后的 curl 代理请求均为 200，没有将超时检查算作通过。没有执行真实模型推理或 MATLAB 脚本。
