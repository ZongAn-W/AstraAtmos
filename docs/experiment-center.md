# 实验中心（`#/training`）· 大气实验控制台

本页说明模型训练页现在的承载方式：**实验中心 · Atmospheric Mission Control**。版式与交互对齐设计参照 `output/training-console-preview.html`；实验中心是现有训练能力的**前端重组**，页面地址、训练接口、任务数据结构、日志与轮询机制都没有改变。

当前视觉收口遵循[前端共享令牌](../frontend/README.md#界面样式与可访问性)：实色面板、12px 主面板圆角、8px 控件圆角、16px 分区间距，正文 14px、辅助文字至少 12px / 18px 行高；11px 仅用于英文代号、单位或非主要元数据。冰蓝标记数据集/模型来源等选择状态，橙色用于训练主操作；检查器内部以分隔线和间距分组。下文带日期的改造记录是历史验收快照，其中旧字号、13px 圆角和 14px 间距不再代表当前样式。

深色训练页复用首页 `--bg-gradient` 与既有 `StarField`，取消原有深色局部光晕和实色 body 覆盖；面板、输入和图表保留实色，浅色主题不显示星点，首页鼠标视差与地球交互不延伸到训练页。

桌面配置视图的固定运行条默认约 56px，使用随主题变化的半透明表面、subtle 顶边和轻微阴影；正文初始预留条高 + 24px，挂载后继续按 ResizeObserver 实测高度调整，放大字号时也不遮挡页尾。访客只在底栏保留「登录后开始实验」这一主要行动，检查器登录项为辅助说明，不再提供第二个「去登录」按钮；未就绪徽章保持中性色，与橙色行动入口区分。

相关入口：

- 训练任务与标签：[训练模型标签](training-model-tags.md)
- 自定义模型接入：[自定义模型接入说明](uploaded-model-training.md)
- 数据集身份：[数据集注册表](dataset-registry.md)
- 已训练模型预测：[README 的预测与缓存章节](../README.md#已训练模型预测与缓存)
- 地球训练与历史预测（已开放）：[地球训练与历史预测](earth-training.md)

Earth 默认数据集读取 `/api/datasets` 的 `default_earth_dataset_id`。日频保留 7→3 和原上传契约；三小时使用 UTC、240×480、56→24，支持官方 DLinear 与[独立 v1 上传模型](earth-3hourly-uploaded-model.md)。三小时提供专用模板、自定义参数、可用/不可用/未知兼容性及原因，旧日频/Mars 结论不自动放行。任务与窗口独立，切换不迁移旧任务；数据包可用性由 descriptor 单独报告。

火星数据集约定：`mcd_overview` 只表示逻辑数据集身份，训练和预测均读取 `MCD_RAW_3H_DIR` 的原始全量 MCD MY24–MY35（默认 `data/MCD_Output_global_10m_ls_lst/`）；`data/mcd_overview/*.nc` 仅是生成的 overview 产物。`openmars_mcd` 始终读取 OpenMARS + `MCD_DIR`。

新火星 checkpoint 保存训练时每个输入通道的空间均值/尺度以及目标 `target_mean/target_scale`，预测和测试指标复用 checkpoint 参数；旧纯 `state_dict` 任务保持 legacy compatibility，并明确标记为重新拟合归一化。

## 1. 页面结构

```text
#/training
├── 页头视图按钮：实验配置 / 训练监视 / 实验矩阵
└── 控制台网格
    ├── 视图 1「实验配置」：配置画布 + 配置检查器（目录收起）
    ├── 视图 2「训练监视」：实验目录 + 画布（检查器收起；已完成任务在监视下方展示结果）
    └── 视图 3「实验矩阵」：可配置列的实验表格（任务轮询复用 TrainingContext）
底部：配置视图的提交运行条（position: fixed，横跨浏览器）；监控区自身提供停止训练 / 取消排队按钮
```

阶段（configure / monitor / result）仍由任务状态推导，视图只决定工作区布局。首次加载若已选中任务（运行中的任务优先，否则选列表最新记录）则进入训练监视；没有任务时默认实验配置。用户主动切换视图后，轮询不会覆盖选择。点「新建实验」或「复制配置」进入实验配置，选任务或成功开始训练进入训练监视。配置画布和实验矩阵始终保持挂载（只用 `hidden` 隐藏），所以切换视图不会清空表单，也不会丢失矩阵筛选、排序或横向滚动位置。

### 训练状态同步

`TrainingContext` 在训练页每 5 秒刷新任务列表，每 3 秒独立读取所选任务日志，WebSocket 提供额外的实时进度。排队/运行/完成状态和目录计数来自任务列表，日志不用于猜测状态。2026-10-17 修复轮询请求重叠导致的状态停滞：旧实现每次轮询都会作废上一请求，响应持续超过 5 秒时，所有响应均被丢弃，页面可能始终显示排队，而训练日志继续增长。

定时轮询现在复用同账号、同页面作用域内尚未完成的请求，允许慢响应更新状态与 loss history。提交、改名及状态消息触发的主动刷新仍发起新请求，防止操作前的快照覆盖新数据；账号/页面切换和卸载仍使旧响应失效。断开 WebSocket 时，HTTP 轮询仍能更新任务状态。修复只涉及前端，不要求重启正在训练的后端；生产页面构建更新后需刷新浏览器加载新资源。

隔离回归脚本为 [browser-check.js](../scripts/audit/training-status/browser-check.js)。在仓库根目录先用 `npx --yes --package @playwright/cli playwright-cli -s=training-status open http://127.0.0.1:5173` 打开页面，再执行 `npx --yes --package @playwright/cli playwright-cli -s=training-status run-code --filename scripts/audit/training-status/browser-check.js`。脚本拦截全部 API 与 WebSocket，以 6.2 秒任务响应和即时训练日志检查 queued→running→completed，不创建或停止真实任务。返回值应为 `updated=true`、`finalStatus=completed`、`maxInFlight=1`、`errors=[]`。

### 实验矩阵

实验矩阵直接消费 `TrainingContext` 的任务快照，不建立第二套轮询。每一行是一条训练任务记录，默认列为模型名称、标签、架构、数据集、状态、进度和开始时间；表头固定在顶部，选择框与实验名称固定在左侧。表格区域宽于 700px 时也固定左侧标签和右侧操作列；宽度不超过 700px 时，标签及操作列随横向滚动，实验名称缩至 180px，区域不超过 400px 时缩至 140px 并最多显示两行（悬停仍可查看完整名称）。该规则按表格实际宽度判断，因此展开属性面板后也不会让固定列占满表格。固定单元格使用不透明背景，避免滚动后的文字透叠。

矩阵外壳按面板顶部到视口底部的剩余高度实测尺寸，窗口和页头尺寸变化后重新计算。表格容器是矩阵的单一横向 / 纵向滚动区域，横向滚动条持续位于可见区域底部，不需要滚到长表格最后一行；滚动到边界不会把滚轮操作继续传给页面。全选和标签管理放在表格外的底部计数栏，避免额外一行工具栏挤压短横屏的表格。属性面板在桌面端侧置，窄屏堆叠且限制高度。状态实时变化时只更新任务快照，不覆盖正在编辑的名称草稿。

「显示属性」面板按基本信息、训练参数、架构参数、自定义参数和评价指标分组。属性搜索、勾选、拖动及键盘方向按钮调整顺序、全部显示和恢复默认均可用；列配置以 `astraatmos_experiment_matrix_columns:<user id>` 保存到当前浏览器，访客使用独立的 `guest` 键。历史任务没有保存的字段显示 `—`，不会用当前默认值补写；数值 `0` 和布尔值 `false` 保留。上传模型的 `custom_model_params` 使用 `custom:<key>` 标识，历史配置中的其他顶层扩展字段使用 `config:<key>` 标识，因此同名字段不会合并；内部路径、源码哈希、数据指纹等字段不进入属性清单。Earth 指标列读取 `metrics.splits.test.overall` 并标注完整测试集；MSE 为 DU²、RMSE/MAE 为 DU、R² 无量纲、MAPE/SMAPE 为 %，旧产物缺项显示“未提供”。Mars 顶层指标解析及原有展示保留，不同数据集不会计算或标记全局最优。

Earth 结果页默认显示完整测试集六项指标，并独立显示验证集；不能拿验证值补齐测试值。本次历史预测指标只属于选定窗口。新旧契约、公式、单位及比较缺失值规则见[地球评价指标](earth-evaluation-metrics.md)。

矩阵行内保留「查看详情」「复制配置」，「更多操作」菜单提供「用于预测」「模型测试」「删除实验记录」，均复用页面既有处理函数，不新增后端接口或批量删除。「用于预测」复用 `canUseTaskForPrediction`：任务须已完成、产物有效且未绑定已停用的数据集，选中模型通过既有 handoff 进入对应行星的预测模式。

「模型测试」仅对已完成且产物有效的任务开放；矩阵中的 Earth 任务在打开菜单后额外复核 checkpoint 与数据身份，核验期间禁用，不可用时显示原因并可重试。Earth 弹窗与预测页共用[训练后诊断](earth-post-training-diagnostics.md)：分别展示 checkpoint 完整 test 指标和固定 test 抽样范围下的 DU 散点、prediction-reference 直方图、ΔRMSE PFI；条件切换清空旧结果并取消请求，错误显示结构化 message，可重试。日频任务仍不能运行。

「删除实验记录」复用页面确认框与单任务删除接口，确认前说明会删除实验记录、关联标签关系、日志、训练权重及该任务的预测分析缓存，保留标签本身。排队、准备中和运行中的任务禁用删除，须先在训练监视中取消排队或停止训练；已完成、失败和已取消的记录可进入确认流程。

Earth 服务重启恢复要求严格校验并重建 checkpoint，任务指标恢复自产物；新任务的预期指标版本由服务器保存，不能用日志标记绕过完整指标门禁。下文日志解析恢复说明仅适用于原有 Mars 路径。

### 训练草稿切换与参数保留

`trainingDraftSession.js` 集中管理当前页面会话的 Earth/Mars 草稿转换及上传参数归属，页面应用完整转换结果。`openmars_mcd` 与 `mcd_overview` 共用 Mars 草稿，直接切换保留配置，任一数据集往返 Earth 都恢复模型来源、上传模型、通道、窗口/步长、划分比例、模型结构、迁移来源/冻结策略与解除迁移前的原结构。重复点击当前数据集不会改写任何快照。进入 Earth 清空当前迁移来源和结构恢复记录，切回 Mars 恢复其记录；Earth 仍强制关闭 SPHERE 与迁移，保持三小时契约。

上传参数按场景与模型归属保留：相同模型可在 Mars/Earth 保存不同参数，复制、场景恢复、迁移恢复及用户编辑的值在 schema 延迟到达或等价刷新后仍保留；schema 新增字段补默认值，手动改选另一模型应用该模型默认参数。场景快照深拷贝嵌套参数、结构和迁移恢复记录，编辑不会反向修改已保存草稿。通用默认值的编辑标记跨场景共享，窗口、比例与迁移策略的编辑标记随场景恢复，避免切换后的默认值同步覆盖用户配置。

“复制配置”和“新建实验”都清除旧场景快照：复制回填的新配置成为当前草稿，新建恢复默认上传参数；之后往返不会复活此前实验的草稿。名称、标签、轮次、批大小和学习率等通用字段继续由页面共享。草稿不写数据库或浏览器存储，刷新页面不承诺恢复。

行为测试为 `frontend/src/pages/ModelTrainingPage/trainingDraftSession.test.js`，覆盖转换序列、深拷贝、默认值编辑归属、迟到 schema、模型切换、复制/新建重置及最终 Mars/Earth 提交内容。浏览器回归脚本为 [browser-check.js](../scripts/audit/training-draft/browser-check.js)：先在 `frontend/` 构建，在仓库根目录执行 `npx --yes --package @playwright/cli playwright-cli -s=training-draft open http://127.0.0.1:5173/#/training`，再执行 `npx --yes --package @playwright/cli playwright-cli -s=training-draft run-code --filename scripts/audit/training-draft/browser-check.js`，结束后执行同会话 `close`。脚本读取当前页面 origin，隔离全部 API、WebSocket 和提交，检查桌面/390px，不写真实任务；截图放在忽略的 `output/playwright/`。合成验证不代表真实模型训练或精度验收。

本轮前端全量 `node --test` 为 817 passed，生产构建成功；最新生产构建上的合成浏览器回归 24 项通过，包含迁移中跨场景并解除锁定、复制后往返及新建后不恢复旧草稿，无页面运行时错误。既有前端生产服务更新后，后端 `/health`、前端首页和 `/api/datasets` 代理均返回 200；未创建真实训练任务。

排队任务的正常执行与重启恢复统一从持久化记录准备，不再依赖内存启动配方。Mars 迁移任务在启动前重新解析任务/上传权重来源并复核状态、文件与当前权限；管理员权限被撤销、来源被删除或失效时，任务会在启动子进程前失败，失败原因写入任务 `metrics`，后续任务继续按 FIFO 调度。Mars 上传模型还核对固定包版本、路径和源码摘要，Earth 继续使用冻结划分与嵌入源码。具体恢复约定见[训练队列](specs/2026-10-04-training-queue-design.md)。

矩阵支持名称、状态和标签筛选，适用列按数值或时间排序并把缺失值放在末尾。名称单元格可就地编辑，Enter 保存、Esc 取消，也有保存/取消按钮；沿用 255 字符、去首尾空格和账号内重名校验。排队、运行、完成、失败和取消记录都允许改显示名称，上传模型包名称仍单独显示。标签单元格复用账号私有标签选择器，支持搜索、多选、新建、批量增删和标签管理。标签编辑面板复用 MUI Popper，通过 portal 挂到 `body` 并使用视口固定定位，脱离固定单元格的层叠上下文和表格滚动裁切；靠近边缘时翻转或平移，尺寸随视口约束，搜索结果、新建标签或错误信息改变尺寸后重新定位。标签选项独立滚动，取消 / 保存操作区保持可见。Esc 取消并返回编辑按钮焦点，点击外部或切换离开矩阵视图会关闭浮层。行内操作可打开训练监视或复制配置，更多菜单提供预测、模型测试与单条删除；保存和删除后由同一任务刷新同步目录、矩阵和监视视图。

矩阵浏览器回归使用[隔离夹具脚本](../scripts/audit/experiment-matrix/browser-check.js)，在已构建的前端服务上运行：从仓库根目录执行 `npx --yes --package @playwright/cli playwright-cli open http://127.0.0.1:5173/#/training --headed`，再执行 `npx --yes --package @playwright/cli playwright-cli run-code --filename scripts/audit/experiment-matrix/browser-check.js`。夹具拦截浏览器 API 与 WebSocket，以 36 行、18 列和 40 个标签检查宽屏、窄屏、短横屏、边缘定位、滚动、保存 / 取消和权限失败；不写数据库，不启动训练，保存断言仅验证前端请求契约与刷新。结束后执行 `npx --yes --package @playwright/cli playwright-cli close` 清理隔离浏览器会话。截图输出到忽略目录 `output/playwright/`。

三个行操作另有[隔离交互夹具](../scripts/audit/experiment-matrix/actions-browser-check.js)，同一浏览器会话执行 `npx --yes --package @playwright/cli playwright-cli run-code --filename scripts/audit/experiment-matrix/actions-browser-check.js`。覆盖预测 / 测试状态门禁、Earth 核验失败和重试、按行星跳转、删除取消 / 权限失败 / 重试 / 防重复提交、桌面与窄屏菜单定位及 Esc 焦点恢复。所有测试与删除请求均由夹具拦截，不运行真实推理或删除真实任务；本轮该夹具的 33 项检查通过，生产构建成功。前端全量测试仍有两项与既有配置切换及 batch 默认值有关的旧源码断言失败，不属于本轮行操作验证范围。

**监控视图里监控常驻，结果接在下面**：任务完成后 `stage` 会变成 `result`，但监控工作区（状态、进度、Loss 曲线、实时日志）不再被结果顶替，因此点「训练监控」总能同时看到运行状态与结果指标。

## 2. 列宽、默认状态与吸附

| 视图 | 网格 | 说明 |
| --- | --- | --- |
| 配置实验（默认） | `minmax(0, 1fr) 284px` | 画布 + 检查器；目录收起。检查器不可用时退回单列 |
| 训练监控 | `235px minmax(0, 1fr)` | 目录 + 画布；检查器收起 |
| ≤ 1180px（桌面） | 配置 `minmax(0, 1fr) 250px`；监控 `200px minmax(0, 1fr)` | 侧栏收窄 |
| ≤ 900px | `minmax(0, 1fr)` | 单列，侧栏取消 sticky |

- 桌面端外边距 28px、列间距 14px，工作区上限 1500px 居中；**1920px 下不再按视口比例放大侧栏**（目录保持 235px、检查器 284px，多出的宽度留给画布）。
- `--experiment-*` 设计参数放在 `:root`，因为运行条被 portal 到 `body`，必须能在页面容器之外解析这些变量。
- 视图状态是唯一的列组合开关（`useState('config')`）：原来的「页头开关 + 阶段指示」已删除，页头替换为固定视图栏，因此不再有"手动收起目录后本次会话保持"这一层状态。

### 吸附与固定的三条规则

```css
@media (min-width: 901px) {
  .experiment-center-grid[data-view='monitor'] .experiment-directory {
    position: sticky; top: 86px;
    max-height: calc(100dvh - 86px - 72px);   /* 扣除导航偏移与运行条净空 */
    overflow: hidden; display: flex; flex-direction: column; min-height: 0;
  }
  .experiment-inspector-panel {
    position: sticky; top: 86px;
    max-height: calc(100dvh - 86px - 72px);
    overflow-y: auto; overscroll-behavior: contain;
  }
}
.experiment-run-bar { position: fixed; right: 0; bottom: 0; left: 0; }
```

1. **中间画布不加 sticky**，它保持正常文档流（`position: relative` 只为内部装饰定位）。
2. **运行条用 `position: fixed` 并 portal 到 `document.body`**：`App.jsx` 的页面切换动画在外壳上写了 `transform`，带 transform 的祖先会成为 fixed 的包含块，挂在页面内部时运行条会随页面滚动而不是贴住视口。portal 只影响渲染位置，不改业务。
3. 页面底部按**实测运行条高度**预留空间（`paddingBottom: runBarHeight + 20`），滚到页尾时最后一个字段完整可见，不被底栏遮挡。

目录外框固定在视口中（`overflow: hidden` + flex 列），列表自己滚动（`.experiment-directory-list` 用 `flex: 1 1 auto; min-height: 0; overflow-y: auto; overscroll-behavior: contain`）；内容少时不会人为撑出空长栏。

## 3. 配置画布

### 画布头部

### 画布头部

可编辑的**实验名称**（`.experiment-canvas-name`，占位符为命名示例）+ 一行任务说明（`EXPERIMENT / 001 · 火星臭氧时空预测`）+ 右侧就绪胶囊。旧的「当前配置」大段摘要已删除：摘要集中在右侧检查器，底部只留一行运行摘要。

### 四个部分

画布正文按「模型名称 → 数据集 → 模型 → 超参数」四段自上而下排列，每段都有 `01–04` 序号与标题（第 29 节）：

| 编号 | 分区 | 内容 |
| --- | --- | --- |
| 01 | 模型名称 | 可编辑的实验名称（`.experiment-canvas-head`，`data-config-group="name"`）与字段级错误 |
| 02 | 数据集 | 训练数据集把 Mars 与 Earth 选项直接列成单选列表（独占分区，选项横排），包括 `openmars_mcd`、`mcd_overview`、`earth_merra2_daily_v2`、`earth_merra2_3hourly_v1`；Earth 默认入口值来自服务端 catalog |
| 03 | 模型 | Mars 保留原来源/架构；日频 Earth 保留原官方/上传来源；三小时 Earth 开放官方 DLinear / 独立 v1 契约上传模型、dataset_id 兼容性和专用模板 |
| 04 | 超参数 | 侧边页签（148px）+ 右侧真实字段；页签依次是输入与预测、训练参数，再按模型来源追加专家字段页签；Earth 使用各自时间步与任务级原始 UTC 分区，无迁移学习 |

四段各自是一张卡片（surface-1 + 13px 圆角 + 细描边，段间 14px 留白），卡片内左侧 38px 是编号轨道（第 30 节）。
卡片标题已经写明「数据集」「模型」，卡内的「训练数据集」「模型来源」两个小标签已删除（第 32 节），
分组语义保留在 `role="radiogroup"` / `role="group"` 的 `aria-label` 上。
训练参数的每一格只有「标签 + 数值」，下面的量纲小字（时间步 / 预测步 / epochs / 样本 / 学习率）已删除（第 33 节）。

- 输入与预测、训练参数是超参数模块的头两个页签（第 25 节）；画布不再有独立的载荷 / 训练参数分区。
- 参数矩阵的辅助标识用 `WINDOW`、`BATCH`、`LR` 这类短码，界面不暴露 `windowValue`、`batchSize` 等内部变量名。
- 参数矩阵在超参数面板里按可用宽度自动排布（`repeat(auto-fit, minmax(150px, 1fr))`），窄视口下由既有断点回落三列 / 两列 / 单列。

### 超参数页签

页签集合**按模型来源决定**，不出现要求用户“去另一个地方”的空页签；权重上传属于迁移学习，不再单独占页签：

| 来源 | 页签 | 字段 |
| --- | --- | --- |
| 两种来源（前置） | 输入与预测 / 训练参数 | `O₃ 基础输入` + U/V/D/S/T 驱动胶囊 + 计数；WINDOW / HORIZON / EPOCHS / BATCH / LR 参数矩阵 |
| 上传模型 | 自定义参数 / 训练策略 / 迁移学习 / 实验标签 | 自定义参数（`param_schema` 动态表单）、随机种子、早停轮数、启用开关 / 来源类型 / 来源任务 / 权重文件 / 冻结策略 / 微调学习率 / 严格匹配、`TagPicker` |
| 官方模型 | 模型结构 / 训练策略 / 迁移学习 / 实验标签 | ST-LSTM 层数、各层隐藏维度或当前架构的专属结构参数、随机种子、早停轮数、迁移与标签同上 |

六个页签的正文用**同一套字段块**（三列网格 + surface-2 面层块，见第 26 节），切换页签时右侧形态一致。

原「输入序列 → 当前模型 → 输出序列」序列示意图已在信息减法中删除，图示标题与帧数说明随之不存在。

页签面板渲染的是**真实对应字段**（切换后 `data-expert-panel` 与字段一起变化，不是只换标题）。页签状态由页面控制器持有（`expertTab` / `onExpertTabChange`），因此检查器的「编辑自定义参数」可以直接切页并聚焦第一个参数；表单状态全部来自控制器，切页签或收起目录都不会丢值。

## 4. 上传模型是主入口

| 情形 | 界面表现 |
| --- | --- |
| 默认 | `modelSource` 初始值就是 `uploaded`；**登出不会改回官方模型** |
| 账号没有上传模型 | 同一区域给出「上传 .py」、`该模型尚未选择` 说明、模板 / 说明下载，区内 inline error 解释原因 |
| 已选择上传模型 | 紧凑卡片：截断的显示名称（缺失时回退原始文件名，`title` 给出全名）、`v版本 · N 个自定义参数`、状态胶囊（可训练 / 待校验 / 不可用）；动作只有「上传模型」与次要的「管理模型」、「下载说明」、「下载模板」 |
| 管理模型（展开） | 全部模型列表（选择）、下载源码、重命名（名称输入 / 保存 / 取消）、重新校验、上传模型、删除、校验详情（第 31 节删掉了「文件格式要求」与底部「校验状态」两行） |
| 校验失败 / 版本不匹配 | 区域内 `role="alert"` 的 inline error 逐条列出原因 |
| 没有有效上传模型 | 运行条「开始实验」保持 `disabled` + `aria-disabled="true"`，就绪状态显示「配置未完成」 |
| 自定义参数 | 继续由 `DynamicModelParamsForm` 按上传模型 `param_schema` 渲染；编辑入口只保留在配置检查器，画布上的上传卡片不再重复提供 |

官方模型是次级兼容入口：切到官方模型后隐藏上传区域、显示紧凑架构选择器（当前值 + 展开模型库：搜索 + 家族筛选 + 全部 29 个架构）与 SPHERE 开关。

火星与地球的上传模型使用同一套管理动作和中英文文案。下载源码与重命名通过认证接口校验上传者；下载返回原始上传的 `.py` 文件，改名只更新所选包的显示名称（1–120 个字符），不会改写源码、版本或历史训练任务快照。重新校验保留编辑后的名称。接口约定见[自定义模型接入说明](uploaded-model-training.md#downloading-and-renaming-uploaded-models)。

## 5. 配置检查器

检查器只做摘要，不复制整张表单（组件内不渲染任何 `input` / `select`）。顶部显示已完成分区数、警告数、问题数及实际可训练状态；四个折叠分区依次是模型名称、数据集、模型来源、超参数。有问题的分区默认展开，问题来自页面控制器现有的 `readiness.blockers`，定位按钮滚动到对应表单分区或专家页签。自定义参数只显示一个汇总问题，避免在第 04 分区重复列出每个字段。没有独立 warning 校验来源时警告数为 0；“通过”表示当前无已知阻塞项，最终可训练性仍以 `readiness.canTrain` 为准。

左侧目录复用原有搜索、标签筛选和批量标签操作；状态筛选按两列排布。实验搜索占满一行，排序与标签筛选按钮始终等高并排；展开面板独立占满侧栏宽度，不挤开排序控件。面板依次提供全部 / 未分组切换、标签搜索、紧凑多列标签选项和底部管理入口，短标签并排、长标签自动换行；取消重复的「按标签筛选」可见标题与内层边框，保留字段组的可访问名称。选项独立滚动，收起面板后仍保留筛选，按钮显示已启用条件数量。列表增加最近实验、运行中、已完成、失败或需修复分组及折叠，支持最近更新、最早更新、名称排序。最近实验最多列出三条，不在其他分组重复展示。每张目录卡片仅展示实验名称和全部实验标签。标题行左侧提供批量选择框，右侧为常驻的标签编辑铅笔图标；名称最多两行。全部标签直接显示在独立区域，用弱底色、小圆角与紧凑间距自动换行，不与选择或编辑控件混排。点击卡片查看实验，选择框与编辑图标保留可访问名称，编辑图标提供提示。模型、数据集、通道、进度和指标详情在监控、结果或矩阵工作区查看。当前任务接口不提供持久化草稿，因此目录没有伪造“草稿”分组。停止训练与取消排队由监控工作区提供；复制配置、预测、模型测试和删除可从结果工作区或实验矩阵进入。

**删除资源估算**：过去用 `window − horizon + 1` 推导“训练样本数”，那不是数据集长度。没有真实数据长度、切分与采样信息时不给样本数、显存或耗时。

## 6. 就绪状态：三处必须一致

```js
const readiness = useMemo(() => {
  const blockers = [];
  if (!user) blockers.push({ code: 'login', … });
  if (!customModelName.trim()) blockers.push({ code: 'name-missing', … });
  if (modelSource === 'uploaded' && !selectedUploadedModel) blockers.push({ code: 'model-missing', … });
  // …上传模型校验、自定义参数、脚本可用性、迁移来源
  return { canTrain: blockers.length === 0, blockers };
}, […]);
```

- **`canTrain`**：能否开始**真实训练**（登录 + 校验全过）。
- **`startDisabled`**：主按钮是否允许交互。访客必须可点（点了弹登录），因此两者是不同的问题。
- 检查器（`data-inspector-state`）、画布胶囊（`data-canvas-state`）、运行条（`data-run-state` + `data-run-interactive`）都读这一个 `readiness`，不会出现“列着错误却显示可以开始训练”。
- 修复前的问题：访客 + 缺名称时 `startDisabled === false` 被当成就绪信号，检查器显示绿色「可以开始训练」，同时下方又列出「登录后才可以开始实验 / 缺少实验名称 / 未选择上传模型」。

## 7. 底部运行条

- 桌面配置阶段：`position: fixed`，横跨浏览器；深色连续底栏 + 上方一条细边线；普通字号下约 63px。
- 结构：左侧状态（`配置就绪 / 配置未完成 / 登录后即可开始实验`）+ 单行摘要（`数据集 · 模型 · 输入 · 轮数`，超长用省略号截断，`title` 保留完整信息）；右侧唯一主按钮。
- 放大文字时允许换行变高（1.3 倍字号实测 66px），不裁掉内容；≤ 900px 回到普通流并整行显示。
- 「开始实验」调用页面控制器的 `handleStartTraining`，与改版前完全同一条链路：实验名唯一性 → 官方脚本可用性 → 上传模型校验与自定义参数 → 迁移来源 → `startTrainingTask(script, hyperparameters, name, dataSource, { modelSource, uploadedModelId, tagIds })`。运行条**不做任何自己的校验**。

## 8. 阶段与任务选择

| 阶段 | 进入条件 | 主要内容 |
| --- | --- | --- |
| 配置实验（configure） | 点击“新建实验”或“复制配置”进入配置视图 | 画布四个部分（模型名称 / 数据集 / 模型 / 超参数）+ 右侧检查器 + 底部运行条 |
| 训练监控（monitor） | 当前任务状态为 `pending` 或 `running` | 进度、Epoch、当前 Loss、ETA、Loss 曲线、终端式实时日志、停止训练、自动跟随状态 |
| 实验结果（result） | `completed`、`failed` 或已停止等其他终态 | 指标、失败/停止原因、折叠运行日志、完整参数与数据集身份、用于预测 / 去模型比较 / 复制配置 / 重命名 / 模型测试 / 删除 |

阶段由纯函数 `getExperimentStage({ activeTask, isCreating })`（[experimentCenterModel.js](../frontend/src/pages/ModelTrainingPage/experimentCenterModel.js)）推导：`isCreating` → configure；无活动任务 → configure；`pending`/`running` → monitor；其他 → result。

配置工作区容器**始终挂载**（`hidden` 隐藏），切换视图不会清空正在编辑的配置；监控和结果工作区仅在需要时渲染。

### 任务自动选中与抑制

`TrainingContext` 每 5 秒轮询任务列表，并用 `reconcileActiveTrainingTaskId(tasks, preferredTaskId, { suppressAutoSelect })` 决定当前任务：

- 刷新页面后，账号里运行中的实验仍会被自动选中并进入监控阶段（既有行为）。
- 用户点「新建实验」或「复制配置」进入配置阶段后，页面调用 `setSuppressAutoSelect(true)`：**轮询或目录重渲染都不会再把用户拽回监控阶段**。此前展开目录会立刻跳进监控，正是这个原因。
- 用户主动点击目录里的实验（`handleSelectTask`）或返回目录时解除抑制。

## 9. 现有能力如何迁移

| 原有能力 | 现在的位置 |
| --- | --- |
| 官方模型 / 上传模型切换 | 「模型」分区的模型来源胶囊（上传模型默认、官方模型次级入口） |
| 上传模型列表、校验、重新校验、删除、上传与替换、模板与说明下载 | 上传模型卡片 + 「管理模型」展开区（含区内 inline error） |
| 自定义模型参数动态表单与校验 | 超参数「自定义参数」页签（检查器有一级入口） |
| 模型架构、SPHERE、结构参数 | 「模型」分区的官方模型区域 + 超参数「模型结构」页签 |
| 迁移学习、上传权重、冻结策略、微调学习率 | 超参数「迁移学习」页签 |
| 训练标签与批量标签 | 超参数「实验标签」页签；目录标题行的批量选择框与标签编辑图标、卡片独立区域中的全部标签，以及目录筛选、批量标签与标签管理 |
| 进度、Loss 曲线、日志与自动跟随 | 监控工作区（`TrainingProgressMonitor`、`LossEvolutionChart`、原日志面板样式） |
| 停止训练 / 取消排队 | 监控工作区 |
| 重命名 | 结果工作区；实验矩阵名称单元格内编辑 |
| 删除、单任务模型测试 | 结果工作区；实验矩阵「更多操作」菜单 |
| 用于预测 | 结果工作区；实验矩阵「更多操作」菜单，继续使用 `TRAINING_TASK_HANDOFF_KEY` 与 `buildTrainingTaskHandoff` |
| 复制配置 | 结果工作区；实验矩阵行内按钮，载入新建实验表单，不自动开始训练 |

工作区组件、检查器与运行条都不调用 `startTrainingTask`、`fetchLogs`，也不创建 `setInterval` 或 WebSocket；训练提交、日志轮询与实时更新仍由 [ModelTrainingPage.jsx](../frontend/src/pages/ModelTrainingPage.jsx) 与 [TrainingContext.jsx](../frontend/src/contexts/TrainingContext.jsx) 持有。

实时日志仍是**终端式输出区域**：`role="log"`、等宽字体、行号 gutter、按级别着色、自动跟随与手动滚动暂停。它没有被静态图表替换，也没有引入浏览器内伪终端。

## 10. 结果动作的口径

- **用于预测**：复用 `canUseTaskForPrediction`，仅对已完成、产物有效且未绑定已停用数据集的任务开放；写入预测 handoff，火星跳转 `#/predict?from=training&mode=trained`，活动三小时地球任务使用 `mode=earth`。入口位于结果工作区与实验矩阵「更多操作」菜单。
- **去模型比较**：同样只对 `completed && model_available` 开放，火星跳转 `#/predict?from=training&mode=trained_compare`，活动三小时地球任务使用 `mode=earth_compare`；预测页先选行星、再选单模型/多模型。没有批量预选协议，因此不写入任何选择结果。
- **复制配置**：读取任务完整训练配置载入新建实验表单，实验名加“副本”后缀并保证唯一，字段缺失时按 `warnings` 逐条提示，任何情况下都不会自动开始训练；上传模型实验会带回上传模型 ID、版本、名称与 `custom_model_params`（深拷贝，由现有参数表单按 schema 渲染）。
- **重命名 / 模型测试 / 删除**：复用页面既有处理函数，结果工作区保留入口，实验矩阵支持名称就地编辑、菜单内模型测试和单条删除。测试仅允许已完成且产物有效的任务，Earth 须通过 checkpoint / 数据身份核验。删除沿用确认框与单任务接口，说明日志、权重及该任务预测分析缓存的连带清理；活动任务禁用删除，先停止或取消，不提供批量删除。

### 复制配置覆盖的字段

| 分组 | 字段 |
| --- | --- |
| 命名与标签 | 任务名称（追加副本后缀并保证唯一）、标签 |
| 数据 | 训练数据集（`dataset_id`，兼容旧字段 `hyperparameters.training_dataset`；地球任务回到训练表单时会重新切到地球模式并恢复规范通道顺序与 7/3） |
| 模型 | 模型来源、上传模型 ID 与版本、上传模型名称、模型架构、SPHERE |
| 上传模型参数 | `custom_model_params`（深拷贝；仅上传模型实验携带） |
| 输入 | 输入通道、输入窗口、预测步长 |
| 训练 | 训练轮次、批大小、学习率、随机种子、早停轮数 |
| 结构 | 循环网络的 `stlstm_hidden_dims`、非循环网络的专属结构参数 |
| 迁移 | 一律关闭，来源类型、来源任务、冻结策略、微调学习率回到默认值 |

### 失败与停止原因

后端把结果写在任务 `metrics` JSON 里：失败写 `error`，CUDA 显存不足额外写 `error_code: cuda_out_of_memory`，用户停止写 `note`，训练成功但没有有效权重写 `error: Invalid model artifact…`。`getExperimentFailureMessage(task)` 逐级解析并映射到可读原因与恢复建议，非法 JSON 与非法字段安全回退；结果工作区显示原因并提供默认折叠的“查看运行日志”（复用 `TrainingContext` 的 `logs`，不新增请求）。如果训练在创建日志文件前就失败，日志接口会把 `metrics.error` 返回为首行，避免显示无信息的日志占位文本。服务重启时，带有完整 `Model saved:` 日志且权重文件有效的运行中任务会恢复为已完成并重建指标，其余运行中任务才记录“训练被服务重启中断”。

## 11. 未开放的功能边界

- **没有后端实验实体。** 仍是 `ModelTrainingTask`、`TrainingModelTag`、`TrainingTaskTag`；没有新增数据库表、API 或训练字段。
- **没有配置草稿。** 表单只在页面内存，刷新即丢失；界面**没有**假的“保存草稿”按钮（设计参照里的草稿提示是概念稿演示）。
- **没有实验备注、独立收藏和实验组。**
- **没有自动实验结论。**
- **没有资源估算。** 见第 5 节。
- **个人上传数据仍不能直接作为训练数据。** 训练数据由服务器管理。
- **Earth 其他官方架构、SPHERE 和迁移学习未开放。** 日频支持原 Earth 上传契约，三小时支持独立 v1 上传契约；两者均保留官方 DLinear。
- **没有跨实验批量重跑或多任务并行调度。**

## 12. 代码入口与验证

| 文件 | 责任 |
| --- | --- |
| [ExperimentCenterShell.jsx](../frontend/src/pages/ModelTrainingPage/ExperimentCenterShell.jsx) | 固定视图栏（实验配置 / 训练监视 / 实验矩阵三个按钮）、控制台网格与列组合、运行条 portal 与底部空间预留 |
| [ExperimentDirectory.jsx](../frontend/src/pages/ModelTrainingPage/ExperimentDirectory.jsx) | 左侧目录、状态筛选与轻量实验行 |
| [ExperimentMatrix.jsx](../frontend/src/pages/ModelTrainingPage/ExperimentMatrix.jsx) | 实验矩阵表格、筛选排序、名称与标签编辑、批量标签、属性面板及预测 / 测试 / 单条删除菜单 |
| [ExperimentMatrixActions.jsx](../frontend/src/pages/ModelTrainingPage/ExperimentMatrixActions.jsx) | 矩阵行内按钮与「更多操作」菜单、状态门禁、按需 Earth 测试核验与重试；调用页面既有预测 / 测试 / 删除流程 |
| [experimentMatrixModel.js](../frontend/src/pages/ModelTrainingPage/experimentMatrixModel.js) | 矩阵属性定义、历史配置整理、列偏好恢复、筛选与排序 |
| [ExperimentConfigWorkspace.jsx](../frontend/src/pages/ModelTrainingPage/ExperimentConfigWorkspace.jsx) | 画布四个部分（模型名称 / 数据集 / 模型 / 超参数）、超参数页签（输入与预测载荷条、训练参数矩阵、专家字段） |
| [ExperimentConfigInspector.jsx](../frontend/src/pages/ModelTrainingPage/ExperimentConfigInspector.jsx) | 右侧简短摘要与就绪检查，纯展示 |
| [ExperimentRunBar.jsx](../frontend/src/pages/ModelTrainingPage/ExperimentRunBar.jsx) | fixed 运行条的状态、单行摘要与唯一主操作 |
| [ModelArchitectureSelector.jsx](../frontend/src/pages/ModelTrainingPage/ModelArchitectureSelector.jsx) | 官方架构紧凑选择器（当前值 + 搜索 + 家族筛选） |
| [UploadedModelPanel.jsx](../frontend/src/pages/ModelTrainingPage/UploadedModelPanel.jsx) | 上传模型紧凑卡片与可展开的管理区 |
| [ExperimentRunMonitor.jsx](../frontend/src/pages/ModelTrainingPage/ExperimentRunMonitor.jsx) | 进度、Loss、实时日志与停止训练 |
| [ExperimentResultPanel.jsx](../frontend/src/pages/ModelTrainingPage/ExperimentResultPanel.jsx) | 结果指标、失败原因、上传模型名称/版本/文件身份、折叠日志与后续动作 |
| [ExperimentLogPanel.jsx](../frontend/src/pages/ModelTrainingPage/ExperimentLogPanel.jsx) | 只读终端式日志面板，不请求日志 |
| [experimentCenterModel.js](../frontend/src/pages/ModelTrainingPage/experimentCenterModel.js) | 阶段推导、摘要、指标解析、目录筛选、复制配置、失败原因与官方架构注册表 |
| [experimentCenter.css](../frontend/src/pages/ModelTrainingPage/experimentCenter.css) | 控制台网格、sticky 侧栏、fixed 运行条、画布分区与响应式规则 |
| [trainingTaskSelection.js](../frontend/src/contexts/trainingTaskSelection.js) | 当前任务推导与「新建配置期间不自动选中运行中任务」 |
| [trainingStatusMeta.js](../frontend/src/pages/ModelTrainingPage/trainingStatusMeta.js) | 训练状态配色与文案的单一来源 |

### 界面约定（改版时必须一起维护）

1. **只有一条提交路径。** 画布内不得出现第二个 `onStart`，运行条不得自己调用训练接口。
2. **侧栏 sticky、运行条 fixed + portal。** 不要给运行条套回页面外壳内，也不要给中间画布加 sticky。
3. **目录只属于监控视图。** 配置视图不渲染目录；手动切换视图后，任务轮询不得覆盖用户选择。
4. **就绪状态只有一个来源。** 检查器 / 画布 / 运行条都读 `readiness`，不要各自判断 `startDisabled`。
5. **字段栅格用 `minmax(min(240px, 100%), 1fr)`**，`--experiment-*` 设计参数放 `:root`（运行条在 body 上）。
6. **官方架构清单只改一处**：`experimentCenterModel.js` 的 `ARCHITECTURE_LABELS` 与 `trainingParamSanitizers.js` 的结构参数配置。
7. **文字层级只用一套令牌。** 页面里的 `font-size` 必须写成
   `calc(var(--experiment-text-*) * var(--font-scale, 1))`，不再写死像素值；桌面断点的整体上调集中在
   [experimentCenter.css](../frontend/src/pages/ModelTrainingPage/experimentCenter.css) 末尾的
   「桌面端文字层级」媒体查询，见第 15 节。
8. **分组靠面层，描边留给状态。** 容器与卡片只从 `--experiment-surface-1/2/3` 取背景，
   不再写死 `rgba(13,29,46,0.x)`；整圈描边只允许出现在「选中 / 聚焦 / 错误 / 可点控件」上，
   静态内容容器一律 `border: 1px solid transparent`。新增容器前先读第 16 节的框线清单。
9. **并排分组按信息量自然排布。** `.experiment-task-grid` 用 `align-items: start`，不要改回
   `stretch`：把「训练数据集」拉到与「模型来源」等高会凭空多出约 150px 空白（见第 17 节）。
10. **检查器是摘要，不是第二张表单。** 区块之间最多 3 条分隔线（头部与最后一个区块都不画线）；
    通过的 ✓ 用淡底 + 绿环，未通过的 ! 才用实心橙底；正常状态一律比错误状态克制（见第 18 节）。
11. **目录的三种空状态必须分开。** 展示判定只用 `data-directory-state`（`loading` / `error` /
    `empty` / `ready`）与真实加载标记，不要用登录信息或任务数量反推；取任务失败时不得显示
    「暂无实验」（见第 19 节）。
12. **实心火星橙只给「真的可以开始实验」。** 运行条主按钮在未就绪与未登录时改用中性底
    （禁用态加细描边、未登录态用橙字保留可点性），就绪态才是页面上唯一的实心橙按钮；
    按钮尺寸 44px / 150px 不得缩小（见第 20 节）。
13. **新实验配置视图不做重复说明。** 已经由控件本身表达的信息（来源副标题、分区提示、
    数据集复述、实现说明）与装饰（分区序号、角标、英文小码、`.PY` 徽标、统计）不再新增；
    常态摘要只出现在一个地方（画布控件本身），检查器只讲「当前模型 + 阻止开始的问题」。
    需要保留的例外只有：控件名称与当前值、选中 / 禁用 / 加载状态、错误与警告、字段量纲、
    以及**用户选择时必须知道的独有差异**（如 MCD 数据集的来源与覆盖年份）——见第 21 节。

从 `frontend/` 执行：

```bash
node --test
npm run build
```

## 13. 既有浏览器验收记录（本轮未重复执行）

前次实现的验收记录记载：使用 Edge 无头浏览器（Chrome DevTools Protocol）对本地构建产物做了 **39 项控制台检查 + 22 项阶段检查，全部通过**，并保留截图对照设计参照（`shots/final/`：预览 1440/1920、真实页面默认两列 / 展开三列 / 滚动 600px / 页尾 / 1920 / 1024 / 390 / 字体放大 1.3 / 浅色英文）。

控制台检查（登录使用真实账号 + 真实上传模型，参数 schema 为 `probe_alpha` / `probe_ratio` / `probe_flag`）：

- 1440×900：默认两列 `1008px 284px` 且目录未渲染；检查器 284px、sticky；运行条 `fixed` 宽 1434px（等于视口）、高 63px；画布不是 sticky；无横向溢出；检查器无内部滚动条（655px 高）。
- 画布：可编辑实验名称存在；不再有重复的「当前配置」摘要；任务定义两列（478 / 478）；参数五列且辅助标识为 `WINDOW,HORIZON,EPOCHS,BATCH,LR`；专家参数侧边页签 `148px 816px`；载荷条与序列图示存在。
- 展开目录：`235px 759px 284px`，目录 235px、检查器 284px，两者 `sticky`，无横向溢出；滚动 600px 后两者 `top` 均为 **86px**，运行条仍 `bottom=900`（视口 900）。
- 页尾：可滚到 `892/892`，专家分区底边 639 在运行条上沿 837 之上（不被遮挡），运行条仍贴底。
- 1024×768：`200px 412px 250px` 三列、无横向溢出；390×844：单列、侧栏 `static`、运行条 `static`、上传入口与页签可用、主按钮 44px。
- 字体放大 1.3：运行条自适应到 66px、无横向溢出。

阶段检查（注入到本地库的 running / completed 任务走真实任务列表与日志接口）：

- 监控：`running` 任务进入监控阶段、目录默认展开，进度 42.5%，`role="log"` 终端日志 15 行含 epoch / loss 行与 warning 着色，自动跟随状态、停止按钮与 Loss 曲线齐备。
- 结果：`completed` 任务进入结果阶段，RMSE / MAE / MSE / R²、完整参数与数据集身份、折叠的「查看运行日志」（展开 21 行含评估行）都在。
- 复制配置：回到配置阶段并载入画布，实验名「副本 2」、Epochs 8，`probe_alpha=23 / probe_ratio=0.35 / probe_flag=true` 原样回填，检查器显示上传模型。

本轮修复的三个状态错误：

1. **访客 + 缺名称时检查器显示绿色「可以开始训练」**：`startDisabled` 为访客返回 false（为了允许点击登录）被当成就绪信号。现在由 `readiness.canTrain` 决定就绪，与按钮可点性分离，三处显示一致。
2. **检查器编造训练样本数**：删除按 `window − horizon + 1` 推导的资源估算区。
3. **展开目录把用户拽出配置阶段**：`TrainingContext` 的 5 秒轮询会在 `preferredTaskId` 为空时自动选中运行中的任务，用户点「新建实验」后目录重渲染即触发。现在新建配置期间抑制自动选中，用户主动选实验时解除。

另修复登出时把模型来源悄悄切回官方模型（此前访客永远看不到「上传模型」主入口）。

已知边界：账号里存在运行中的实验时，首次加载会自动进入监控阶段（既有行为），验收脚本因此等待配置阶段稳定后再检查配置项。验收用的临时账号、上传模型、任务与日志已由 `seed-probe-profile.mjs cleanup` 清理；运行训练子进程要求 `TRAINING_PYTHON_PATH` 指向真实解释器，仓库内 `.env` 默认仍是占位路径（不影响本页面的阶段推导与判定）。验收中没有启动真实训练。


## 14. 2026-09-27 布局与交互精修

- 页边距只由控制台容器提供：1440px 桌面两侧各 28px，去掉页面外层重复的 36px 内边距；标题改为局部标题样式，避免共享标题组件额外的 32px 留白。任务定义左右区域等高。
- 未选择上传模型时显示一次中性的上传指引；模型校验失败仍在上传区显示错误。新建画布不再展示固定的示例实验编号。
- 检查器过滤空白登录行，普通阻塞原因只展示一次，具体自定义字段错误仍保留。未登录或没有有效模型时，参数检查不会显示绿色通过。
- 画布和检查器的「编辑自定义参数」均切换页签、滚动并聚焦目标；导航上方留出滚动间距。
- 侧栏底部净空使用运行条的实测高度，桌面设置浮钮移到操作条上方，避免遮挡状态文本。
- 本轮只运行配置、检查器、外壳与日志面板的 52 项现有测试；其中旧的固定侧栏净空断言随实测高度规则更新后，该文件的 18 项复跑通过，其余测试首次通过。生产构建通过（仍有既有的大包体积提示）。1440px 浏览器确认两列画布、展开后 235px 目录与 284px 检查器、滚动吸附及全宽底部运行条；没有启动训练，也没有重跑上一节的全量阶段验收。

## 15. 2026-09-28 桌面端字体层级

上一轮版式在 1440×900 下大量使用 9–11px 承载说明、字段标签、检查器信息与底部运行摘要，读起来偏小。
本轮只调整**桌面断点（≥ 901px）**的文字层级，不改位置、结构、交互、状态配色与训练流程；
手机与平板（≤ 900px）保持原样。

### 字号令牌（三级层级）

类型全部走 `--experiment-text-*`，定义在 `:root`，`calc(令牌 * var(--font-scale, 1))` 保留全站文字缩放。
基准值等于改造前的原始字号，因此 ≤ 900px 渲染不变；桌面媒体查询只上调这几个令牌：

| 层级 | 令牌 | 基准（≤ 900px） | 桌面（≥ 901px） | 典型位置 |
| --- | --- | --- | --- | --- |
| 辅助说明 | `--experiment-text-helper` | 10px | 12px | 选择器说明、上传卡元信息、常量元信息、检查器就绪项与详情 |
| 标签 | `--experiment-text-label` | 12px | 13px | 字段标签、小节标题、参数矩阵标签、检查器区块标签 |
| 字段值 | `--experiment-text-value` | 13px | 14px | 下拉选择值、检查器主值、专家字段输入 |
| 次级控制件 | `--experiment-text-small` | 11px | 12px | 页头 eyebrow、阶段指示、画布元信息与就绪胶囊 |
| 控制件 | `--experiment-text-control` | 12px | 13px | 按钮、来源胶囊、专家页签、目录筛选、搜索框 |
| 代码标记 | `--experiment-text-micro` | 9px | 10px | `WINDOW` / `HORIZON` / `EPOCHS` / `BATCH` / `LR`、`.PY`、`v1`、`TASK #` |
| 序号徽标 | `--experiment-text-tiny` | 9px | 11px | 阶段序号、专家页签计数、就绪检查的 ✓ / ! 记号 |

- **标题不动**：页标题 32px、画布实验名 19px、参数输入数值 19px、检查器主值 14px 保持原值。
- 桌面端还单独上调了：小节标题 14 → 15px、专家面板标题 12 → 14px、专家页签 11 → 13px、
  序列图示帧与 `输入 / 3 步` 说明 10 → 12px、目录实验名 12 → 13px、目录徽标 9 → 12px。
- 9–11px 只保留给上表的**代码标记与序号徽标**；除此之外桌面端说明文字不再低于 12px。
- **次要文字对比度**：检查器区块标签、专家参数说明与字段标签、运行条说明、目录元信息由
  `--text-40` 改用既有的 `--text-60`；目录元信息的行高沿用既有值。没有新增或修改任何配色值。
- 检查器「时序配置」两列在字号上调后由 `1fr 1fr` 改为 `minmax(0, 1.4fr) minmax(0, 1fr)`，
  避免中文标签被压成竖排；运行条高度、sticky 吸附位置与列宽规则都没有改动。

### 验收与截图

验收脚本 [verify-font-hierarchy.mjs](../scripts/audit/experiment-console/verify-font-hierarchy.mjs)
在两个桌面视口采集每个关键节点的 computed 字号 / 颜色，扫描隐藏溢出的文字节点，并输出截图
（`scripts/audit/experiment-console/shots/`，不入库）。本地实测：

| 视口 | 结果 |
| --- | --- |
| 1440×900 | 列宽 `1086px 284px`（展开 `235px 837px 284px`）、横向溢出 0、运行条 63px 贴底、无文字裁切；仅 3 个节点仍是 9–11px（`.PY` 代码标记、`WINDOW` 类短码、页签计数） |
| 1920×1080 | 列宽 `1202px 284px`（展开 `235px 953px 284px`）、横向溢出 0、运行条 63px 贴底、无文字裁切，与 1440 相同的三个小号节点 |
| 1024×768（回归） | 三列 `200px 490px 250px`、无溢出；确认桌面令牌在小桌面宽度下同样不裁切 |
| 390×844（回归） | 单列 334px、运行条 121px、字号与改造前逐项一致，确认移动端未受影响 |

截图覆盖两个桌面视口的首屏、参数区、专家参数、检查器吸附、展开目录、页尾运行条与整页；
前端 `node --test`（649 项）与 `npm run build` 通过。

## 16. 2026-09-28 卡片层级与描边收敛

版式稳定后暴露的第二个问题：任务定义、参数矩阵、专家参数与检查器出现「卡片套卡片」，
连续描边把注意力平均分摊到所有容器上，任务配置区反而没有阅读重点。本轮只改训练页的
`experimentCenter.css`：**分组改用面层明度差，描边只留给真实状态**，不动功能、文案、交互、
配色值、训练链路与移动端设计，也不回退第 15 节的字体令牌。

### 面层令牌

新增三档面层，页面里所有分组背景都从这里取，不再各自硬编码 `rgba(13,29,46,0.x)`：

| 令牌 | 深色 | 浅色 | 用途 |
| --- | --- | --- | --- |
| `--experiment-surface-1` | `rgba(7,19,31,0.34)` | `rgba(255,255,255,0.72)` | 分区级分组块：任务定义两栏、载荷条、参数格、专家分栏、选择器容器、目录行悬停 |
| `--experiment-surface-2` | `rgba(11,24,40,0.5)` | `rgba(255,255,255,0.9)` | 控件与字段：来源胶囊、渠道胶囊、架构选项、目录行、检查器时序格、监控/结果指标卡 |
| `--experiment-surface-3` | `rgba(3,11,19,0.36)` | `rgba(23,33,47,0.045)` | 内嵌面板：专家页签列 |

### 框线清单（改后）

| 元素 | 改前 | 改后 |
| --- | --- | --- |
| `.experiment-choice-block` | 整圈描边 + `data-active` 蓝色强调线（固定值，不代表选择） | 无边；仅 `data-model-block=invalid/pending` 时才有状态描边 |
| `.experiment-dataset-option` | 原生下拉（无边，看起来像标题；后改为一条静态下划线） | 无边 + 选中时左强调线；未选中悬停才出现描边 |
| `.experiment-uploaded-card` | 整圈暖色描边 | 无边 + 左侧 2px 暖色强调线 + 暖色渐变底 |
| `.experiment-uploaded-item` | 整圈描边 + `border-left` | 无边 + 选中时左强调线 |
| `.experiment-architecture-current` | 整圈描边 + 蓝色左边框 | 无边 + 左强调线；悬停才出现描边 |
| `.experiment-architecture-option` | 整圈描边 | 无边；悬停描边、选中描边 + 左强调线 |
| `.experiment-payload-bar` / `.experiment-channel-chip` | 整圈描边 | 无边；渠道胶囊悬停才出现描边 |
| `.experiment-param-cell` | 整圈描边 + 每格蓝色左边框 | 无边；数值输入保留虚线基线，悬停/聚焦加强 |
| `.experiment-expert`（专家分栏外壳） | 整圈描边 | 无边；页签列保留一条竖分隔线 |
| `.experiment-expert-field` | 整圈描边 + 无边界输入 | 无边容器 + 输入框静态下划线 |
| `.experiment-inspector-cell` | 整圈描边 | 无边（只留面层） |
| `.experiment-directory-row` | 整圈描边 + `border-left` | 无边 + `inset` 左强调线（当前实验才显示） |
| `.experiment-expert-action` / `.experiment-config-chip` | 静态描边 | 无边；悬停才出现描边 |
| `.experiment-monitor-metric` / `.experiment-result-metric` / `.experiment-result-note` | 整圈描边 | 无边；结果说明仅在 error/warning 状态描边 |
| `.experiment-directory-selection` | 整圈描边 | 无边 |
| **保留**：阶段指示胶囊、通用按钮与主按钮、上传/替换/删除按钮、架构搜索框、模型格式说明中的错误块、空态虚线框、专家页签选中态、运行条顶部细边线、监控终端日志外框 | 保留 | 这些是控件、错误或滚动视口，边界本身就是信息 |

静态整圈描边容器数（脚本统计，见下）由**首屏 9 → 0**；展开目录后仍为 1（当前实验行的左强调线，
不是整圈）。

同时去掉了两处「装饰性强调」：「任务定义」两栏上方常驻的蓝色左强调线（`data-active` 是组件里的
固定值，不表示任何用户选择），以及 MARS / YOUR MODEL 标记的常驻绿色。绿色 `#63e8bf` 在本页已
统一为「真实就绪」语义（就绪胶囊、就绪检查、可训练状态、运行条 ready），因此这两个分区标记改用
`--text-40`；`data-active` 属性本身保留在 DOM 里，不影响任何测试或交互。其余状态色（冰蓝选择线、
火星橙主操作、红/黄错误与警告）一个都没有改动。

### 验收与截图

新增 [verify-interaction-states.mjs](../scripts/audit/experiment-console/verify-interaction-states.mjs)：
用真实鼠标按下与点击逐个触发 hover / focus / 选中 / invalid，输出该元素的实际描边、背景与
`box-shadow`，并保存 11 张状态截图。本地实测 13/13 项通过：

| 状态 | 结果 |
| --- | --- |
| 参数输入框 | 静态 `bottom rgba(146,182,223,0.42)` 虚线 → 悬停 `--experiment-line-strong` → 聚焦 `rgb(154,217,239)` 实线 |
| 数据集选项 | 两个选项直接列出；未选中行聚焦时 `outline 2px rgb(154,217,239)`，选中行无边 + `inset 2px 0 #f19a78` 左强调线 |
| 模型来源胶囊 | 选中态无边 + `inset 2px 0 #f19a78` 左强调线 |
| 官方架构选择器 | 当前值左强调线；29 个架构选项选中态描边 + 左强调线；选择后当前值正确写回 |
| 实验名 | 聚焦下划线；非法（空/重名）时 `aria-invalid=true` + 红色描边 + 行内说明 |
| 次要按钮 / 目录筛选 | 静态无边，悬停才出现 `rgba(146,182,223,0.18)` 描边 |
| 运行条 | 仍保留顶部细边线，高度 63px，贴底 |

`verify-font-hierarchy.mjs` 同步增加了描边与面层采集（`frames` 字段）与 `--theme` 参数：

| 视口 | 结果 |
| --- | --- |
| 1440×900 | 列宽 `1086px 284px`（展开 `235px 837px 284px`）、横向溢出 0、运行条 63px 贴底、无文字裁切；静态整圈描边 **9 → 0** |
| 1920×1080 | 列宽 `1202px 284px`（展开 `235px 953px 284px`）、横向溢出 0、无文字裁切；静态整圈描边 **9 → 0** |
| 1024×768（回归） | 三列 `200px 490px 250px`、无溢出 |
| 390×844（回归） | 单列 334px、运行条 121px、字号与第 15 节一致，移动端未受影响 |
| 1440×900 浅色主题 | 面层方向相反且渲染正常，无「白底无边」问题 |

前端 `node --test`（649 项）与 `npm run build` 通过。

## 17. 2026-09-28 任务定义区留白与首屏信息密度

第三个问题：「任务定义」两块被拉成相近高度，数据集侧一大片空白占着首屏，后面的载荷条与训练参数
被推到视口外。本轮只调桌面端的高度、对齐与内部间距，**不改数据选择逻辑、训练链路、文案与移动端
设计**，保留前两步的字号令牌与面层/描边规则。

### 空白成因

`.experiment-task-grid` 用的是默认 `align-items: stretch`。两栏高度由更高的「模型来源」决定
（上传模型卡 + 动作行约 219px），而「训练数据集」只有标签 + 下拉 + 一行说明约 104px，
于是被拉伸到 244px——**其中约 150px 是空白**。实测（1440×900，登录 + 已选上传模型）：

| 元素 | 改前 | 改后 |
| --- | --- | --- |
| `.experiment-task-grid` 高度 | 244px | 219px（按内容） |
| 「训练数据集」块高度 | 244px（内容 104px） | 104px |
| 「模型来源」块高度 | 244px | 219px |

### 调整内容

| 位置 | 改前 → 改后 | 目的 |
| --- | --- | --- |
| `.experiment-task-grid` | `align-items: stretch` → `start` | 两块按信息量自然排布，去掉数据集侧拉伸出的空白 |
| `.experiment-canvas-section` | `padding: 20px` → `16px 18px` | 四个分区各回收 8px 纵向留白 |
| `.experiment-section-kicker` | `margin-bottom: 13px` → `10px` | 分区标题更贴近自己的内容 |
| `.experiment-choice-block` | `padding: 13px` → `10px 12px` | 任务定义两块的内部留白 |
| `.experiment-choice-label` | `margin-bottom: 9px` → `8px` | 同上 |
| `.experiment-source-toggle` | `margin: 5px 0 10px` → `4px 0 8px` | 模型来源切换与上传区的间距 |
| `.experiment-source-option` | `min-height: 38px / padding: 8px` → `36px / 5px` | 胶囊本身仍在 36px 以上，可点区域不缩水 |
| `.experiment-uploaded-model` | `gap: 9px` → `8px` | 卡片与动作行之间 |
| `.experiment-uploaded-empty` | `padding: 11px / gap: 4px` → `9px 11px / 3px` | 未选模型时的引导块 |
| `.experiment-center-header` | `margin-bottom: 16px` → `12px` | 页头到阶段指示的间距 |
| `.experiment-center-stage-nav` | `margin-bottom: 18px` → `14px` | 阶段指示到控制台的间距 |

**没有动**：字号令牌、控件最小可点高度（按钮仍 36–44px）、面层与描边规则、列宽、sticky 与
运行条样式、任何文案。

### 首屏效果（1440×900，运行条上沿 837px）

| 内容 | 改前 | 改后 |
| --- | --- | --- |
| 任务定义（01） | 完整可见（底部 655） | 完整可见（底部 611） |
| 载荷条（02 顶部） | 完全被运行条挡住 | 完整可见（666–724） |
| 输入 / 输出序列图示 | 不可见 | 完全可见（740–847） |
| 训练参数（03） | 不可见（起于 919） | 标题与参数格上沿在视口内（起于 864） |
| 页面总高 | 1938px | 1499px |

1920×1080 下更进一步：任务定义与载荷条（含序列图示）完整可见，训练参数标题与参数格露出上沿。
移动端（390×844）与窄桌面（1024×768）因只收紧了纵向留白而同样变紧凑，**字号与节点可见性逐项
一致**，横向溢出仍为 0。

### 验收

`verify-font-hierarchy.mjs` 现在会在首屏采集时明确**收起**「管理上传模型」，并把展开态单独记为
`01b-uploaded-manage`：展开会多出模型列表与格式说明（约 220px），若不收起会把首屏几何量成中间态
（本步排查中曾因此得到错误的高度读数）。新增输出：`groupFirstScreen`（每个分区在运行条上沿以上
有多少可见）、`taskGrid` / `datasetBlock` / `modelBlock` / `payloadBar` / `flowDiagram` /
`paramGrid` 的首屏位置。

| 项目 | 结果 |
| --- | --- |
| 1440×900 | 列宽 `1086px 284px`、横向溢出 0、运行条 63px 贴底、无文字裁切、静态整圈描边 0 |
| 1920×1080 | 列宽 `1202px 284px`、同样无溢出 / 无裁切 / 描边 0 |
| 1024×768（回归） | `704px 250px`，任务定义 666 → 405px |
| 390×844（回归） | 单列 334px，运行条 121px，任务定义 676 → 433px，字号无变化 |
| 交互状态 | `verify-interaction-states.mjs` 13/13 通过（聚焦、选中、invalid、悬停边界不变） |

前端 `node --test`（649 项）与 `npm run build` 通过。

## 18. 2026-09-28 右侧检查器的视觉权重

第四个问题：检查器里 5 个区块之间各画一条同色分隔线（加上头部共 5 条），区块标签与辅助说明又用了
同一个颜色，信息层级被压平；同时「通过」的实心绿底和「未通过」的橙底一样抢眼，正常状态反而比
错误状态更吵。本轮只调桌面端样式，**检查器宽度（284px）、信息内容、校验逻辑、就绪判定与移动端
设计都不变**。

### 调整的视觉层级

| 项 | 改前 | 改后 | 目的 |
| --- | --- | --- | --- |
| 静态分隔线 | 5 条（头部 1 + 区块 4） | **3 条**（头部不画线，最后一个区块不画线） | 少一条线、少一次视觉打断 |
| 区块间距 | `padding-bottom/margin-bottom: 14px` | `12px`；最后一个区块归零 | 节奏更均匀，末尾不留空档 |
| 区块标签 | `--text-60` | `--text-30`（mono 小字，最弱一档） | 标签退到背景，字段值更突出 |
| 字段值 | 14px / `--text` | 不变 | 仍是检查器的视觉焦点 |
| 辅助说明 | 12px / `--text-60` | 不变 | 三级层次：标签 → 值 → 说明 |
| 通过的 ✓ | 实心绿底 + 深色字 | 淡绿底 `rgba(99,232,191,0.14)` + 绿环 + 绿字 | 正常状态克制，不抢焦点 |
| 未通过的 ! | 淡橙底 + 橙边 | **实心橙底 + 深色字** | 需要处理的状态更醒目 |
| 未通过行文字 | 普通字重 | 字重 700（颜色仍 `--text-60`） | 不整行提亮也能一眼看到 |
| 时序列格标签 | `--text-40` | `--text-30` | 与区块标签同级 |
| 「展开详细信息」 | 蓝色链接（与一级动作同色） | `--text-40`，悬停转蓝 | 次要动作不再争焦点 |
| 链接命中区 | 仅有文字本身 | 上下 2px / 左右 8px 内边距，左对齐位置不变 | 更好点，外观位置不漂 |

检查器总高：1440×900 下 **659 → 604px**（区块更紧、少 2 条线），仍无内部滚动；284px 宽度不变。

### 全部就绪 / 存在错误两种状态

新增 [verify-inspector-states.mjs](../scripts/audit/experiment-console/verify-inspector-states.mjs)：
用真实登录态与真实上传模型分别构造两种状态，逐项断言并截图（`shots/<label>/<视口>-A-blocked.png`
与 `-B-ready.png`）。本地实测 **16/16 项通过**：

| 状态 | 结果 |
| --- | --- |
| 全部就绪（登录 + 有效上传模型 + 唯一实验名） | 胶囊 `ready`「可以开始训练」（绿）；4 行全 ✓；✓ 为淡底绿环 `rgba(99,232,191,0.14)` + `1px rgba(99,232,191,0.45)`；0 个未通过行；分隔线 3 条 |
| 存在错误 / 待处理（未登录） | 胶囊 `guest`「还不能开始训练」；4 行 `!` 为实心橙底 `rgb(241,154,120)`；胶囊 `blocked` 时为 `rgb(241,154,120)` |
| 两种状态共同 | 面板 284px、无内部滚动（1440/1920）、无文字裁切、横向溢出 0、字号不变（标签 12 / 值 14 / 说明 12） |

### 回归

| 视口 | 结果 |
| --- | --- |
| 1440×900 / 1920×1080 | 列宽与运行条不变；检查器 284px、无溢出、无裁切；静态整圈描边仍为 0（新增的两处是 `.experiment-source-option` 与就绪记号的 1px 环，属于控件与状态反馈） |
| 390×844 | 分隔线仍为 5 条、间距 14px（媒体查询只覆盖 ≥ 901px），字号与节点可见性逐项不变 |
| 1024×768 | 分隔线 3 条；面板内容 620px 超出上限 603px 约 19px，`overflow-y: auto` 可滚动。该上限由「视口高 − 86px − 实测运行条高度」得出，窄桌面运行条换行变高后本来就紧：改前内容 659px 超出约 **35px**，本轮把溢出减到 19px，属于改善而非回归 |
| 交互状态 | `verify-interaction-states.mjs` 13/13 通过 |

前端 `node --test`（649 项）与 `npm run build` 通过。

## 19. 2026-09-28 实验目录的空状态与状态区分

第五个问题：目录没有实验时，状态筛选栏、标签筛选仍完整占位，配合 `sticky` 面板撑满侧栏，
左侧是一大片空白，「暂无内容」和可执行操作都被淹没。顺带暴露一个真实缺陷：目录**只有
「有实验 / 没有实验」两种展示**，取任务列表失败时会误报「暂无实验」。

### 三种状态

目录根元素现在带 `data-directory-state`，空态块带 `data-empty-reason`：

| 状态 | 触发条件 | 展示 |
| --- | --- | --- |
| `loading` | `tasksLoading`（首次取任务列表尚未返回） | 「正在加载实验…」，`role="status"`，不显示「新建实验」 |
| `error` | `tasksError`（取任务列表失败） | 「实验目录加载失败。」红色 + **重试**按钮，`role="alert"`，不再误报「暂无实验」 |
| `empty` | 请求完成且确实 0 个实验 | 「暂无实验。新建实验后，实验会出现在这里。」+ **新建实验**主按钮 |
| `ready` | 有实验 | 完全维持原有筛选、搜索、批量标签与行的扫描效率 |

判定只用真实的加载/失败标记（`TrainingContext` 新增 `tasksLoading` / `tasksError`，在
`loadTasks` 里置位与复位），**不参与任何查询、筛选或选中逻辑**。中途曾用 `tagState.scope`
推断加载态，导致登录用户永远停在「加载中」，已改为只看加载标记。

### 密度调整（仅 ≥ 901px）

| 项 | 改前 | 改后 |
| --- | --- | --- |
| 目录面板高度（空） | 727px（sticky 面板撑满侧栏） | **398px**（只占内容高度，`align-self: start`） |
| 状态筛选栏 | 空态也完整显示 4 个胶囊 | `empty` / `loading` 时**隐藏**（DOM 仍保留 4 个按钮；有内容时不变） |
| 搜索框 / 管理标签 | 36px 高、带底色 | 32px、透明底（保留可用，退回次要） |
| 空态块 | 内联 `padding: 24px` | `.experiment-directory-empty`（`24px 18px`，桌面 `20px 14px`），文字 12px、居中、最长 30ch |
| 空态动作 | 无 | 一个主按钮「新建实验」，复用页头同一个 `handleCreateExperiment` |

**没有改动**：搜索 / 标签筛选 / 状态筛选的判定与数据流、批量标签、行的信息与交互、
移动端（≤ 900px）的设计（那里目录本来就不是 sticky，空态只换了类名与文案来源）。

### 验收与截图

新增 [verify-directory-empty.mjs](../scripts/audit/experiment-console/verify-directory-empty.mjs)：
每个场景用**独立页面会话**（避免注入的 fetch 包装残留），按账号里有没有任务自适应断言哪一组。
`error` 态通过 `Page.addScriptToEvaluateOnNewDocument` 只让 `GET /api/training/tasks` 返回 500，
不改仓库代码、不改后端。

| 场景 | 1440×900 / 1920×1080 结果 |
| --- | --- |
| 空目录（登录 + 0 个实验） | 状态 `empty`、面板 398px、筛选栏隐藏且 DOM 仍有 4 个按钮、空态给出「新建实验」并实测可点击、搜索与标签管理仍可用、无裁切、横向溢出 0 |
| 加载失败（注入 500） | 状态 `error`、文案「实验目录加载失败。」、`role="alert"`、有重试按钮、**不出现「暂无实验」** |
| 有内容（2 个实验） | 状态 `ready`、面板 727px、筛选栏 `flex`、2 行渲染、列表 `flex: 1 1 auto` 可滚动、无裁切与溢出 |
| 筛选无匹配 | `no-match` + 「没有匹配的实验，请调整筛选条件。」，筛选栏保留、不显示「新建实验」 |

三组各自 19/19、16/16 项通过（同一脚本按账号切换断言组）。回归：交互状态 13/13、
390×844 与 1024×768 无溢出无裁切，前端 `node --test`（649 项）与 `npm run build` 通过。

## 20. 2026-09-28 桌面端视觉收口（底部运行条与跨区域一致性）

最后一轮做桌面端验收与细节收口。先量后改：用一次性探针采集跨区域左/右边界、运行条内部几何与
纵向节奏，再只修截图中能明确指出、且度量支持的项。

### 度量结论（1440×900 / 1920×1080 均相同）

| 检查项 | 结果 |
| --- | --- |
| 运行条与工作区对齐 | 运行条主行左/右边界与 `.experiment-center`（1500px 上限、`max(28px, 50vw − 750px)` 内边距）**完全一致**：1440 下同为 28 / 1412，1920 下同为 210 / 1710，左右偏差均为 **0**（此前怀疑的 17px 偏差不存在） |
| 外层容器内边距一致 | `.experiment-canvas-head` 18/20、`.experiment-canvas-section` 16/18、`.experiment-directory-head` 14/15、`.experiment-inspector-head` 14/16、`.experiment-inspector-body` 12/16 —— 差异 ≤4px，属于同级容器，保持不动 |
| 字号层级 | 运行条状态 13px/800、摘要 12px mono、主按钮 14px/800；与页头、分区、检查器同属第 15 节的令牌体系 |
| 文字裁切 / 区域重叠 | 三种状态、两个视口全部为 0 |
| 运行条可见性 | `position: fixed`，底边始终等于视口高（900 / 1080），滚到页尾仍贴底；底部预留 40px + 实测条高 |

### 改动清单（只改运行条，两处）

| 问题 | 改前 | 改后 |
| --- | --- | --- |
| **未就绪时主按钮与就绪态同样醒目**：禁用态是「橙底 + `opacity: .5`」，是就绪态淡一点的版本；未登录时按钮反而是**实心橙**，比「配置就绪」更抢眼，与检查器的橙色错误行一起让页面出现三处橙色 | `.experiment-run-bar-start:disabled { opacity: .5 }`（仍保留橙底） | 实心火星橙只留给**真的可以开始实验**：禁用时中性底 + `--text-40` + 细描边（观感变暗，不再像可点）；未登录时中性底 + 橙色文字 + 橙色细描边（仍可点、点了弹登录，但让位于真正的操作）。运行条根元素新增 `data-run-state` 供样式区分，状态判定仍只用既有的 `canStart` / `user` |
| **状态与摘要黏连**：左侧状态句（如「配置未完成，先处理检查器里的问题」）与后面的 `数据集 · 模型 · 输入 · 轮数` 只靠 14px 间距分隔，加粗状态句紧邻 mono 摘要，读起来像一整句 | 无分隔 | `.experiment-run-bar-meta` 增加左侧竖细线（`border-left: 1px solid var(--experiment-line)`）与 14px 内边距，两块各自成组 |

**没有改动**：文案、就绪判定、`startDisabled` / `aria-disabled` 语义、按钮尺寸（44px 高、150px 宽，仍是最大控件）、
页头 / 任务定义 / 参数区 / 目录 / 检查器（前三步与第四、五步的成果逐项保留）。

### 三种状态 × 两个视口

| 状态 | 1440×900 | 1920×1080 |
| --- | --- | --- |
| 未配置（登录 + 空名称） | 状态句「配置未完成，先处理检查器里的问题」橙色；按钮中性底禁用；运行条 64px 贴底 | 同左，运行条 64px |
| 配置就绪（唯一实验名） | 状态句「配置就绪，可以开始实验」绿色；按钮 `rgb(241,154,120)` 实心橙、这是页面唯一实心橙按钮；运行条 63px | 同左，运行条 63px |
| 校验错误 / 未登录 | 状态句「登录后即可开始实验」；按钮中性底 + 橙字，仍可点；检查器 4 行 `!` 实心橙底；运行条 64px | 同左 |

三态均：无文字裁切、无区域重叠、横向溢出 0、运行条底边等于视口高。

### 验收

| 项目 | 结果 |
| --- | --- |
| `verify-font-hierarchy.mjs`（新增 `--ready`：把实验名填成唯一值后采集） | 未配置 / 就绪 / 访客三态 × 两个视口，裁切 0、溢出 0、运行条 63–64px 贴底 |
| `verify-interaction-states.mjs` | 13/13 通过（含运行条顶部细边线） |
| `verify-directory-empty.mjs` | 19/19 通过（空目录、加载失败注入、新建动作可点） |
| 移动端 / 窄桌面回归 | 390×844 运行条 121px、1024×768 运行条 64px，均无裁切无溢出 |
| 前端 | `node --test` 649 项通过、`npm run build` 通过、`git diff --check` 干净 |

`verify-interaction-states.mjs` 的官方架构断言在本轮加入「等节点出现再加断言」的等待，
避免模型列表慢加载时把 `null` 当成失败（属验收脚本健壮性，与应用行为无关）。

## 21. 2026-09-28 新实验配置视图的信息减法

范围只限 `#/training` 的「新实验配置」视图（不含训练监控、结果页，也不做手机 / 平板设计）。
原则：**用户先看到要选择或填写的内容，以及阻止训练的真实问题**；删掉装饰性标签、重复说明与
重复摘要后同步收紧高度，不留空容器。视觉语言与配色未改。

### 删除的元素

| 分类 | 删除内容 | 位置 |
| --- | --- | --- |
| 重复说明 | 「你的 PyTorch 文件 / 平台模型库」来源副标题 | `.experiment-source-option small` |
| 重复说明 | 分区提示「先决定实验要回答什么」「O₃ 为基础输入，可叠加气象驱动变量」「直接输入精确数值」「结构、策略与微调，按需展开」 | `.experiment-section-hint` ×4 |
| 重复说明 | 默认数据集下的复述「使用当前默认训练数据，OpenMARS 臭氧目标与 MCD 驱动变量融合」 | `.experiment-choice-meta` 条件渲染 |
| 重复说明 | 「自定义参数按上传模型声明的 schema 动态渲染，修改后立即用于本次实验」 | `.experiment-expert-note` |
| 重复图示 | 整块「输入时间步 → 上传模型 → 预测时间步」序列示意图（含帧方块、连接线、`INPUT / N STEPS`） | `.experiment-flow*` 全部规则与 `frameCount()` |
| 装饰 | 分区序号 `01–04` | `.experiment-section-index` |
| 装饰 | 角标 `MARS` / `YOUR MODEL` / `OFFICIAL` | `.experiment-choice-label em` |
| 装饰 | 参数标签旁的 `WINDOW` / `HORIZON` / `EPOCHS` / `BATCH` / `LR` 英文小码 | `.experiment-param-label code` |
| 装饰 | 上传卡片的 `.PY` 徽标 | `.experiment-uploaded-icon` |
| 统计 | 上传卡「N 个可训练」，检查器自定义参数数量、`validCount` 变量 | `uploadedModelValidCount`、`customParamCount` |
| 重复状态 | 画布头部副标题「火星臭氧时空预测 · 新实验配置」与就绪胶囊（就绪由运行条与检查器播报） | `.experiment-canvas-meta`、`.experiment-canvas-state` |
| 重复摘要 | 运行条的 `数据集 · 模型 · 输入变量 · N epochs` 长摘要 | `.experiment-run-bar-meta` |
| 重复摘要 | 检查器的「输入变量」「时序配置」「数据集」常态区块与「展开详细信息」折叠区 | `.experiment-inspector-telemetry`、`.experiment-inspector-cell`、`.experiment-inspector-details` |
| 重复播报 | 逐条「已通过」的就绪行（改为：正常时一行结论，有问题时只列原因） | `readinessRows` |

清理同步完成：删除失效中英文文案 **32 个**（`flow*`、`section*Hint`、`inspectorSequence`、
`runBarSummaryEmpty` 等）、删除失效 CSS 规则（含桌面断点里的对应覆盖），并把名称输入的
`aria-describedby="experiment-name-hint"` 改为 `aria-errormessage` 指向真正的错误节点
（提示节点已删，避免悬挂引用）；字段级错误、`role="alert"` 与状态播报全部保留。

### 保留的元素（含「因承载独有信息而保留」的文案）

- **数据集第二项说明已删除**：`datasetHintMcdOverview`「使用 data/MCD_Output_global_10m_ls_lst 中的原始 3 小时 MCD 数据，覆盖 MY24-MY35。」曾因“只有该项才有的数据来源与覆盖年份”保留；第 23 节把它挂在选项行内常显后，用户判定列表里不需要小字说明，文案与 `.experiment-dataset-option small` 规则一并删除。数据集选项现在只有名称，默认融合数据集同样没有复述文案。
- 控件名称、当前值、选中 / 禁用 / 加载状态、错误与警告：数据集选项、模型来源切换、上传 / 替换（现名「上传模型」）/ 编辑参数 / 管理模型 / 下载说明与模板、渠道胶囊与计数、参数矩阵五个字段及**量纲**（时间步 / 预测步 / epochs / 样本 / 学习率）、专家页签与全部字段、标签选择器。
- 上传卡片：文件名、版本、自定义参数数量、校验状态（可训练 / 待校验 / 不可用）。
- 检查器：就绪胶囊、当前模型身份（文件名或架构 + 来源 · 版本 · 校验）、**需要处理的阻塞原因**，以及可定位入口（去登录 / 填写名称 / 编辑参数 / 选择来源）；正常时一行「配置检查通过」。
- 运行条：明确的就绪 / 受阻 / 未登录状态、提交中反馈、唯一的「开始实验」主按钮。
- 页头：可编辑实验名称与字段级错误提示。
- 实现说明性质的文案（如 `customParamsPrimaryHint`）按范围删除；**字段级说明**（取值范围 `范围 1 - 128`、`范围 0 - 1`、字段错误）保留。

### 未改动

模型来源切换、数据集选择、上传与校验、参数编辑、就绪判定与开始训练的逻辑全部未动；
`readiness.blockers` 仍是唯一就绪来源，`startDisabled` 仍只决定按钮可点性。
`code:` 字段仍留在参数矩阵数据里（供后续复用），只是不再渲染成标签旁的小码。

### 验收

| 状态 | 1440×900 | 1920×1080 |
| --- | --- | --- |
| 已上传模型 · 未就绪 | 首屏露出任务定义 / 输入与预测 / 训练参数；检查器 251px、1 条待处理；页高 649px | 训练参数与专家参数标题进入首屏；页高 441px |
| 配置就绪 | 检查器一行「配置检查通过」；运行条绿色状态 + 实心橙「开始实验」 | 同左，页高 440px |
| 官方模型 | 架构选择器正常；检查器 1 条待处理 | 同左 |
| 校验错误（未登录） | 检查器 3 条待处理：登录[去登录] / 名称[填写名称] / 模型；运行条「登录后即可开始实验」 | 同左 |

四种状态 × 两个视口：**无文字裁切、无重叠、横向溢出 0、底部运行条始终贴底**；脚本确认
16 个已删除节点全部不在页面上（`state: 'removed'`）。任务定义两块实测各只有 11px 内边距余量，
没有留下空容器。

| 检查 | 结果 |
| --- | --- |
| `verify-font-hierarchy.mjs` | 4 状态 × 2 视口通过；新增「已删除节点确认」「检查器结论 / 问题清单」读数 |
| `verify-inspector-states.mjs` | 18/18（正常态一行结论、问题态逐条可定位、宽度 284px、无裁切） |
| `verify-interaction-states.mjs` | 13/13（聚焦、选中、invalid、悬停边界不变） |
| `verify-directory-empty.mjs` | 19/19（空目录、加载失败注入、新建动作可点） |
| 移动端 / 窄桌面 | 390×844 运行条 121 → **102px**（摘要删除后的自然收紧）、1024×768 运行条 64px，均无裁切无溢出 |
| 前端 | `node --test` 649 项通过、`npm run build` 通过、`git diff --check` 干净 |

被删除节点相关的断言按约定只调整了那部分（画布头部顺序、序列示意图、运行条摘要、检查器常态
摘要与折叠区、参数短码）；业务行为断言（就绪判定、提交路径、来源切换、上传与参数表单、
检查器不复制表单）全部保留。

### 顺手修正的两处焦点问题

- 已选中模型时，上传区的主按钮是「替换文件」——与运行条唯一的实心橙主按钮同级。现在
  `.experiment-uploaded-model[data-has-selection='true'] .experiment-uploaded-upload` 改为描边，
  实心暖底只在未选模型（那一屏的动作就是上传）时出现。
- 运行条删除长摘要后，`.experiment-run-bar-info` 的 14px 间距与 `.experiment-run-bar-meta` 规则
  一并清除，避免留下无用的空槽。

## 22. 2026-09-28 上传卡片去掉重复的参数编辑入口

任务定义里上传模型的紧凑卡片原先在「替换文件」旁再放一个「编辑自定义参数」，与右侧检查器的
同名入口重复。现在只删掉卡片上的这一个按钮：

- `UploadedModelPanel` 不再渲染 `data-uploaded-model-params` 按钮，`onEditParams` 属性与画布传入的
  `labels.editParams`（`copy.editCustomParams`）一并移除；动作行剩「替换文件」「管理上传模型」「下载说明」「下载模板」。
- 右侧检查器 `data-inspector-action="edit-custom-params"` 保持不变，仍是切到「自定义模型参数」页签并聚焦的唯一定位入口。
- 上传、替换、校验、管理、模板与说明下载、`DynamicModelParamsForm` 动态表单本身全部未改；中英文文案键 `editCustomParams` 继续由检查器使用。

## 23. 2026-09-28 训练数据集改为直接列出选项

训练数据集只有两项，原生下拉要多一次点击、展开前也看不到另一项。现在改成列表式单选：

- 控制器里用 `datasetOptions` 组好 `{ value, label }`，任务定义区用 `role="radiogroup"` +
  两个 `role="radio"` 的 `.experiment-dataset-option` 按钮渲染，`aria-checked` 表示选中；
  点击仍走原来的 `onTrainingDatasetChange`，取值、校验与提交路径未变。
- 选项行只显示数据集名称：`datasetHintMcdOverview`（原始 3 小时数据、覆盖 MY24-MY35）先随列表改为常显，
  随后按用户要求删除，文案键与 `.experiment-dataset-option small` 规则一并移除。两项都没有小字说明。
- 视觉沿用模型来源胶囊的选中语言（无边 + `inset 2px 0 #f19a78` 左强调线），未选中行悬停出现描边，
  键盘聚焦进入 `:focus-visible` 的 `outline 2px #9ad9ef`。
- CSS 里 `.experiment-choice-select`（含 `option`）与 `.experiment-choice-meta` 规则删除，替换为
  `.experiment-dataset-list` / `.experiment-dataset-option`；`experimentConfigStructure`、
  `modelTrainingDataSourcePanel` 两个测试文件的断言与 `verify-font-hierarchy.mjs` /
  `verify-interaction-states.mjs` 的选择器同步更新。
- 验证范围：前端 `node --test` 649 项通过、`npm run build` 通过；本轮未运行浏览器审计脚本，
  也未启动训练。

## 24. 2026-09-28 画布第 04 分区更名「超参数」

用户要求把「专家参数」改名为「超参数」。只改对外文案，不动结构与代码标识：

- 画布分区标题与页签列表的 `aria-label` 用的 `copy.sectionExpert` 由 `专家参数 / Expert parameters`
  改为 `超参数 / Hyperparameters`；副本对象里未渲染的 `groupExpert` 同步改名。
- `expertTab*`、`groupExpertHint`、`.experiment-expert-*` 类名、`data-expert-panel`、控制器状态
  （`expertTab` / `onExpertTabChange`）等内部标识全部保留，避免一次改名牵动测试与样式。
- 本文第 3 节分区表、超参数页签小节、第 9 节能力对照表与 README 的分区描述已改为新名称；
  第 14–23 节属于历史轮次记录，正文里的「专家参数」保留当时的叫法，均指现在的「超参数」。
- `verify-console.mjs` 中该项检查的显示名同步改为「超参数」。
- 验证范围：前端 `node --test` 649 项通过、`npm run build` 通过；本轮未运行浏览器审计脚本。

## 25. 2026-09-28 输入与预测、训练参数并入超参数模块

用户要求把画布上的「输入与预测」和「训练参数」也放进超参数模块，并选定**作为它的两个页签**：

- 页签集合改为 `CONFIG_EXPERT_TABS = ['payload', 'training']` 打头，再拼各自来源的专家页签：
  上传模型为 输入与预测 / 训练参数 / 自定义参数 / 训练策略 / 迁移学习 / 实验标签，
  官方模型把「自定义参数」换成「模型结构」。原 02、03 两个独立分区卡片删除，画布只剩
  01 任务定义与 02 超参数。
- 默认页签改为第一页 `payload`（输入与预测）：新建实验与切换模型来源都落在这一页，
  画布打开时仍能看到载荷胶囊，而不是空的超参数容器。检查器的「编辑自定义参数」仍可把页签切到
  `customParams`。
- 载荷条与参数矩阵移入 `.experiment-expert-content`：面板底色是 `--experiment-surface-1`，
  这两块改用 `--experiment-surface-2` 才看得出分组；参数矩阵由固定五列改为
  `repeat(auto-fit, minmax(150px, 1fr))`，因为内容列要减去 148px 的页签列，不再等于整张画布的宽度。
- 页签只有激活页在 DOM 里（沿用原有条件渲染），因此依赖这些节点的审计脚本改为先切页签再读：
  `verify-console.mjs`（参数矩阵列数、`epochs` 输入、页签清单）、`verify-font-hierarchy.mjs`
  （`activateExpertTab` 后采集参数区与专家区）、`verify-interaction-states.mjs`（参数输入框三态）、
  `verify-stages.mjs`（复制配置回填时逐个页签读取）。
- 测试：`experimentConfigStructure.test.js` 的分区顺序、参数矩阵切片、页签常量与锁定量切片断言改为
  新的页签结构；其余 645 项未改动。前端 `node --test` 649 项通过、`npm run build` 通过。
- 验证范围：只跑了前端单元测试与生产构建，没有启动后端、浏览器审计脚本或训练。

## 26. 2026-09-28 六个页签统一成同一套字段块

上一轮把六个页签排到一起后暴露下一个问题：每个页签的展示形式都不相同。载荷条是带底色的胶囊排、
参数矩阵是五列大数字格、专家字段是三列小块、「自定义模型参数」表单是单列卡片、标签选择器自带
`fieldset` 描边——根因是各有一套度量。现在统一为一套字段块：

- **网格**：`.experiment-payload-bar` / `.experiment-param-grid` / `.experiment-expert-fields` /
  `.experiment-expert-stack` 共用 `repeat(3, minmax(0, 1fr))` 与 9px 间距；窄视口在既有断点统一回落两列 / 单列。
- **块**：`.experiment-payload-lock` / `.experiment-channel-chip` / `.experiment-param-cell` /
  `.experiment-expert-field` 共用 surface-2 面层、`9px 10px` 内边距、9px 圆角、6px 行距与同一条标签行字号。
- **页签正文骨架**：输入与预测、训练参数补上 `experiment-expert-heading`，六个页签都是
  「标题（+ 可选说明）+ 字段块网格」。
- 载荷胶囊改成「变量名 + 短码」的块，选中态仍是暖色左强调线；O₃ 基础输入是同一块的静态版本，
  计数移到网格末行右对齐。
- 参数数值保留 18px 等宽（它们是这一屏的主要数字），但基线、悬停与聚焦反馈改为与专家字段完全一致
  （原先用虚线基线）。
- 「自定义模型参数」表单改用共用块与三列网格，只有类名没有样式的 `model-training-field-grid` 删除；
  表单不再自带标题，由面板的 `文件 / 自定义模型参数` 一行承担（新增 `hideTitle` 属性）。
- 迁移学习的启用开关（`.experiment-toggle-line`）与实验标签的 `TagPicker` 一并收进字段块，
  标签选择器自己的 `fieldset` 描边在块内去掉，图例改成同一条标签行。
- 验证范围：前端 `node --test` 650 项通过（新增 1 项「六个页签同一套字段块」结构断言）、
  `npm run build` 通过；未运行浏览器审计脚本，也没有启动训练。

### 实验标签页签的返工（同轮）

第一版把 `TagPicker` 直接塞进 `.experiment-expert-field` 块，随后被字段块的后代规则反噬：
`.experiment-expert-field input` 与 `.experiment-expert-field > span` 命中了选择器内部的搜索框与标签项，
搜索框丢掉自己的盒式样式、标签名被压成一列一个字（一行一个汉字，块里大片空白）。

- 标签页签改用独立类 `.experiment-tag-block`：只共用外层块度量（面层、`9px 10px` 内边距、9px 圆角），
  不再继承字段块的后代规则；测试新增 `assert.doesNotMatch(css, /\.experiment-expert-field \.training-tag-/)`
  守住这条边界。
- 标签选项改为 `repeat(auto-fill, minmax(150px, 1fr))` 的可点块（surface-1 面层 + 悬停描边），
  长标签名在列内正常换行；`fieldset` 保持普通块级布局（grid 化后百分比宽度会算成半宽），间距用相邻兄弟外边距补。
- 搜索框在块内铺满：删除旧的 `.experiment-expert-content .training-tag-input { max-width: 420px }` 上限
  （它会让搜索框只占整行块的一半）。
- 几何实测（无头 Edge + CDP，1440×900）：标签块 864px、搜索框 842px、标签列 160px、长标签名两行换行、
  无横向溢出；载荷 / 训练参数 / 自定义参数 / 迁移学习 / 实验标签五个页签各截一张图核对（探针脚本为临时文件，未入库）。

## 27. 2026-09-28 上传卡片按钮更名

用户要求把任务定义里上传模型卡片的两个按钮改名，只动文案：

- `copy.uploadedModelReplace`：`替换文件 / Replace file` → `上传模型 / Upload model`。该键同时被卡片主按钮与
  「管理模型」展开区里的取文件按钮使用（两处都是同一个上传入口），因此两处一起改名。
- `copy.uploadedModelManage`：`管理上传模型 / Manage uploaded models` → `管理模型 / Manage models`，
  只影响展开区的开关按钮；展开区的 id（`experiment-uploaded-manage`）与 `aria-expanded` 行为未变。
- 空态的 `copy.uploadModel`（`上传 .py`）、重新校验、删除、下载说明与模板等文案未动。
- `verify-font-hierarchy.mjs` 里按按钮文字定位展开区的正则同步改为 `/管理模型|Manage models/`。
- 本文第 4 节、上传模型主入口小节与 README 的当前状态描述已改用新名称；第 21、22 节属于历史轮次记录，
  正文里的「替换文件 / 管理上传模型」保留当时的叫法。
- 验证范围：前端 `node --test` 650 项通过、`npm run build` 通过；未运行浏览器审计脚本。

## 28. 2026-09-28 实验目录改为默认展开

用户要求训练页默认显示实验目录。原先只有监控 / 结果阶段默认展开，配置阶段收起（两列）。现在三个阶段统一默认展开：

- `ExperimentCenterShell` 改为 `useState(false)`；删掉按阶段覆盖默认值的 `useEffect` 与
  `directoryTouchedRef`——默认值不再随阶段变化，用户手动收起后本次会话保持收起，行为链路更短。
- 网格状态不变：默认 `[data-directory='open'][data-inspector='present']` = `235px minmax(0,1fr) 284px`；
  手动收起回到 `minmax(0,1fr) 284px`；监控 / 结果收起后是单列。
- `experimentCenterShellStructure.test.js` 的默认状态断言改为「默认展开 + 无 `directoryTouchedRef`」。
- 审计脚本同步：`verify-console.mjs` 的默认检查改为「默认三列（目录 + 画布 + 检查器）」，并把原来的
  「点击展开 → 三列」步骤改成「点击收起 → 两列 → 再展开 → 三列」；`verify-font-hierarchy.mjs` 首屏
  说明与目录步骤改为「默认展开 / 点击收起」；`verify-interaction-states.mjs` 的目录悬停改为按需打开
  （`verify-directory-empty.mjs` 原本就只在按钮写着「显示目录」时才点击，无需改动）。
- 本文第 1、2 节的状态表、第 9 节的硬规则与 README 的版式描述已改为默认展开；第 13–27 节属于历史轮次记录。
- 验证范围：前端 `node --test` 650 项通过、`npm run build` 通过；未运行浏览器审计脚本。

## 29. 2026-09-28 画布拆成四个部分

用户要求画布「明显分成四个部分」：1. 模型名称、2. 数据集、3. 模型、4. 超参数。原先的 01 任务定义把数据集与模型来源并排放在一个分区里，名称只是画布头部。

- 原 01 任务定义拆成两个分区：`data-config-group="dataset"`（数据集）与 `data-config-group="model"`（模型），各自带标题与 `.experiment-choice-block` 纸面；`.experiment-task-grid` 两列网格及其媒体查询删除。
- 画布头部升级为 01 模型名称分区：`<header className="experiment-canvas-head" data-config-group="name">`，内部结构不变（同一输入框、同一 `aria-errormessage`），排到「数据集」之前。
- 四个分区的标题行都加了 `.experiment-section-index`（`01–04`，等宽 12px、蓝色），边界一眼可辨；第 21 节删掉的序号装饰在这里按用户要求恢复，只保留序号本身，没有恢复角标与英文小码。
- 数据集独占分区后两个选项横排：`.experiment-dataset-list` 用 `repeat(auto-fit, minmax(220px, 1fr))`，窄视口自动回落单列。
- 文案键：`sectionTask`（任务定义）删除，新增 `sectionName` / `sectionDataset` / `sectionModel`；`sectionPayload` / `sectionTraining` / `sectionExpert` 继续用于超参数页签。
- 测试与脚本同步：`experimentConfigStructure.test.js` 的分区顺序、名称头部、数据集 / 模型分区断言重写；
  `modelTrainingDataSourcePanel.test.js` 的分组切片改为 `dataset → model`；`verify-console.mjs` 的「并排两列」检查
  改为「四个分区顺序 = name,dataset,model,expert」，`verify-font-hierarchy.mjs` 的分区遍历、稳定性快照与探针键同步为
  `name / dataset / model / expert`，并把 `section.index` 从 REMOVED 改回 `.experiment-section-index`。
- 本文第 3 节分区表、第 9 节能力对照表、第 152 / 230 行的入口描述与 README 的版式描述已改写；第 13–28 节属历史轮次记录。
- 01 模型名称原先仍带画布头部的蓝色渐变底（`linear-gradient(90deg, rgba(121,187,255,0.07), transparent 70%)`）与 18/20 内边距，
  与其它三个分区不同底色；当时改为 `background: transparent` + `16px 18px` 拉平。第 30 节改成四张分区卡后，
  这一段由卡片的面层与内边距统一接管。
- 验证范围：前端 `node --test` 650 项通过、`npm run build` 通过；另外用无头 Edge 实测 1440×1200，四段高度
  116 / 156 / 278 / 326px、序号与标题正确、无横向溢出，四段计算样式均为 `background: none / rgba(0,0,0,0)`、
  `padding: 16px 18px`（探针脚本为临时文件，未入库）。

## 30. 2026-09-28 四个分区改成独立卡片（+ 左侧编号轨道）

第 29 节把画布拆成四段后，四段共用同一底色、只有 1px 细线分隔，「区分得不够明显」；中间试过一版
通栏标题带，太重、也被否掉。最终方案是用户确认的 **A + B + D**：

- **A 留白**：四段之间不再共用一条边，改成 `.experiment-center-stack { gap: 14px }`，卡片之间露出画布底色。
- **B 左侧编号轨道**：分区改成两列网格 `var(--experiment-rail-gutter) minmax(0, 1fr)`（38px + 14px 间距），
  `01–04` 放大到 16px 等宽、居中放在第 1 列，标题与全部内容统一在第 2 列；轨道右侧用 `::before` 画一条
  从标题上沿到底部留白的竖细线。标题行 `.experiment-section-kicker` 改 `display: contents`，让序号与标题
  直接落进网格，不需要包一层新容器。
- **D 卡片**：`.experiment-canvas-head` 与 `.experiment-canvas-section` 统一为 `--experiment-surface-1` 卡片 +
  `1px solid var(--experiment-line)` 细描边 + 13px 圆角 + 18px 内边距。这是用户明确允许的「卡片套卡片」，
  因此同时把分区内多余的一层面层去掉：`.experiment-choice-block` 与 `.experiment-expert` 改为
  `background: transparent`，避免卡片里再套一张同色卡片（校验失败 / 待校验的状态描边仍然保留在分组块上）。
- 段间渐隐分隔线随卡片一起删除；`.experiment-canvas-section:last-of-type` 的收尾规则也不再需要。
- 窄视口（≤900px）：内边距 14px、列间距 10px、轨道线位置与序号字号同步收小。
- 测试：`experimentConfigStructure.test.js` 里「01 与其余分区同底色」的断言改为「同一套卡片（13px 圆角 +
  surface-1）」，其余 649 项不变，前端共 650 项通过。
- 验证范围：`node --test` 650 项通过、`npm run build` 通过；无头 Edge 实测 1440×1300：四张卡高度
  119 / 143 / 265 / 334px、圆角 13px、面层 `rgba(7,19,31,0.34)`、网格 `38px 739px`、段间距 14px、无横向溢出
  （探针脚本为临时文件，未入库）。

## 31. 2026-09-28 上传卡片展开区删掉两行

用户指出「管理模型」展开区里的「文件格式要求」与底部「校验状态：可训练」两行不要：

- `UploadedModelPanel` 删除 `.experiment-uploaded-format` 折叠块（`formatOpen` 状态、`formatItems` 变量一起删）
  与 `.experiment-uploaded-foot` 那一行；展开区剩下模型列表、重新校验 / 上传模型 / 删除三个动作与校验详情
  （`ValidationMessages`：通过时一行「可以训练」，失败时逐条列出原因）。
- 随之清理：`copy.uploadedModelFormatTitle` / `uploadedModelFormatItems` 两个文案键删除，
  画布传入的 `formatTitle` / `formatItems` / `validationLabel` 标签与 `fieldLabelStyle` 属性移除，
  CSS 里 `.experiment-uploaded-format*`、`.experiment-uploaded-foot` 规则（含桌面字号覆盖）删除。
  卡片上的状态胶囊与 `labels.ready` 校验结论保留，所以状态信息没有丢。
- 测试：`experimentConfigStructure.test.js` 里该展开区的断言改为 `assert.doesNotMatch(panelSource, /formatItems|experiment-uploaded-format|experiment-uploaded-foot|validationLabel/)`。
- 验证范围：`node --test` 650 项通过、`npm run build` 通过；未运行浏览器审计脚本。

## 32. 2026-09-28 数据集 / 模型卡片去掉卡内小标签

用户指出「训练数据集」「模型来源」两个卡内小标签也不需要——卡片标题已经分别写了「数据集」「模型」：

- `ExperimentConfigWorkspace` 删除 02 数据集卡与 03 模型卡里的 `.experiment-choice-label` 两处；
  分组语义仍在无障碍标签上：数据集列表保留 `role="radiogroup" aria-label={copy.trainingDataset}`，
  来源切换保留 `role="group" aria-label={copy.modelSource}`，读屏仍然能读出分组名。
- `.experiment-choice-label` 这条 CSS 保留：官方模型卡里的架构选择器（`ModelArchitectureSelector`）
  仍在用它显示「骨干模型 + 家族」。
- 数据集列表原来的 `margin: 4px 0 0` 改为 0，让两个选项直接贴住卡片的 12px 行间距。
- 验证范围：`node --test` 650 项通过（数据集 / 模型分区断言读的是 `aria-label`，无需改动）、
  `npm run build` 通过；未运行浏览器审计脚本。

## 33. 2026-09-28 训练参数格删掉量纲小字

用户指出参数矩阵每格下方的量纲小字（时间步 / 预测步 / epochs / 样本 / 学习率）也不需要：

- `parameterFields` 去掉 `unit` 字段，参数格只剩「标签 + 数值」（`<small>{field.unit}</small>` 删除）。
- 连带清理：`copy.unitSteps` / `unitPredictSteps` / `unitEpochs` / `unitSamples` / `unitLearningRate`
  五个文案键删除，CSS 里 `.experiment-param-cell small` 规则与桌面字号覆盖删除，
  `verify-font-hierarchy.mjs` 的 `params.unit` 探针标为 `REMOVED`。
- 测试：`experimentConfigStructure.test.js` 里「量纲保留」的断言改成
  `assert.doesNotMatch(trainingBlock, /field\.unit|<small>/)` 并断言五个 `unit*` 文案键不存在。
- 副作用：参数格只剩两行（标签 + 数值），`epochs` / 学习率这些数值不再有中文单位提示；
  字段含义由标签本身承担（输入窗口 / 预测步长 / 训练轮次 / 批大小 / 学习率）。
- 验证范围：`node --test` 650 项通过、`npm run build` 通过；未运行浏览器审计脚本。

## 34. 2026-09-28 训练页接通地球 MERRA-2 数据集

用户要求在训练页选择 `earth_merra2_daily_v2`，完成官方 DLinear 训练后进入预测页按历史起点查看三天预测场。训练页的改动如下：

- **数据集分区**由两个选项扩为三个（`openmars_mcd`、`mcd_overview`、`earth_merra2_daily_v2`），仍是直接列出的单选按钮，没有退回原生下拉。选项来源仍是 `trainingParamSanitizers.js` 的常量，`sanitizeTrainingDataset()` 新增 `allowEarth` 开关：通用构建器只接受火星身份，避免地球身份被静默带进火星请求。
- **切换数据集是一套换挡逻辑**（`handleTrainingDatasetChange`，不再是原始 setter）：**地球与火星各自保存完整草稿**（`captureTrainingDraft`），进入地球时保存火星草稿、切回时 `resolveMarsTrainingRestore` 恢复，从火星回到地球时由 `resolveEarthTrainingRestore` 恢复地球自己的模型来源、上传模型选择与自定义参数；地球的固定契约（`window=7`、`horizon=3`、SPHERE 关闭、迁移关闭、通道切换为 `TO3 + 四个辅助变量`）始终生效，不受草稿影响。请求顺序与模型通道顺序解耦，规范顺序固定。
- **地球数据集分区不再渲染说明面板**（原 `EarthTrainingDatasetPanel.jsx` 已移除）：发布日期划分、36×72 全球网格、通道与单位、发布指纹都由服务端在创建任务时校验并绑定，页面侧不重复展示；数据不可用时仍在就绪检查里给出阻塞原因并禁用「开始实验」。
- **地球模式允许官方 DLinear 与用户上传模型两种来源**：模型分区提供来源切换，选上传模型时复用同一套上传卡片，并向 `GET /api/user-models/{id}/earth-compatibility` 取服务端 Earth 兼容性结论——未取到结论一律按不可用处理，不兼容时显示具体原因并禁用提交。此处曾出现一处过期文案（地球面板仍写「不支持上传模型」）与来源切换按钮矛盾，随面板移除一并消除。
- **地球模式隐藏无效控件**：SPHERE、迁移学习页签、架构选择器（官方来源时）都不出现；专家页签为输入与预测 / 训练参数 / 训练策略 / 实验标签；参数格的上下限改用地球自己的范围（epochs 1…1000、batch 1…64、学习率 (0,1]），不悄悄夹到火星范围。
- **提交走独立路径**：`buildEarthTrainingHyperparameters()` 只发送服务端白名单字段（不发送 `dataset_version` / `dataset_fingerprint` / `dataset_snapshot`；上传模型只发送模型 ID 与自定义参数值，版本与内容哈希由服务端固定），`startTrainingTask()` 新增顶层 `datasetId` 并把结构化错误（`detail.code` / `detail.message` / HTTP 状态）保留下来，前端不再只显示裸状态码。
- **预测页**新增独立的「地球历史预测」模式（`earth`）：从已完成地球任务读取可选起点范围与网格，按起点展示三天预测 / 参考 / 残差场（DU）与总体及逐日 RMSE/MAE。该模式隐藏火星侧栏与 MY/Ls 控件，并使用地球自己的色彩区间（预测与参考共用物理色阶、残差零中心）；缓存键包含 `planet`、数据集 ID/版本/指纹、任务与预测起点，切换任务或起点会清空旧结果。
- **单位**：地球 TO3 本身就是 DU，因此地球场与指标不做火星的 μm-atm 换算。
- 后端契约、错误码与验证入口见 [地球训练与历史预测](earth-training.md)；本文件只描述训练页与预测页交互。
- 验证范围：后端地球相关测试逐文件通过（契约 41、数据 19、模型 24、产物 48、runner 17、服务 10、隔离 14、预测服务 23、路由 20），既有注册表/身份/通道/总览回归逐文件通过；前端 `node --test` 680 项通过、`npm run build` 通过；真实 v2 小包经 HTTP 完成任务 27 的训练（测试集 RMSE 46.791 DU）与 2021-07-08 起点的预测（RMSE 29.139 DU），错误路径 7/7 符合预期。**未运行浏览器点击验收**。

## 35. 2026-10-01 设置训练默认值

偏好设置新增训练默认值分区，可保存训练轮次、批大小、学习率、输入窗口、预测步长、训练 / 验证 / 测试比例、随机种子、早停轮数，以及迁移学习开关、冻结策略和微调学习率。设置沿用 `astraatmos_settings` 浏览器本地存储，适用于当前浏览器配置，不会改写已创建任务。

- 新建实验读取已保存的默认值。2026-10-08 修复已挂载配置页只初始化一次的问题：修改有效默认值后，当前草稿中未手动修改的参数即时同步，切换实验配置 / 训练监视也沿用最新值；手动修改过的字段保留原值。「复制配置」从源任务回填参数，并保持到下一次「新建实验」为止。
- 同步只接收完整有效的输入，清空字段或输入越界值时不会把当前草稿重置为兜底值。训练 / 验证 / 测试比例作为一组同步；任一比例手动修改过后，该组全部保留草稿值。
- 训练集和测试集比例必须大于 0，Mars 验证集比例允许为 0，三者合计 100%；Earth 三项均须大于 0。不完整或合计不正确时，设置界面提示当前比例无效，新建表单使用安全默认值，提交时再次校验。Mars 默认验证比例为 0 时，Earth 回退 70/20/10。
- Earth 三小时窗口默认 56→24，可配置 1–240 个时间步，比例使用任务级原始 UTC 分区，不开放迁移学习。窗口默认值在进入或新建实验时应用；Earth 已挂载草稿不随设置重写窗口。已选迁移来源任务的结构窗口也不会被设置覆盖。
- 设置只影响前端新实验初始值与未编辑的草稿字段。实际启动请求仍携带完整训练参数并保存在对应任务中，不新增后端设置接口、数据库表或持久化草稿。

Earth 的“训练策略”页签支持比例编辑（默认 70/20/10，请求为 0–1），按当前窗口预览实际 UTC 起止、时间步数和窗口数；非空验证集及过短分区会阻止提交。Earth/Mars 草稿分别保存比例，复制配置回填请求值；参数详情和矩阵可查看服务器固定策略与边界。完整口径及旧 manifest 任务兼容见[地球任务级划分](earth-task-splits.md)。
