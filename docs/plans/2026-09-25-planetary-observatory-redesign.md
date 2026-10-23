# 数据总览「行星观测台」改版计划书

> **For agentic workers:** 使用 `executing-plans` 按任务执行本计划；步骤采用复选框跟踪。本文为项目功能设计与实施计划，不是智能体 skill 文件。

**Goal:** 将数据总览从左右固定面板的三栏工作台，改为“顶部观测条件 + 全幅行星画布 + 时间轨道 + 底部分析区”，让观测与分析各自获得完整空间。

**Architecture:** 复用 `OverviewShell`，由它管理观测/分析布局及抽屉；Earth 与 Mars 保留各自的数据读取、选择、单位、能力和请求取消逻辑。先拆分现有控件与图表的呈现边界，再接入新外壳，不将 Mars 强行迁入 Earth 控制器。

**Tech Stack:** React 19、现有 CSS 与品牌变量、MUI 控件/图标、Three.js、Plotly、Node.js 内置测试。首期不增加 UI 框架、状态库或后端接口。

**状态：已实施（2026-09-25）。** 2026-09-25 根据当前源码与项目文档编写；同日完成实施与验证。
七个阶段（基线 → 新外壳 → Earth 接入 → Mars 接入 → 分析/点位 → AI/视觉 → 回归与文档）全部落地：
新增 `workbench/observatoryLayout.js`（含测试）、`ObservatoryToolbar.jsx`、`ObservatoryTools.jsx`、
`ObservatoryToolParts.jsx`、`AnalysisDock.jsx`、`ObservatoryMars.jsx`、`MarsSourceControls.jsx`、
`PointProbeContent.jsx`；`OverviewShell.jsx` 改为四槽位用例并输出真实场景尺寸。

实测（Chrome 153 headless + CDP，1440×900，导航 70px）：观测档画布 **766px**、左右竖栏各 124px；
分析档**不渲染三维球体**，分析区 702px、主图宽 1396px（分析内容宽 1434px 的 97%）；
390/768/1024/1280/1920 五种宽度横向溢出均为 0px。前端测试全量 447 项通过，生产构建成功。
（主图高度随下方辅助面板预留变化，整理后的实测值见下文「分析档版式整理」。）

**与原计划 3.3 节的偏差（按用户反馈调整）：** 计划原定观测档保留 192px 底部分析区放紧凑预览。
实施后用户要求「选择观测时前台只展示与观测有关的内容」，因此改为：观测档分析区整行收起（0 高），
画布吃掉全部剩余高度；观测档的工具栏只保留「显示」面板，分析分组、图层（多源/验证/差值）与点位面板
随分析档出现。原计划的紧凑预览能力保留在 `AnalysisDock` 中，未被删除，只是不在观测档渲染。

**时间轴改为左侧观测刻度轨（2026-09-25 追加）：** 观测档的底部时间轨道改为立在画布左侧的竖轨
（`ObservatoryTimelineRail`），包含年份、等宽日期读数、年内第几天、当天全球面积加权均值、
月份刻度、按年内均值中心化的全球均值偏离带，以及播放/步进/重播；轨道交互复用 MUI `Slider`
的垂直模式（键盘、焦点、ARIA 由它负责），形状与刻度自己绘制。右侧再加一条镜像的
**点位年变化轨**（`PointTrendRail`），画所选点位的全年序列与当前值，与左轨共用时间轴语义；
未选点位时显示占位说明。

**分析档不显示三维球体（2026-09-25 追加）：** 球体那一行整行不渲染，高度全部让给分析区
（1440×900 下分析区 702px）；两条竖轨只在观测档出现。这与计划 3.2 节「分析模式保留
缩小的行星空间参照」相反，是按用户要求改动，`OverviewShell` 因此支持不传 `scene` 的调用方式。
实测观测档左轨 124×766px、画布 1186×766px、右轨 124×766px。窄屏与火星的适配尚未做
（见共用分析工作台文档的「未覆盖项」）。

**分析档版式整理（2026-09-25 追加，按用户反馈）：** 用户指出分析页「乱七八糟」。分析区头部只留
「范围」（年份与数据源在条件栏已有同名同值选择器），变量只由卡片药丸行表达（删除卡片内的
「当前变量: X (单位)」行）；主图下方改为**一行两格**：点位/序列面板与 AI 解读面板并排，
主图高度按「分析区高度 − 头部与注脚 − 下方预留」算出（`observatoryLayout.dockMainChartHeight` 的 `reserve`），
因此两块辅助面板完整落在第一屏内（实测分析区正文 611/611，滚动溢出 0px）。Mars 的 AI 入口从
分析区里的浮动按钮改成内嵌面板（`AICopilotWidget` 的 `embedded` 形态，原浮动形态的锚点变量
`--overview-scene-right` 已不存在，整块退化成 0 高度并飘到左下角）。卡片图标模式栏收成图内右上角横向一行。

未覆盖项：真实摄像头手势识别（仅验证面板与真实画布矩形映射的代码路径）、登录后的个人上传来源实际加载、
浅色主题与英文界面的逐屏截图（已做四种主题×语言组合的无溢出与文案检查、以及分析档浅色+英文截图）。
验证命令、截图位置与发现并修复的缺陷见 [共用分析工作台](../earth-analysis-workbench.md) 的「验证记录」。

---

## 1. 改版目标与范围

### 1.1 交付后的页面

进入 `#/overview`，默认地球、观测模式。用户首先看到大尺寸行星、当前观测条件和清晰的时间轨道；需要研究细节时，展开底部分析区，保持当前星球、数据源、变量、日期和相机朝向。

首期达到以下结果：

1. 默认桌面视图没有常驻左参数栏和右图表栏。
2. 日常操作围绕“选观测条件 → 看空间分布 → 播放/定位时间 → 打开分析”展开。
3. 一次突出一张分析主图，所有现有分析项目仍可找到。
4. 地球与火星使用一致的界面结构，同时准确表达各自的数据能力。
5. 深浅主题、中英文、窄屏和文字放大均可使用。

### 1.2 本期边界

| 纳入本期 | 继续保留的边界 |
| --- | --- |
| 页面空间重组、控件迁移、分析导航、图表尺寸适配、点位结果和 AI 入口重排 | Earth/Mars 数值叠加、跨星球比较不开放 |
| 地球/火星共用观测台外壳 | Earth 训练与预测不开放 |
| 现有三维、Earth 二维切换及 WebGL 降级入口迁移 | 不新增 Mars 二维地图，不重写地理采样和科学计算 |
| 现有数据源、上传来源、色带和单位配置的可达性 | 不改变个人上传与训练数据的边界 |
| 可展开设置与两种固定布局 | 不增加任意拖拽排版、多窗格布局编辑器、工作区云端保存 |
| AI 解读当前图表的入口及上下文校验 | 不自动发送解读请求，不新增模型服务或后端 AI 协议 |

页面在全站导航中仍叫“数据总览”，内部工作区可显示“行星观测台”。首页、训练、预测、数据管理页面不随之整体改版。

## 2. 当前实现与改造依据

| 当前事实 | 设计影响 | 代码入口 |
| --- | --- | --- |
| 默认左栏 300px、右栏 540px，≤1120px 切换顺序布局 | 1440px 宽屏下中央可用宽度约 600px；需要重新分配空间 | [overviewVisualContract.js](../../frontend/src/pages/DataOverviewPage/workbench/overviewVisualContract.js)、[OverviewShell.jsx](../../frontend/src/pages/DataOverviewPage/workbench/OverviewShell.jsx) |
| Mars 使用页面逻辑及 `DataOverviewContext`；Earth 使用 `useOverviewController` | 布局统一不等于数据控制器统一，分开接入 | [DataOverviewPage.jsx](../../frontend/src/pages/DataOverviewPage.jsx)、[EarthWorkbenchScene.jsx](../../frontend/src/pages/DataOverviewPage/EarthOverview/EarthWorkbenchScene.jsx) |
| 当前图表在右栏按模式分组、逐卡展开 | 改为底部的分组导航与单主图区域 | [DetailPanel.jsx](../../frontend/src/pages/DataOverviewPage/DetailPanel.jsx)、[OverviewAnalysisPanel.jsx](../../frontend/src/pages/DataOverviewPage/workbench/OverviewAnalysisPanel.jsx) |
| 地球画布已有容器 `ResizeObserver`；部分 Plotly 图表仍依赖固定高度或窗口 resize | 球体复用容器响应；图表需要显式适应分析区尺寸 | [SphericalFieldCanvas.jsx](../../frontend/src/components/SphericalFieldCanvas.jsx)、[EarthResearchViews.jsx](../../frontend/src/pages/DataOverviewPage/EarthOverview/EarthResearchViews.jsx) |
| Mars 手势坐标仍按左右栏宽与窗口计算 | 必须改为真实画布矩形，否则选点位置偏移 | [DataOverviewPage.jsx](../../frontend/src/pages/DataOverviewPage.jsx) 的 `mapGesturePointerToClientPoint` |
| Mars 球面点击打开 `pointProbe`；`selectedCoordinate` 另有分析上下文 | 不能把两个状态直接合并，避免现有点位分析丢失 | [PointProbeModal.jsx](../../frontend/src/pages/DataOverviewPage/PointProbeModal.jsx)、[DetailPanel.jsx](../../frontend/src/pages/DataOverviewPage/DetailPanel.jsx) |
| Earth 有真实 ISO 日期、DU 等原始单位和日平均能力限制 | 不套用火星 MY/Ls、臭氧换算或昼夜数据 | [OverviewAdapter.js](../../frontend/src/pages/DataOverviewPage/workbench/OverviewAdapter.js)、[earthOverviewAdapter.js](../../frontend/src/pages/DataOverviewPage/workbench/earthOverviewAdapter.js) |

上述为源码核对结果；最终截图、响应速度和图表可读性须在真实运行环境中验收。

## 3. 页面结构

### 3.1 观测模式：默认入口

```text
┌────────────────────────────────────────────────────────────┐
│ 全站导航：AstraAtmos / 数据总览 / 训练 / 预测 / 其他入口      │
├────────────────────────────────────────────────────────────┤
│ 地球·火星  数据源  变量  年份          [观测] [分析]          │
├────────────────────────────────────────────────────────────┤
│ [图层]   当前对象、数据时间、单位                           │
│ [点位]                                                     │
│ [显示]                  大尺寸行星                          │
│                                                            │
│                                     图例、覆盖与加载状态   │
├────────────────────────────────────────────────────────────┤
│ 播放/暂停  上一步  ───── 时间轨道 ─────  下一步  日期/Ls     │
├────────────────────────────────────────────────────────────┤
│ 年内变化 / 季节结构 / 极区 / 全部分析          展开分析 ↗    │
│ 当前分析的紧凑预览与范围说明                               │
└────────────────────────────────────────────────────────────┘
```

- 地球/火星、变量、年份直接操作；数据源按钮显示当前来源，点开选择官方/个人来源及文件。
- 画布工具只保留“图层、点位、显示”三个有文字的入口。
- 数据来源、发布时间范围、实际展示时间和单位可追溯；发布指纹等技术信息放入数据详情。
- 底部默认展示“年内全球变化”紧凑预览。若来源不支持该图，显示实际能力说明与其他分析入口，不造假曲线。
- 对多子图或需要大量控制的分析，紧凑区展示图名、范围和“展开查看”；不把完整复杂图压进很矮的区域。

### 3.2 分析模式：主图优先

```text
┌────────────────────────────────────────────────────────────┐
│ 全站导航 + 同一条观测条件工具栏                            │
├────────────────────────────────────────────────────────────┤
│ 当前对象与数据状态        缩小的行星空间参照                │
├────────────────────────────────────────────────────────────┤
│ 同一条时间轨道                                             │
├────────────────────────────────────────────────────────────┤
│ 变化规律 / 变量关系 / 空间诊断 / 点位        返回观测 ↙      │
│ 当前分组的图表选择                                        │
│                                                            │
│                    一张宽幅分析主图                        │
│                                                            │
│ 范围、单位、数据源                         AI 解读此图      │
└────────────────────────────────────────────────────────────┘
```

- “观测/分析”和“展开/返回”是同一个状态的两个入口，不产生两套独立布局状态。
- 初期提供固定的观测、分析两档，不实现拖动改变分析区高度。
- 切换两档时保留同一个三维组件实例，不重建相机，不触发业务数据请求。
- 分析区内部复杂图表可纵向滚动；禁止在图表内部再嵌套一层没有必要的滚动列。
- 展开后主图宽度至少达到分析内容区域的 85%；AI 结果默认放在图下，不恢复为常驻右栏。

### 3.3 桌面尺寸基准

`H` 表示扣除实际导航高度、工具栏高度后，工作区可用高度。以下是首轮实现参数，视觉验收可微调，但功能验收必须保持。

| 区域 | 观测模式 | 分析模式 |
| --- | --- | --- |
| 顶部条件栏 | 桌面基准 64px，可因翻译/文字缩放换行 | 相同 |
| 时间轨道 | 基准 64px，内容增加时允许增高 | 相同 |
| 球体场景 | `H − 时间轨道 − 192px`，至少 320px | 可用主体高度约 28%，限定在 160–240px |
| 底部分析区 | 基准 192px，只展示紧凑内容 | 使用其余空间，至少 288px |
| 临时设置面板 | 宽 320px，贴画布左边，覆盖场景 | 相同 |
| 默认外边距 | 16–24px，画布不被双侧栏裁切 | 相同 |

1440×900、导航 70px、条件栏 64px 的基准下，`H=766px`：观测场景约 510px 高，分析场景约 197px 高、主图区约 505px 高。以上为尺寸预算，不代表已测量结果。

当宽度小于 1024px，或 `H<576px`，使用文档流布局，避免把图表和球体压扁。实际导航、工具栏高度用 DOM 测量；禁止到处复制 70px 偏移。

## 4. 控件与现有功能迁移

| 现有内容 | 新位置 | 必须保留的行为 |
| --- | --- | --- |
| 星球切换 | 顶部条件栏最左 | 默认地球；两星球互斥挂载；切换取消旧请求 |
| 官方/个人 MCD、上传文件选择 | 顶部“数据源”弹层 | 原登录限制、加载、不可用来源提示和年份范围 |
| 年份、变量 | 顶部直接选择 | Earth 实际年份/单位；Mars MY、变量和换算 |
| 当前日期 / Ls | 时间轨道右侧，窄屏另起一行 | 真实可选时间、步进、播放边界、覆盖信息 |
| MCD / 多源 / 验证 / 差值 | “图层”面板 | MCD、OpenMARS、NOMAD 各自能力、来源与图例 |
| 地球数据场、海岸线、经纬网；火星浓度、纹理、标注 | “图层”面板 | 各星球支持项按能力显示 |
| 自动旋转、地球视角重置、火星手势 | “显示”面板 | 不新增 Earth 摄像头手势；维持已有授权交互 |
| Earth 3D/2D | “显示”面板 | 二维点选、范围、变量与日期延续；WebGL 失败自动降级 |
| 色带、单位偏好 | 现有设置保留；“显示”提供直达入口 | 单一设置源；Earth 原始物理单位不被 Mars 偏好覆盖 |
| 三种分析模式、折叠卡片 | 分析区分组 + 图表选择 | 全部现有分析及 `unsupported/loading/error` 状态 |
| Mars 点位弹窗 | 分析区的点位视图 | 原点位数值、序列、对比、加载错误和关闭取消 |
| Earth 坐标输入、点位序列、全球序列 | “点位”面板 + 分析区 | 最近网格、实际采样坐标、返回全球、曲线选择日期 |
| AI / Copilot | 主图旁入口，回答在图下展开 | 用户主动请求；统计摘要、来源与上下文身份 |
| 状态栏、覆盖范围 | 场景左上简洁状态 + 数据详情 | 加载/失败/实际展示时刻明确，旧场不冒充新场 |
| 图表工具栏 | 当前主图内部 | 已有缩放、平移等功能；不承诺新增数据导出 |

### 4.1 分析目录

保留内部 `temporal / drivers / dynamics` ID，只调整面向用户的名称，降低迁移风险。

| 分组 | 图表与原有 ID |
| --- | --- |
| 变化规律（`temporal`） | 年内全球变化 `globalTrend`、季节结构 `seasonal`、季节极值 `seasonalExtremes`、环境因子 `environment`、极区 `polar`、昼夜变化（Mars `realtime` / Earth `diurnal`） |
| 变量关系（`drivers`） | 太阳辐射与臭氧 `solarsens`、变量相关 `correlation`、温度与臭氧 `coupling` |
| 空间诊断（`dynamics`） | 波动/空间距平与纬带诊断 `wave` |
| 点位 | Mars 原 `pointProbe` 结果、原 `distribution` 上下文；Earth `EarthSeriesPanel` 点位/全球序列 |

“点位”是独立的用户任务视图，不向现有 adapter 增加第四个分析 mode。Earth 昼夜项保留可见的能力说明：“日平均数据没有日内采样”，不得发出 Mars 昼夜请求。

火星图表可能固定分析臭氧及环境关系，不能因为顶部球体变量切换就把所有图表标题、单位都改成该变量。每张图按自身真实数据标注范围和变量。

## 5. 交互与状态规则

| 用户动作 | 明确的结果 |
| --- | --- |
| 点击“分析”或“展开分析” | 打开当前主图；保留数据选择、播放状态和相机，不产生额外数据刷新 |
| 点击“观测” | 回到大球体；保留分析选择；关闭当前图表 AI 展示；不自动清除点位 |
| 切换分析分组 | 选中该组上次查看且仍有效的图；否则选择首个可用图 |
| 打开图层/显示/数据源/坐标面板 | 一次仅有一个设置面板打开；点击关闭/Escape 可退出，焦点返回触发按钮 |
| 点击球面点位 | 暂停播放，展示真实坐标和点位结果，进入分析模式；保留进入点位前的普通图表选择 |
| 关闭点位 | 调用原点位关闭/清除动作，取消对应请求；回到此前普通图表，不重置全局观测条件 |
| 改变变量/来源/年份 | 调用原业务动作；清理不匹配的点位与 AI 展示，拒绝过期回包 |
| Earth 更改年度 | 沿用 `selectYear` 的月日保持与闰日处理；暂停播放，避免调整年度时继续跳帧 |
| Earth 逐日播放跨年 | 保留当前年度分析选择；图表标题明确“分析年份”，球体明确“展示日期”，不暗示整组年度图随日播放重新计算 |
| Earth → Mars → Earth | 保留已有的各自选择恢复规则；临时面板/AI/点位结果面板关闭；回到观测模式，相机恢复沿用现有机制 |
| 请求失败或数据缺失 | 在对应内容区给原因与重试；顶部星球切换仍可用，页面不整体消失 |

短暂加载时，数据场和图例的“已展示时间”绑定实际响应；用户请求的新时间独立标注“加载中”。不得用新日期标题配旧曲线。来源身份使用现有数据源 ID、指纹及上传 ID，不用可翻译的显示名称作缓存键。

### 5.1 AI 的具体调整

- Mars 保留 `buildExpandedCardSnapshot` 与现有请求字段，将“右侧展开图表”等方位文案改成“当前分析图表”。
- Earth 将 `insightCards` 筛选为当前选中、数据已就绪的卡片；继续通过现有 `buildEarthInsightSnapshot` 发送摘要。
- `pointProbe`、仅说明的卡片及没有摘要构建器的图不显示“解读此图”的可执行按钮。
- AI 请求绑定星球、来源身份、图 ID、数据年份/时间、变量和区域；上下文改变时取消或使旧请求失效，旧回答不能显示在新图下。
- 收起 AI 回答不触发新请求；再次请求必须由用户点击。

## 6. 视觉规范与响应式

### 6.1 风格

采用克制的科学仪器界面：深蓝灰工作底色、冰蓝主操作、少量橙色强调、细分隔线。保留正式品牌标志及 `--brand-primary / --brand-accent`，科学数据色带、正负差值配色与来源色保持独立。

- 常规正文 14px，控件标签 12–13px，重点数值 18–22px；科学坐标轴目标不低于 11px，均响应现有字体缩放。
- 间距采用 4/8px 节奏，区域间距 16/24px；面板圆角 10–12px，控件圆角 6–8px。
- 主内容面板使用稳定、不透明或近不透明表面；只在少量浮层使用模糊。
- 场景背景降低星点密度与动态干扰，避免多层星空叠加；修改仅作用于总览场景。
- 面板淡入和按钮反馈 150–200ms；不为大画布高度变化制造连续全帧重排，布局切换后立即重测尺寸。
- `prefers-reduced-motion` 下关闭非必要过渡；自动旋转与手势能力仍按原控制与偏好管理。

### 6.2 断点与可访问性

| 条件 | 布局策略 |
| --- | --- |
| ≥1024px，且工作区高度 ≥576px | 完整桌面两档布局 |
| 720–1023px 或桌面高度不足 | 顶部可换行，场景和分析按文档流排列；分析模式降低场景高度 |
| <720px | 控件分两行或更多行，画布至少 280px 高；分析导航使用分组选择器；设置用 MUI 模态抽屉 |
| 字号放大/英文标签较长 | 允许增高与换行，不隐藏选择结果，不产生页面横向滚动 |

按钮有效点击面积至少 44×44px；拖动时间轴必须有键盘/步进替代；图例同时显示单位和范围。模式按钮提供 `aria-pressed`，抽屉提供可见标题和关闭按钮。桌面非模态面板不锁住全页焦点，窄屏模态抽屉管理焦点并在关闭后归还。控件区域阻止事件落入三维拖拽/选点。

## 7. 技术结构与文件责任

### 7.1 状态所有权

```text
DataOverviewPage / OverviewSceneContent
├─ planet、已有 Earth 选择恢复
├─ observatoryView：observe | analyze
├─ openPanel：null | source | layers | display | coordinates
└─ 互斥挂载 EarthWorkbenchScene / DataOverviewPageContent
   ├─ 原星球数据状态、加载器、请求取消与缓存
   ├─ 原 mode、expandedCard（单一主图身份）
   └─ OverviewShell
      ├─ 顶部观测条件
      ├─ scene 容器（模式变化不 remount）
      ├─ timeline
      └─ AnalysisDock（含点位与 AI 的呈现槽位）
```

布局状态不进入后端请求 key；当前主图继续使用原 `expandedCard`，不再维护一个与它竞争的 `activeChartId`。抽屉状态不持久化到全局偏好；刷新回到默认观测模式。

### 7.2 文件映射

以下“新增”均为实施时创建，本计划交付时尚不存在。

| 文件（仓库相对路径） | 动作与责任 |
| --- | --- |
| `frontend/src/pages/DataOverviewPage/workbench/observatoryLayout.js` | 新增：尺寸计算、断点与图表默认选择纯函数 |
| `frontend/src/pages/DataOverviewPage/workbench/observatoryLayout.test.js` | 新增：小高度、窄屏、无可用卡片等行为测试 |
| `frontend/src/pages/DataOverviewPage/workbench/ObservatoryToolbar.jsx` | 新增：共同条件栏，接收星球专属控件与模式切换槽位 |
| `frontend/src/pages/DataOverviewPage/workbench/ObservatoryTools.jsx` | 新增：图层、点位、显示入口及抽屉外壳 |
| `frontend/src/pages/DataOverviewPage/workbench/AnalysisDock.jsx` | 新增：两档分析区域、分组导航、单主图、点位/AI 展示 |
| `frontend/src/pages/DataOverviewPage/workbench/OverviewShell.jsx` | 修改：从两侧固定栏转为条件栏/场景/时间轴/分析区；输出真实场景尺寸 |
| `frontend/src/pages/DataOverviewPage/workbench/overviewWorkbench.css` | 修改：新布局与局部主题变量；按页面作用域处理旧固定样式 |
| `frontend/src/pages/DataOverviewPage/workbench/overviewVisualContract.js` | 修改：布局常量；保留独立的 `OVERVIEW_GLOBE` 科学场景参数 |
| `frontend/src/pages/DataOverviewPage.jsx` | 修改：界面状态、Mars 槽位组装、点位呈现、按真实画布矩形映射手势 |
| `frontend/src/pages/DataOverviewPage/SidebarMenu.jsx` | 修改：拆出可独立挂载的来源、变量/年份、图层、显示内容；保留原业务绑定 |
| `frontend/src/pages/DataOverviewPage/DetailPanel.jsx` | 修改：复用原图表组件目录，以受控单主图替代右侧折叠列表 |
| `frontend/src/pages/DataOverviewPage/workbench/OverviewAnalysisPanel.jsx`、`OverviewCard.jsx` | 修改：保留卡片状态处理，提供非折叠的主图呈现；去除新布局中的重复头部 |
| `frontend/src/pages/DataOverviewPage/EarthOverview/EarthWorkbenchScene.jsx` | 修改：拆分条件/图层/显示/坐标控件，组装共同外壳 |
| `frontend/src/pages/DataOverviewPage/workbench/useOverviewController.js` | 小范围修改：默认主图选择与可见性接入；保留数据身份、取消及缓存机制 |
| `frontend/src/pages/DataOverviewPage/TimelineController.jsx`、`GlobeLegend.jsx`、`TopStatusBar.jsx` | 修改：改用所在容器布局，保留覆盖/图例/格式逻辑 |
| `frontend/src/pages/DataOverviewPage/EarthOverview/EarthTimeline.jsx`、`EarthSeriesPanel.jsx` | 修改：新容器尺寸和展示位置；保留日期选择与序列含义 |
| `frontend/src/pages/DataOverviewPage/PointProbeModal.jsx` | 修改：抽出结果内容，支持嵌入分析区；保留请求模型、单位换算和完整对比内容 |
| `frontend/src/pages/DataOverviewPage/AICopilotWidget.jsx`、`EarthOverview/EarthInsightPanel.jsx` | 修改：嵌入入口、上下文失效处理、当前图摘要 |
| `frontend/src/pages/DataOverviewPage/Mars3DBackground.jsx`、`workbench/OverviewScene.jsx` | 修改：消除旧双栏偏移，场景始终在其容器中居中 |
| `frontend/src/pages/DataOverviewPage/OverviewCharts/GlobalTrendLinesChart.jsx`、`EarthOverview/EarthResearchViews.jsx` | 修改：紧凑趋势预览与主图尺寸参数，复用原数据序列 |
| `frontend/src/components/SphericalFieldCanvas.jsx` | 仅在容器尺寸验证发现必要问题时修改；保留已有 ResizeObserver、拾取与相机缓存 |
| `frontend/src/components/Navbar.jsx`、`frontend/src/App.jsx` | 最小改动：总览导航测量入口及局部布局类，保持其他页面行为 |
| `frontend/src/i18n/zh.js`、`frontend/src/i18n/en.js` | 修改：观测/分析、面板标题、当前图表等文案 |
| `README.md`、`docs/earth-analysis-workbench.md`、`docs/earth-overview.md`、`frontend/README.md` | 实施完成后同步实际行为；保留数据协议与尚未开放能力 |

只有已确认需要单独复用的模块才拆文件；不为本次改版顺便重构全部图表或后端。已有源码中失效的固定栏逻辑可在验证后原文件内移除，不执行文件或目录批量删除。

## 8. 分阶段实施清单

每个任务完成后执行其验证。提交仅包含明确属于本任务的文件；如果工作区已有其他修改，不使用 `git add .`，不重置、覆盖或自动清理。当前计划不执行提交、推送或部署。

### Task 1：建立迁移基线

**文件：** 上述现有入口、对应测试、`docs/earth-analysis-workbench.md`。

- [ ] 检查最新 Git diff，记录本次允许修改的文件和已有用户修改。
- [ ] 使用当前可访问的 Earth 与 Mars 数据，记录桌面观测、现有分析、点位、图层和窄屏状态；浏览器不可用时明确记录缺项，不宣称截图验收通过。
- [ ] 按第 4 节逐项核对现有入口，记录测试账号/数据不可用导致的未覆盖项，不将其误认成功能已移除。
- [ ] 运行第 10 节的总览基线测试，区分既有失败与本次引入的失败。

**完成条件：** 每个现有功能都有新位置；测试基线与可用数据范围已记录。

### Task 2：建立观测台布局契约

**新增：** `observatoryLayout.js`、`observatoryLayout.test.js`。

- [ ] 先使用第 9 节测试定义尺寸预算、紧凑降级和卡片选择行为，执行测试确认缺少实现时失败。
- [ ] 写入第 9 节纯函数，实现后重跑同一测试。
- [ ] 在页面层建立 `observatoryView`、`openPanel` 两个界面状态；数据 mode 和主图身份继续使用原状态。
- [ ] 使用下述约定连接模式切换，禁止以 view 给三维组件设置 key：

```jsx
const [observatoryView, setObservatoryView] = useState('observe');
const [openPanel, setOpenPanel] = useState(null);
const switchObservatoryView = (next) => {
  if (next !== 'observe' && next !== 'analyze') return;
  setOpenPanel(null);
  setObservatoryView(next);
};
```

**完成条件：** 两档尺寸合法，小高度进入顺序布局；主图选择不会丢失或返回不存在的 ID。

### Task 3：搭建共同外壳与条件栏

**新增：** `ObservatoryToolbar.jsx`、`ObservatoryTools.jsx`、`AnalysisDock.jsx`。**修改：** `OverviewShell.jsx`、`overviewWorkbench.css`、`overviewVisualContract.js`、导航与页面测量入口。

- [ ] 将 Shell 槽位改为 `toolbar / tools / scene / timeline / analysis / overlay / notification`；外层接收 `view` 与 `onViewChange`，不直接读取星球业务状态。
- [ ] 在既有布局 Context 中保留兼容字段 `offsetX: 0`，加入 `sceneRef` 供手势和浮层使用；通过真实容器尺寸驱动布局。
- [ ] 测量导航、条件栏和时间轨道的实际高度；用 `ResizeObserver` 更新预算，并在卸载时清理。
- [ ] 将场景、时间轨道和底部分析区放入 Grid 普通布局；图例/相机小窗只在 scene 内绝对定位，不再按窗口减去左右栏宽。
- [ ] 实现桌面非模态设置面板和窄屏 MUI 模态抽屉；一次一个，关闭还原焦点。
- [ ] 核对 1440×900、1280×720、390×844 的尺寸；高度不足时使用文档流，不隐藏主操作。

新 Shell 对调用者的槽位契约为：

```jsx
<OverviewShell
  planet={planet}
  isLight={isLight}
  view={observatoryView}
  onViewChange={switchObservatoryView}
  toolbar={toolbar}
  tools={tools}
  scene={scene}
  timeline={timeline}
  analysis={analysis}
  overlay={overlay}
  notification={notification}
/>
```

其中 `planet/isLight` 为已有场景值，七个槽位均为调用方创建的 ReactNode；`notification` 继续用于错误/通知，不再默认承载全屏点位弹窗。

**完成条件：** 新页面轮廓成立；无常驻两侧栏；不同模式下 scene 是同一实例。

### Task 4：迁移 Earth 与 Mars 控件、图例和时间轨道

**修改：** `DataOverviewPage.jsx`、`SidebarMenu.jsx`、`EarthWorkbenchScene.jsx`、`TimelineController.jsx`、`EarthTimeline.jsx`、`GlobeLegend.jsx`、`TopStatusBar.jsx`、两个场景组件。

- [ ] 先迁移 Earth 条件、坐标、图层和显示控件，继续绑定原 controller 与已有 state。
- [ ] 再拆分 Mars 控件，保留官方/个人文件筛选、MCD/多源/验证/差值等完整流程。
- [ ] 更新两种时间组件的 embedded 布局；保留地球日历、火星 Ls 格式、覆盖与步进，不人为统一时间单位。
- [ ] 将 source error 分支也放入新 Shell；失败时仍能切换星球或重试。
- [ ] 用 `sceneRef.current.getBoundingClientRect()` 替换手势的窗口/栏宽推算；统一指针映射后仍调用原 `pickGlobeAtClientPoint`。
- [ ] 验证抽屉关闭、模式切换与语言切换只改变表现，不重发 field/年度请求。

**完成条件：** 第 4 节控制功能全部可达，两星球相互切换时数据和单位正确。

### Task 5：迁移底部分析与点位

**修改：** `DetailPanel.jsx`、`OverviewAnalysisPanel.jsx`、`OverviewCard.jsx`、`useOverviewController.js`、`EarthWorkbenchScene.jsx`、趋势图组件、`PointProbeModal.jsx`、`EarthSeriesPanel.jsx`。

- [ ] 提供受控的分组/图表选择：当前主图仍读写 `expandedCard`，默认取 `globalTrend`，组切换通过 `pickActiveCard` 选择合法项。
- [ ] 调整 Mars 的“首图自动挂载”逻辑，使默认主图一致，不再提前加载未显示的旧首张 seasonal。
- [ ] 将非活动图表停止渲染或暂停其 effect；缓存仍由原数据客户端管理。保持 Earth 年度 suite 请求去重，逐日播放不重新拉年度 suite。
- [ ] 为原趋势图增加紧凑呈现选项，复用相同数据序列和单位；详细图表不缩小成不可读的小图。
- [ ] 将 Mars 点位结果内容和 Earth 序列迁入分析区；保存点位前的普通分析选择，关闭恢复，不混淆 `pointProbe` 与 `selectedCoordinate`。
- [ ] 根据父容器真实宽高调整 Plotly；分析区显示后执行必要的 resize，避免仅靠多个延时触发窗口 resize。
- [ ] 针对日期/来源/变量/点位变化验证过期请求；没有新数据时显示加载或原因，不显示旧标题下的旧曲线。

**完成条件：** 全部分析目录、点位结果可达；展开后图表真正变宽变高，单位、坐标与图例可读。

### Task 6：迁移 AI 与完成视觉收敛

**修改：** 两个 AI 组件、Earth 摘要组装、总览 CSS、`zh.js`、`en.js`。

- [ ] 实现第 5.1 节的当前图摘要、显式请求和上下文失效规则；移除所有“右侧卡片”的过时描述。
- [ ] 统一条件栏、按钮、面板、图表标题与状态样式；图表数据配色不随品牌强调色被覆盖。
- [ ] 调整总览局部星空与旧 GlowCard 包装，检查浅色主题的文字、边框与色带对比。
- [ ] 检查浮层层级：页面内面板高于画布、低于全站导航；模态抽屉及遮罩高于当前 `z-index: 2000` 的导航，使用 MUI 焦点管理，避免导航浮在模态遮罩之上。
- [ ] 验证英文、200% 文字/浏览器缩放、减少动态效果、键盘操作和关闭面板的焦点位置。

**完成条件：** AI 不发多余请求、不串图；两主题与中英文布局均可操作。

### Task 7：回归、文档和可审查交付

**修改：** 对应旧布局测试、四份项目文档及实现中发现的具体回归。

- [ ] 将只断言三栏类名/固定栏宽的旧测试改为新布局行为；保留科学语义、请求隔离、点位与单位相关断言。
- [ ] 执行第 10、11 节验证；针对失败修复并重跑受影响项，不以构建成功替代浏览器验收。
- [ ] 在 README 更新实际总览结构、业务链路与功能边界；专题文档记录控件新位置、能力限制及验证范围。
- [ ] 交付深浅主题的观测/分析截图、窄屏截图、功能迁移清单、实际测试结果及未覆盖项。
- [ ] 仅在实施完成且验证通过后将本计划状态改为“已实施”；保留历史三栏计划作为背景，标注由本计划替代的布局部分。

**完成条件：** 所有必验项有证据；计划、README、实际页面三者一致。

## 9. 首个实现任务的可执行参考

以下代码是计划中的首个纯函数模块，不属于已实现的项目功能。后续组件读取它，避免 CSS、测试、JS 各自维护一套尺寸规则。

### 9.1 `observatoryLayout.js`

```js
export function getObservatoryLayout({
  width, height, navHeight, toolbarHeight,
  timelineHeight = 64, view = 'observe',
}) {
  const contentHeight = Math.max(0, height - navHeight - toolbarHeight);
  if (width < 1024 || contentHeight < Math.max(576, timelineHeight + 512)) {
    return { flow: true, contentHeight, sceneHeight: view === 'analyze' ? 280 : 360, dockHeight: null };
  }
  const bodyHeight = contentHeight - timelineHeight;
  const sceneHeight = view === 'analyze'
    ? Math.max(160, Math.min(240, Math.round(bodyHeight * 0.28)))
    : bodyHeight - 192;
  return { flow: false, contentHeight, sceneHeight, dockHeight: bodyHeight - sceneHeight };
}

export function pickActiveCard(cards, previous, preferred = 'globalTrend') {
  const rows = Array.isArray(cards) ? cards : [];
  if (rows.some((card) => card.key === previous)) return previous;
  const supported = rows.filter((card) => card.state?.status !== 'unsupported'
    && card.status !== 'unsupported');
  return supported.find((card) => card.key === preferred)?.key
    ?? supported[0]?.key
    ?? rows[0]?.key
    ?? null;
}
```

所有尺寸输入来自真实 DOM 测量的非负 CSS px；`view` 由页面状态限制为两个合法值。`dockHeight: null` 表示自然内容高度，不写入 `height: nullpx`。

### 9.2 `observatoryLayout.test.js`

```js
import test from 'node:test';
import assert from 'node:assert/strict';
import { getObservatoryLayout, pickActiveCard } from './observatoryLayout.js';

test('observing and analysing share one finite vertical budget', () => {
  const input = { width: 1440, height: 900, navHeight: 70, toolbarHeight: 64 };
  const observe = getObservatoryLayout(input);
  const analyze = getObservatoryLayout({ ...input, view: 'analyze' });
  assert.equal(observe.flow, false);
  assert.equal(observe.sceneHeight, 510);
  assert.equal(observe.dockHeight, 192);
  assert.ok(analyze.dockHeight > observe.dockHeight);
  assert.ok(analyze.sceneHeight >= 160);
  assert.equal(analyze.sceneHeight + analyze.dockHeight + 64, analyze.contentHeight);
});

test('narrow screens, large chrome, and short screens use document flow', () => {
  const base = { width: 1440, height: 900, navHeight: 70, toolbarHeight: 64 };
  assert.equal(getObservatoryLayout({ ...base, width: 390 }).flow, true);
  assert.equal(getObservatoryLayout({ ...base, height: 600 }).flow, true);
  assert.equal(getObservatoryLayout({ ...base, toolbarHeight: 350 }).flow, true);
  assert.equal(getObservatoryLayout({ ...base, timelineHeight: 300 }).flow, true);
});

test('card selection preserves a valid selection and explains unsupported catalogs', () => {
  const cards = [
    { key: 'diurnal', state: { status: 'unsupported' } },
    { key: 'seasonal', state: { status: 'idle' } },
    { key: 'globalTrend', state: { status: 'idle' } },
  ];
  assert.equal(pickActiveCard(cards, 'missing'), 'globalTrend');
  assert.equal(pickActiveCard(cards, 'seasonal'), 'seasonal');
  assert.equal(pickActiveCard(cards, 'diurnal'), 'diurnal');
  assert.equal(pickActiveCard(cards.slice(0, 1), null), 'diurnal');
  assert.equal(pickActiveCard([], null), null);
});
```

执行目录为仓库内 `frontend`：

```powershell
node --test src/pages/DataOverviewPage/workbench/observatoryLayout.test.js
```

预期：创建模块前失败；实现后上述测试全部通过。其余 DOM、焦点、3D 与图表尺寸通过真实页面验证，不新增只检查 CSS 字符串的装饰性测试。

## 10. 验证命令与运行约定

### 10.1 自动验证

以下命令是实施阶段要执行的清单，本次计划交付不声称这些应用测试已通过。

工作目录：仓库内 `frontend`。

```powershell
node --test src/pages/DataOverviewPage/workbench/overviewRequestCoordinator.test.js src/pages/DataOverviewPage/workbench/OverviewAdapter.test.js src/pages/DataOverviewPage/overviewSceneModel.test.js src/pages/DataOverviewPage/pointProbeModel.test.js src/pages/DataOverviewPage/timelinePlayback.test.js src/pages/DataOverviewPage/EarthOverview/earthResearchClient.test.js src/pages/DataOverviewPage/EarthOverview/earthOverviewModel.test.js
node --test
npm run build
```

预期：相关测试及全量前端测试通过；Vite 生产构建完成。如果存在既有失败，记录原始基线与本次 diff，不能直接把失败归为既有问题。不要调用不存在的 `npm test` 脚本。

纯前端布局改造默认无需重跑全部训练或后端测试；若实际改动 API、请求字段或服务，必须追加对应后端回归并更新本计划范围。

### 10.2 本地运行

优先复用已有健康服务。需要启动时遵循工作区 `AGENTS.md`：后端必须使用已配置的 conda `AstraAtmos` 环境，不能换成系统 Python 或其他虚拟环境。环境权限遵循当前工具的有效配置；已处于完整权限时不附加工具禁止的参数。

下面的后端命令针对当前 Windows 工作区，解释器路径来自工作区 `AGENTS.md` 的固定运行约定；不得通过 `Get-Command python` 选择另一套环境。跨机器部署应遵循部署文档，不将此路径写入应用源码或 README 的通用示例。

后端工作目录：仓库内 `AstraAtmos_backend/backend`；仅当 `.env` 不存在时复制 `.env.example`，不覆盖现有配置。

```powershell
& "$env:CONDA_PREFIX\python.exe" -m uvicorn main:app --reload --reload-dir . --host 0.0.0.0 --port 8000
```

上式由自动化后台进程执行，不使用开启可见新窗口的一键启动脚本。前端先在 `frontend` 完成构建，再从仓库根目录设置 `ROOT=frontend/dist`、`PORT=5173`、`API_PORT=8000`，后台运行 `node scripts/serve-prod.mjs`。不要终止不属于本任务的服务进程。

浏览器验收前必须确认以下三个请求均返回 HTTP 200：

```powershell
(Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -UseBasicParsing).StatusCode
(Invoke-WebRequest -Uri 'http://127.0.0.1:5173' -UseBasicParsing).StatusCode
(Invoke-WebRequest -Uri 'http://127.0.0.1:5173/api/datasets' -UseBasicParsing).StatusCode
```

只看到进程存活、前端首页可打开或使用旧 `dist` 都不足以完成验收。

## 11. 验收标准

| 编号 | 可判定标准 | 对应任务 |
| --- | --- | --- |
| V1 | 1440×900 默认视图没有常驻双栏；顶部条件栏、全幅画布、独立时间轨道、底部预览结构明确 | 2、3 |
| V2 | 分析模式主图获得至少 85% 的分析内容宽度；球体成为上方空间参照 | 3、5 |
| V3 | 两档切换不改变星球、日期/Ls、变量、数据源、选中图或相机姿态；不增加业务请求 | 3–5 |
| V4 | 所有第 4 节迁移项可操作；功能不可用时原因明确 | 4、5 |
| V5 | Earth 仍为真实 ISO 日期、原始单位、全球真实网格、面积加权均值；Mars 仍为 MY/Ls 和原来源/换算 | 4、5 |
| V6 | 快速切换星球、来源、变量及点位，延迟响应不会覆盖新页面；Earth 不调用 Mars 分析接口 | 4、5 |
| V7 | 年度图表标题显示其分析年份；日播放不造成年度 suite 重复加载；图例与实际展示数据同一时间身份 | 4、5 |
| V8 | 3D 选点、二维选点、点位关闭/恢复、Mars 手势坐标在新尺寸下正确 | 4、5 |
| V9 | Earth 缺数据、WebGL 不可用、分析失败、AI 未就绪均有可恢复界面；不可用卡片不请求数据 | 4–6 |
| V10 | AI 必须主动触发；回答和摘要对应当前图与来源；切上下文后旧回答不冒充新结果 | 6 |
| V11 | 390、768、1024、1280、1440、1920px 宽度；720、900px 桌面高度；中英文、深浅主题、200% 缩放无页面横向溢出 | 3、6、7 |
| V12 | 键盘可访问主操作；抽屉关闭焦点正确；控件点击不触发球体选点；减少动态效果有效 | 6、7 |
| V13 | 模式切换不新增 WebGL 实例；ResizeObserver、resize 监听、动画循环卸载时清理；无 resize 循环错误 | 3、5、7 |
| V14 | 前端测试、生产构建和差异检查有实际结果；README 与专题文档描述最终实现 | 7 |

需要覆盖的真实操作链：

1. Earth：进入 → 切变量 → 定位闰日 → 切年度 → 观测/分析 → 点位 → 返回全局 → AI。
2. Mars：进入 → MY → 数据源 → 多源/验证/差值 → Ls 播放 → 分析 → 点位 → 手势（设备与授权可用时）。
3. 隔离：加载期间 Earth → Mars → Earth；上传来源 → 官方来源；图表 A 请求未完成时切图表 B。
4. 降级：窄屏、缺失数据、接口错误、WebGL 不可用；每项记录可用/未覆盖及原因。

## 12. 交付顺序与风险处理

执行顺序固定为：**基线 → 新外壳 → Earth 接入 → Mars 接入 → 分析/点位 → AI/视觉 → 回归与文档**。先把两星球真实数据放进新结构，再完成视觉细节；不以静态草图冒充业务接入。

| 风险 | 处理方式 |
| --- | --- |
| 老控件和请求逻辑耦合，拆组件导致重复请求 | 原回调和数据 owner 保持唯一，抽屉只负责呈现；用 Network 对照模式切换前后请求 |
| 分析区展开造成球体偏心、手势错位 | 容器尺寸与真实 rect 为唯一依据，移除旧双栏 offset；不按 viewport 假设坐标 |
| 固定高度 Plotly 图表出现空白、NaN、裁切 | 显式父容器高度、可见后 resize、零尺寸保护；复杂图自然高度并允许单层滚动 |
| 控件收起后功能丢失 | 第 4 节迁移矩阵逐项验收；所有原能力保持可达 |
| 页面文案或图例错误地泛化变量/年份 | 每图保留真实分析对象和数据范围；field 与年度研究分别标注 |
| 旧布局结构测试阻碍合法改版 | 只替换布局假设，保留请求、能力、单位、几何、点位断言 |
| 与同时进行的其他改动冲突 | 开始和每阶段结束检查 diff；按文件/区块合并，不覆盖不相关更改 |

建议按“外壳与控件”“分析与点位”“AI、适配与文档”形成三个可审查变更单元。每个单元先完成相应验证；是否提交或发布按后续实际任务授权执行。

完成本计划后，最终产物是一套可长期使用的观测与分析工作区；本期结束点是两星球功能完成迁移并通过上述验收。
