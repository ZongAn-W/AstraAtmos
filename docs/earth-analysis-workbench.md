# 火星 / 地球共用分析工作台

本文记录已实现的共用分析工作台。**布局已于 2026-09-25 改为「行星观测台」**：顶部观测条件栏、
行星画布与两侧观测轨；分析档现为左侧条件栏与右侧结果区。Mars 与 Earth 共用外壳、模式选择、
卡片外壳、三维场景控制与图表解读入口，各自通过适配器声明自己的时间模型、单位、几何与能力。
Earth 的年度分析接口位于 `/api/analysis/earth/overview/*`，与火星 `/api/explore/overview/*` 完全分离。

数据包协议见 [地球小数据包](earth-compact-dataset.md)，二维总览协议见 [二维地球数据总览](earth-overview.md)，
实施背景见 [共用分析工作台实施方案](plans/2026-09-23-earth-shared-analysis-workbench.md)，
布局改版见 [行星观测台改版计划书](plans/2026-09-25-planetary-observatory-redesign.md)（已实施）。

文档最近核对日期：**2026-10-10**。三小时 Earth 总览提供 UTC 时间选择、3 小时播放、降采样地图、原生点位和明确的 UTC 日聚合；日频实现仅作为历史兼容保留，运行入口已停用。

**观测档（默认）：** 地球顶部提供星球、数据源、变量和图层/点位/显示入口；归档日频兼容任务使用日期，三小时任务使用 UTC 日期加中心时间和 3 小时步长，右侧为所选点位对应频率的时间序列。点击“分析”或“展开分析”进入年度分析。

地球左右轨采用火星同款读数卡片、细线曲线、数值范围与底部按钮，桌面轨宽同为 164px。左轨复用 `regionalSeries` 的全球单元面积加权均值，右轨复用 `pointSeries`，不新增请求；**曲线下方按数值填充颜色**：填充从刻度基准线铺到曲线，颜色由本轨数值映射到该变量色带（`railCurveFill.js`），缺测与缺时间步按曲线自身的断点分段闭合。两条曲线共用所选年的时间轴，**数值刻度各用本轨自己的极值**（`earthRailDomain` 每条轨只传自己那一条序列），因此切换取点只重画右轨，左轨形状保持不变；两轨横向位置不再可直接比较，底部刻度读数与提示都写明"本轨独立"。三小时任务使用带 `Z` 的 UTC 时间戳和 3 小时步进，归档日频兼容分支保留日期、闰日和 1 日步进；播放/暂停与从数据集首个发布时间重播仍可用。未选点时右轨显示提示，选点后留在观测档，清除点位恢复占位。地球同样在窄于 720px 时将两轨并排放到球体下方，较矮桌面保留三列并滚动。

火星观测档同样使用左右竖轨：左侧放置当前变量的 MCD 全球均值曲线、Ls 滑块、步进、播放/暂停与重置，右侧常驻单点年变化（未选点显示提示）。点选球面暂停播放并留在观测页；两轨的**真实 Ls 轴与垂直起止位置一致**，数值刻度各自独立（与地球同一口径），曲线下方同样按数值填充。窄于 720px 时，球体在上、两轨并排在下；较矮桌面保留左右位置并允许页面滚动。

填充取色的实现与两条硬约束（两条轨一致，见 `railCurveFill.js`）：轨道高度按 `BAND_COUNT=14` 等分成数值档，每档是一个**实色矩形**，颜色由该档数值在**本轨数值范围**里的位置决定（低值 → 色带低端，高值 → 色带高端），因此整条色带都被用上；色块左边界固定为刻度基准线、右边界为该档内曲线的最右点，与曲线外缘齐平。深色主题对颜色的**最低亮度**做三通道同偏移抬升（`DARK_LUMINANCE_FLOOR=60`），否则色带低端接近纯黑、在黑色画布上等于没有填充。填充透明度为深色 0.55 / 浅色 0.45，曲线、月份刻线和当前日期横线都压在填充之上，读数不受影响。

> **不要改用球体图例的绝对色标给这两条轨着色。** 球体色标是给全球空间分布开的宽度（臭氧 `scope=dataset`，94–547 DU），而全球面积加权均值的年内变化只有十几 DU（2020 年 277.55–294.33 DU）：按绝对色标着色时整条轨只用到色带约 3% 的一小段，看上去接近单色，"颜色随数值变化"完全失效（2026-09-26 返工实测过）。按本轨范围铺满之后颜色如实反映这一年的起伏。代价是颜色与横向位置一样**不可跨轨/跨变量比较**，这与"数值刻度各自独立"是同一取舍；需要绝对口径时看球体图例与读数卡片。

火星左轨复用 `/api/explore/overview/point-probe` 返回的 `series.globalMean`，按年份、变量和 MCD 主源获取，不随播放重复请求；用于读取全局序列的参考坐标不会选中球面点位。均值沿用后端有效网格等权统计，未按面积加权。右轨复用用户点选请求的 `series.point`；两轨均保留原始采样与缺测断点，显示单位沿用火星设置，读数注明最近实际采样 Ls，不插值、不平滑。OpenMARS/NOMAD 叠加和差值模式仅作用于球面，轨道仍明确标注 MCD。请求失败提供重试；切换年份、变量或 MCD 来源取消旧请求并清除旧点位。单项分析仍可展开点位详细对比。

**分析档：** 默认进入“组合看板”。宽屏左侧竖栏提供分析主题、年份与适用条件，右侧以一张主图和两张辅助图展示同一主题，支持逐张放大和返回；图表区宽度不足 820px 时三图顺序排列，页面宽度不足 1024px 时左栏移到上方。“单项深入分析”下拉和“单项分析”切换保留原有全部图表及 AI 解读入口，显式选择优先于默认图。

- 变化规律：所选变量的季节热力图、年内均值曲线和纬带振幅。Earth 年内均值复用后端面积加权序列；Mars 沿用纬向热力图有效纬度行均值，页面说明统计口径。单项“年内全球变化”仍提供原来的多变量比较。
- 变量关系：所选环境变量与臭氧的同期散点、标准化年内曲线和滞后相关。Earth 只提供后端已支持的温度、短波辐射关系，可切换纬带；Mars 按共同 Ls 对齐样本，缺失值不补零，不跨缺口或年界计算时滞。
- 空间诊断：年平均纬向距平地图、纬带 RMS 和峰谷跨度，三图使用同一变量及单位。温度距平和幅度使用温差换算，不减绝对温标偏移量。
- 放大仅改变当前面板尺寸，不重新请求数据；返回组合或按 Escape 可回到同组图表。组合看板与单项分析的火星变量条件分开记忆，避免互相覆盖。当前 AI 解读保留在单项分析中，未开放多图综合 AI 或跨图刷选。

- 地球单项分析条件始终包含年份；季节结构、季节极值和空间距平显示变量选择；环境因子、相关性与两种臭氧关系图显示可直接调整的纬带范围；年内全球变化显示比较方式。
- 单项“年内全球变化”默认展示标准化趋势（Z-score），所有曲线为无量纲；可切回原始数值，各变量使用自己的单位，不能比较绝对大小。标准化数据缺失时显示空值，不以原始物理值替代。
- 地球单项辐射–臭氧、温度–臭氧关系图在宽屏中左右并排展示同期散点与滞后相关，两图等高，统计指标集中在下方；窄屏自动改为上下排列。
- 单项空间诊断在宽屏中左侧展示空间距平热力图、右侧展示纬带 RMS 与峰谷跨度表，说明置于两栏下方；窄屏自动上下排列。
- Earth 分析档隐藏球体、竖轨、独立 UTC 时间栏以及不适用的球体显示工具。UTC 时间、读数和点位序列统一在观测档查看；分析档不重复展示这块内容，也不提供点位操作。坐标输入入口位于观测档的“点位”面板。进入分析档时暂停观测播放。
- 单项分析中的“AI 解读当前图表”按需展开，只有点击生成操作才发请求。辅助区不挤压主图；复杂图表或展开辅助区时允许正文滚动。
- 地球、火星共用数据源弹层入口；桌面浮层挂在外壳独立容器，避免工具栏裁切，窄屏使用模态抽屉。Escape 关闭后恢复入口焦点。
- 火星“数据源”集中放置 MCD 官方/个人来源，以及臭氧显示方式（MCD / 多源 / 验证 / 差值）、OpenMARS 和 NOMAD 官方/个人来源。“图层”与地球对应，提供数据场、经纬网、火星贴图开关；“显示”保留旋转与手势，球体变量在顶部选择。
- 地球观测图例沿用火星的紧凑圆角卡片：变量与原始单位、当前有限值格点数徽标、横向色带及最小/中间/最大值。日期由左侧观测轨承载；图例色带继续与地球数据场一致，风分量使用以零为中心的发散色带。
- 图表加载失败提供“重新加载图表”；地球昼夜分析尚未接入，卡片显示能力说明，不把三小时数据误称为没有日内样本。
- Mars 分析条件集中在左栏：火星年始终可选；季节变化、季节极值、变量相关和空间诊断提供分析变量；太阳辐射关系和昼夜变化提供纬带范围；仅昼夜变化显示 Ls 滑块。空间诊断的地图与纬带指标使用同一个分析变量。顶部不再显示观测变量、球体工具或 MY/Ls/图层来源摘要；图层、点位、显示和播放操作放在观测档，数据源入口两档可用。
- Mars 的变量与纬带选择按图表分别保存，切换图表、分类、观测/分析或星球后恢复；刷新页面恢复默认值。分析档不加载隐藏球体的时间切片或臭氧叠加场，不显示这些请求的加载提示。进入分析时暂停播放，移除时间轴所占空间；年份和 Ls 继续与观测状态同步。
- Mars 主图不再嵌套限高滚动框，复杂结果统一由右侧正文滚动；环境因子、变量关系和极区多图使用响应式网格，空间距平与纬带诊断宽屏并排、窄屏上下排列。每张图只保留一套工作台标题与问题说明，说明按星球对应实际图表，Mars 温度–臭氧耦合展示年内变化曲线。
- Mars 十张活动图表共享加载失败提示与“重新加载图表”按钮，重试保留当前年份、变量和纬带；成功响应但没有数据时保留空数据提示。过期请求的失败不覆盖后续选择。独立使用的旧图表组件仍保留本地条件控件。

外壳不为缺失的时间栏预留空白；桌面主图根据分析区可用高度确定尺寸，筛选栏不再扣减图表高度；左栏与图表正文分别滚动。窄屏或较矮窗口进入文档流布局，导航高度变化会重新测量；数据总览导航在 1100px 及以下分为两行，链接可横向滚动。

## 已实现范围

| 能力 | 状态 |
| --- | --- |
| 共用观测台外壳 `OverviewShell`（条件栏 / 画布 / 时间轨道 / 分析区四个槽位，`observe`+`analyze` 两档，<1024px 或工作区 <576px 转文档流） | 已实现 |
| 共用布局契约（`observatoryLayout.js`：两档尺寸预算、断点、主图选择纯函数；`overviewVisualContract.js`：导航高度、面板宽度、`OVERVIEW_GLOBE`） | 已实现 |
| 共用控制器 `useOverviewController`（请求身份、取消、过期回包拒绝、播放、选择状态；主图身份受控） | 已实现 |
| 适配器契约 `OverviewAdapter`（`planet/sourceId/sourceFingerprint/time/variables/geometry/capabilities/cards`） | 已实现 |
| `marsOverviewAdapter` / `earthOverviewAdapter` | 已实现 |
| 统一卡片 `OverviewCard`：`loading / ready / unsupported / error / idle` | 已实现 |
| 共用三维场景 `OverviewScene` → `SphericalFieldCanvas`（显式 `planet`/`field`/`geometry`/`selection`/`lighting`） | 已实现 |
| Earth 三维全球球体：v2 真实 5°×5° 单元、±90°/±180°、经度接缝不跨接、两极封盖 | 已实现 |
| Earth 本地海岸线底图（`/earth/ne_110m_coastline.geojson`，无在线瓦片） | 已实现 |
| Earth 日期播放（2020-01-01 ~ 2021-12-31，含 `2020-02-29`） | 已实现 |
| Earth 五变量切换与原始单位显示（DU / m s-1 / K / W m-2） | 已实现 |
| Earth 经纬度点选（最近单元中心）、点位曲线、全球面积加权覆盖均值 | 已实现 |
| 三种分析模式：`temporal` / `drivers` / `dynamics` | 已实现 |
| Earth / Mars 主题组合看板：一主两辅、左侧共用条件、单图放大与 Escape 返回 | 已实现 |
| Earth 年度分析（2020 / 2021 分开计算）：季节结构、年内变化、季节极值、环境因子、相关、耦合、空间距平 | 已实现 |
| Earth 极区统计（`\|latitude\| >= 60°`，返回实际采样纬度） | 已实现 |
| Earth 昼夜卡片：`unsupported` + 能力说明，不请求 Mars 昼夜接口 | 已实现 |
| Earth 图表 AI 解读（只发送统计摘要） | 已实现 |
| 手势交互（旋转/缩放/步进/选点动作接口已预留，未接入摄像头识别） | 未启用（第二阶段） |
| Earth/Mars 数值叠加或跨星球比较 | 未实现（首期不做） |
| Earth 训练页、Earth 预测页 | 未实现（训练请求仍返回 409） |

## 组件与代码入口

| 任务 | 入口 |
| --- | --- |
| 观测台外壳与槽位 | [OverviewShell.jsx](../frontend/src/pages/DataOverviewPage/workbench/OverviewShell.jsx)（`toolbar / tools / scene / timeline / analysis / overlay / notification`） |
| 布局与主图选择契约 | [observatoryLayout.js](../frontend/src/pages/DataOverviewPage/workbench/observatoryLayout.js)（`getObservatoryLayout`、`pickActiveCard`、`buildObservatoryGroups`）、[overviewVisualContract.js](../frontend/src/pages/DataOverviewPage/workbench/overviewVisualContract.js) |
| 顶部条件栏与工具面板 | `ObservatoryToolbar` + `ObservatoryTools` + `ObservatoryToolParts`（两颗星球的观测档提供图层/点位/显示，分析档只提供相关分析条件；数据源两档可用。桌面面板挂在外壳浮层，窄屏使用 MUI 抽屉） |
| 分析入口与共用看板 | [AnalysisDock.jsx](../frontend/src/pages/DataOverviewPage/workbench/AnalysisDock.jsx)（组合/单项、条件与目录）、[AnalysisBoard.jsx](../frontend/src/pages/DataOverviewPage/workbench/AnalysisBoard.jsx)（三图、放大、主题与尺寸）、[analysisBoard.css](../frontend/src/pages/DataOverviewPage/workbench/analysisBoard.css) |
| Earth 看板数据 | [EarthAnalysisBoard.jsx](../frontend/src/pages/DataOverviewPage/EarthOverview/EarthAnalysisBoard.jsx)、[earthAnalysisBoardModel.js](../frontend/src/pages/DataOverviewPage/EarthOverview/earthAnalysisBoardModel.js)（复用 controller 年度/空间结果，不独立发请求） |
| Mars 看板数据 | [MarsAnalysisBoard.jsx](../frontend/src/pages/DataOverviewPage/OverviewCharts/MarsAnalysisBoard.jsx)（主题请求、错误重试、过期响应拒绝）、[marsAnalysisBoardModel.js](../frontend/src/pages/DataOverviewPage/OverviewCharts/marsAnalysisBoardModel.js)（均值、共同 Ls 配对、滞后及单位） |
| 模式选择部件 | [OverviewSidebarParts.jsx](../frontend/src/pages/DataOverviewPage/workbench/OverviewSidebarParts.jsx)（`SectionLabel`、`OverviewModeCard`、`AnalysisModePicker`、`modeDefinition`） |
| 状态与请求 | [useOverviewController.js](../frontend/src/pages/DataOverviewPage/workbench/useOverviewController.js)、[overviewRequestCoordinator.js](../frontend/src/pages/DataOverviewPage/workbench/overviewRequestCoordinator.js) |
| 适配器契约 | [OverviewAdapter.js](../frontend/src/pages/DataOverviewPage/workbench/OverviewAdapter.js)、[marsOverviewAdapter.js](../frontend/src/pages/DataOverviewPage/workbench/marsOverviewAdapter.js)、[earthOverviewAdapter.js](../frontend/src/pages/DataOverviewPage/workbench/earthOverviewAdapter.js) |
| 三维场景 | [OverviewScene.jsx](../frontend/src/pages/DataOverviewPage/workbench/OverviewScene.jsx)、[SphericalFieldCanvas.jsx](../frontend/src/components/SphericalFieldCanvas.jsx)、[sphericalRegionalGrid.js](../frontend/src/components/sphericalRegionalGrid.js)、[sphericalEarthBaseMap.js](../frontend/src/components/sphericalEarthBaseMap.js) |
| Earth 场景与卡片 | [EarthWorkbenchScene.jsx](../frontend/src/pages/DataOverviewPage/EarthOverview/EarthWorkbenchScene.jsx)（数据源/图层/点位/显示面板、年度分析条件与分析区组装）、[EarthResearchViews.jsx](../frontend/src/pages/DataOverviewPage/EarthOverview/EarthResearchViews.jsx)、[earthResearchModel.js](../frontend/src/pages/DataOverviewPage/EarthOverview/earthResearchModel.js)、[earthResearchClient.js](../frontend/src/pages/DataOverviewPage/EarthOverview/earthResearchClient.js)、[EarthInsightPanel.jsx](../frontend/src/pages/DataOverviewPage/EarthOverview/EarthInsightPanel.jsx) |
| Earth 后端接口 | [earth_analysis.py](../AstraAtmos_backend/backend/routers/earth_analysis.py)、[earth_research_service.py](../AstraAtmos_backend/backend/services/earth_research_service.py)、[earth_research.py](../AstraAtmos_backend/backend/schemas/earth_research.py) |
| Mars 观测台控件 | [DataOverviewPage.jsx](../frontend/src/pages/DataOverviewPage.jsx)（页面状态、手势 rect 映射、分析区组装）、[ObservatoryMars.jsx](../frontend/src/pages/DataOverviewPage/ObservatoryMars.jsx)（条件栏槽位与四个面板）、[MarsSourceControls.jsx](../frontend/src/pages/DataOverviewPage/MarsSourceControls.jsx)（官方/个人 MCD、OpenMARS、NOMAD）、[MarsObservationRail.jsx](../frontend/src/pages/DataOverviewPage/workbench/MarsObservationRail.jsx)（两侧曲线与 Ls 交互）、[marsObservationRail.js](../frontend/src/pages/DataOverviewPage/workbench/marsObservationRail.js)（刻度、单位、每轨独立数值范围与缺测路径） |
| Mars 分析条件与错误恢复 | [MarsAnalysisSettings.jsx](../frontend/src/pages/DataOverviewPage/workbench/MarsAnalysisSettings.jsx)（页面级条件记忆）、[marsAnalysisSettings.js](../frontend/src/pages/DataOverviewPage/workbench/marsAnalysisSettings.js)（条件适用范围与纬带选项）、[ChartRequestError.jsx](../frontend/src/pages/DataOverviewPage/OverviewCharts/ChartRequestError.jsx)（图表请求错误与重试） |
| 点位结果 | [PointProbeContent.jsx](../frontend/src/pages/DataOverviewPage/PointProbeContent.jsx)（分析区嵌入与独立弹窗共用同一份实现）、[PointProbeModal.jsx](../frontend/src/pages/DataOverviewPage/PointProbeModal.jsx) |
| 观测轨填充 | [railCurveFill.js](../frontend/src/pages/DataOverviewPage/workbench/railCurveFill.js)（曲线下方的数值填充：本轨极值 → 数值分档实色色块、深色主题最低亮度抬升、缺测分段）、[colormaps.js](../frontend/src/utils/colormaps.js)（`makeColorSpec` / `getRgb`：色阶名与发散色阶判定） |

`EarthOverview/EarthOverviewScene.jsx`、`EarthMap2D.jsx`、`EarthTimeline.jsx`、`EarthSeriesPanel.jsx`、`useEarthOverview.js`
作为**兼容保留实现**仍在仓库中：观测台直接复用二维地图；逐日时间轴和序列面板仅供兼容二维场景使用，当前观测档由左右两条观测轨承载日期和点位曲线；
`EarthOverviewScene` 本身不再由页面默认挂载，保留以便对照二维基础实现。

`SidebarMenu.jsx` 与 `DetailPanel.jsx` 也不再由页面挂载：前者只重新导出数据源选择控件
（`SourceScopePicker` / `OzoneSourceModePicker`）并提供最小兼容渲染器，后者的卡片正文改由
底部分析区承载；`workbench/OverviewAnalysisPanel.jsx` 同样退出页面主路径，仅保留给未迁移的调用方。

`TimelineController.jsx` 保留为兼容的横向 Ls 控件，火星观测页当前挂载 `MarsObservationRail`。
`ObservatoryTimelineRail.jsx`、`PointTrendRail.jsx` 与 `PointRailPlaceholder.jsx` 保留旧地球轨道实现，当前页面改由 `EarthObservationRail` 承载；`observatoryTimelineRail.js` 中的日期纯函数与基础 CSS 仍被复用。

### 哪些是真正共用的、哪些是星球专属

两个星球共用外壳、工具栏与分析视图；观测轨分别适配地球日期与火星 Ls，共用 `observatoryTimelineRail.css` 基础样式和 `marsObservationRail.css` 的卡片、曲线及响应式布局。

| 能力 | 共用实现 | 说明 |
| --- | --- | --- |
| 观测台槽位、两档尺寸预算、<1024px / 矮屏文档流 | `OverviewShell` + `observatoryLayout.js` + `overviewVisualContract.js` | Mars 与 Earth 都使用同一份外壳；观测档不预留分析区高度，分析档卸载球体，把空间留给图表 |
| 顶部条件栏与工具面板 | `ObservatoryToolbar` + `ObservatoryTools` + `ObservatoryToolParts` | 工具目录按星球与档位配置；桌面为非模态面板，窄屏使用适配深浅主题的模态抽屉；一次只开一个，Escape 或关闭按钮退出并归还焦点 |
| 左侧轨道底部的「展开分析」 | `ObservatoryToolbar.AnalysisEntryButton` + 时间组件的 `actions` 槽位 | 观测档没有常驻分析区，入口与播放/时间同属观测操作带 |
| 地球全球/单点观测轨 | `EarthObservationRail` + `earthObservationRail.js` + `observatoryTimelineRail.js` | 同一组件区分左右职责；原生竖向滑块按实际 Earth 时间轴定位，纯函数负责年度筛选、缺测断点和**每轨自己的数值范围**（`EarthWorkbenchScene` 分别传入 `railDomain` 与 `pointRailDomain`）；右轨常驻，数据来自既有 `pointSeries` |
| 火星全球/单点观测轨 | `MarsObservationRail` + `marsObservationRail.js` | 同一组件按左右职责配置；Ls 轴与单位共用、数值刻度各用本轨极值（`railDomain` 只传本轨序列），原生竖向滑块支持拖动与键盘，上为较早 Ls、下为较晚 Ls |
| 分析主题、条件与结果 | `AnalysisDock` + `AnalysisBoard` + `analysisGuidance.js` | 默认主题三图、统一放大/返回和条件槽位；单项下拉保留原有结果与 AI，点位/时间序列留在观测档 |
| 分区标题 | `SectionLabel` | 两侧同一份实现 |
| 模式标题与说明文案 | `overviewChartLayout.MODE_DEFS` | 唯一来源；面向用户的分组名由 `observatoryLayout.buildObservatoryGroups` 统一改写 |
| 卡片状态渲染 | `OverviewCard` | `idle/loading/ready/unsupported/error` 五种状态；不可用卡片不挂载请求 |
| 卡片正文的图表交互 | `WORKBENCH_PLOT_CONFIG` + `.overview-card-plot` | Plotly 模式栏（导出 PNG/SVG、缩放、框选、自动缩放）、滚轮缩放、右下角模式栏样式；分析区主图上的模式栏由 `[data-dock-main] .modebar` 收成图内右上角横向一行 |
| 图表变量选择 | `EarthWorkbenchScene` 的分析条件；`MarsAnalysisConditions` 与 `MarsAnalysisProvider` | 两颗星球主路径的适用变量集中到左侧；Mars 组合主题与单项分别保存条件，旧组件独立使用时保留内部选择器 |
| 横向色标（含单位标题） | 各热力图视图 | 与 Mars 季节变化热力图一致 |
| 判定“同距取较小下标”“不夹取、不环绕”的最近单元规则 | `OverviewAdapter.nearestGeometryCell` | 前后端与二维地图使用同一规则 |
| 请求取消、过期回包拒绝、播放步进 | `useOverviewController` / `overviewRequestCoordinator` | Earth 使用共用 controller；Mars 沿用页面计时器与 context，由观测轨控制播放，共用的是身份与取消语义 |
| 时间模型、单位、几何、能力、卡片目录 | `marsOverviewAdapter` / `earthOverviewAdapter` | **星球专属**：MY/Ls 与 ISO 日期、火星单位与原始单位、火星全球插值几何与 v2 真实单元几何 |
| 数据源区块内容 | `MarsSourceControls` / Earth「数据源」面板 | **星球专属**：Mars 提供 MCD/OpenMARS/NOMAD 与上传来源选择；Earth 只有一个已注册发布，因此展示数据集身份、网格与聚合说明 |
| 三维渲染细节 | `SphericalFieldCanvas` + `sphericalRegionalParticles.js` | **共享视觉、专属数据层**：两者共用连续球面材质、粒子密度/尺寸、相机基线与值到径向高度的表达；Mars 保留纹理与 `buildSeasonalSunLight(Ls)`，Earth 使用本地 NASA Blue Marble 彩色影像 CanvasTexture、真实单元粒子、默认自动旋转与固定展示照明 |
| 卡片正文的指标定义 | 各星球视图与后端服务 | **星球专属**：Mars 用 Ls 与火星纬带；Earth 用 ISO 日期、单元面积权重与 5 条纬带 |

仍然不同的地方（有意保留，不是复用缺口）：

- Mars 的球面是纹理 + 粒子层，Earth 是本地 NASA Blue Marble 彩色纹理 + 真实网格单元采样粒子 + 可切换海岸线；两者共用相机、控制器、拾取与粒子材质接口。
- Mars 的“数据源”面板有个人上传来源与 NOMAD/OpenMARS 臭氧对照选择；Earth 没有第二个来源，因此只展示当前发布信息。“图层”均用于数据场和地理参考的可见性。
- 两颗星球的组合看板共用 `AnalysisBoard` Plotly 视图，数据模型分别处理科学口径。单项分析继续使用既有 `OverviewCharts/*`（Mars，自行请求）和 `EarthResearchViews.jsx`（Earth），尚未统一为原方案的七个共用视图。
- 观测档不挂载常驻分析内容，使用明确入口切换到分析档；共用分析组件的观测分支仅保留入口。

## 适配器契约

```js
{
  planet: 'earth' | 'mars',
  sourceId,
  sourceFingerprint,
  time: { kind: 'iso-date' | 'mars-year-ls', start, end, values, years, step, stepUnit, calendar },
  variables: [{ id, label, unit, colormap, centeredOnZero, range }],
  geometry: { latCenters, lonCenters, latBounds, lonBounds, wrapLongitude, shape },
  capabilities: { field, playback, pointProbe, polar, diurnal, researchSuite, spatialDiagnostics, aiInsight },
  cards: { temporal: [...], drivers: [...], dynamics: [...] },
  defaults: { variable, value, mode },
  resolve, loadField, loadRegionalSeries, loadPointSeries,
  validateField, validateRegionalSeries, validatePointSeries,
  normalizeField, normalizeRegionalSeries, normalizePointSeries,
  cellForPoint, loadCard,
}
```

统一卡片状态固定为 `idle / loading / ready / unsupported / error`：

- `unsupported` 必须带稳定原因码，只解释限制，**不挂载数据请求**；
- `error` 带 `errorCode` 与重试入口，不把失败渲染成零值；
- `loading` 不携带上一次的 payload，旧图表不会被贴上新标题。

### 星球切换顺序

`useOverviewController` 在 `planet` / `sourceId` 变化时严格按以下顺序执行：

1. `coordinator.cancelAll()` 取消旧星球全部在途请求；
2. 清空旧场、旧区域序列、旧点位曲线、播放状态、悬停提示与已选点；
3. 重新解析数据源并载入新星球的默认变量与时间（`normalizeSelection` 拒绝跨星球选择）；
4. 递增 `cameraEpoch` 并按新的几何身份重建场景键，相机与几何一起重置；
5. 每个回包在写入前同时检查 `epoch`、通道 token 与请求身份，旧响应无法写回新星球状态。

## 三维地球

`SphericalFieldCanvas` 新增显式接口，Mars 旧调用（`fieldData`/`fieldLayers`/`showMars`/`solarLongitudeLs`）行为不变：

```jsx
<SphericalFieldCanvas
  planet="earth"
  field={field}            // { values, colorRange, unit, variable, date }
  geometry={geometry}      // { latCenters, lonCenters, latBounds, lonBounds, wrapLongitude }
  selection={selection}    // { point: { lat, lon } | null }
  lighting="fixed"         // Earth 固定展示照明
  showBaseMap
  onGlobeClick={onGlobeClick}
/>
```

- Earth 走 `buildRegionalCellGeometry()`：v2 全球 36×72 = **2592 个单元**，每格按 2×3 子面贴合球体，
  三角形从球外看为逆时针；顶点经纬度只做角度插值，格内颜色取该单元采样值，不做数值插值。
- **不跨越经度接缝**：最后一列在 +180 收口、第一列从 −180 开始，两者是不同顶点，不生成 `j=71 → j=0` 的接缝面。
- **两极封盖**：第 0 行覆盖 −90°…−85°，第 35 行覆盖 85°…90°，极点处顶点退化但仍有面，
  不会出现 NaN 或空洞；单元格坐标仍是 −87.5° 与 87.5°。
- 半径分层：连续纹理底球 0.86、数据单元基础半径 0.872、海岸线 0.878、拾取球 0.90。数据单元会按当前场值沿法向抬升，连续变量使用 `t × 0.225`，有符号风场使用以零为中心的高度。
- 颜色与径向高度共同编码数值；未着色区域表示没有数据。
- `geometry` 身份包含坐标与覆盖边界，形状相同但坐标不同会重建几何；切日只更新颜色缓冲。
- Earth 光照是固定展示照明，**不调用** `buildSeasonalSunLight(Ls)`，不画晨昏线，日期不被当作某一时刻的太阳位置。
- WebGL 不可用时自动切到二维地图并说明原因，日期、变量与点位选择保留。

球面坐标转换 `geographicToCartesian(lat, lon, radius)` 与拾取用的 `localPointToLng/LatLng` 是同一套约定，
可在 [sphericalRegionalGrid.test.js](../frontend/src/components/sphericalRegionalGrid.test.js) 中互相验证。

## Earth 分析接口

前缀 `/api/analysis/earth/overview`，公开只读。全部为同步 `def` 处理函数，由框架线程池执行 NumPy 计算。

| 方法 | 路径 | 必传参数 | 用途 |
| --- | --- | --- | --- |
| GET | `/context` | — | 数据源、时间模型、变量与单位、几何、能力、极区范围 |
| GET | `/research-suite` | `dataset_id`、`year` | 选年五变量日期×纬度场、区域/纬带曲线、极值、Z-score、相关与滞后 |
| GET | `/spatial-diagnostics` | `dataset_id`、`year`、`variable` | 选年选变量年平均空间距平与纬带 RMS/峰谷跨度 |
| GET | `/polar-dynamics` | `dataset_id`、`year` | 南北极区日平均统计与南北极对比 |
| POST | `/insight` | JSON body | 图表 AI 解读，只接受统计摘要 |

`expected_fingerprint` 为可选参数（64 位小写 SHA）；缺省时使用注册表当前已验证指纹。
传入过期指纹返回 409 `dataset_version_changed`，不会把新数据配旧元信息。

### 公共响应字段

```text
planet: "earth", dataset_id, dataset_version, dataset_fingerprint, dataset_schema
```

`/context` 额外返回 `source_meta`（来源、日平均节奏、文件数、处理说明、日历；**不含服务器路径**）、
`time`（`kind: "iso-date"`、真实首末日、天数、`years`）、`variables`（五项 id/label/units/色带范围）、
`geometry`（单元中心、单元边界、`wrap_longitude: true`）、`capabilities`、`unavailable`、
`polar_scope`（`min_abs_latitude: 60`、`sampling: "daily_mean"`、`note`）与 `limitations`。

Earth 的能力声明中 `diurnal` 固定为 `false`，原因码为 `daily_data_has_no_diurnal_samples`。

### 年度统计定义

设 `F[t,i,j]` 为该年 `T` 天的原始场，`φ[i]` 为纬度单元中心，纬度为升序。

- **季节结构** `z[i][t] = mean_j F[t,i,j]`：固定纬度上的等面积经度均值（72 个 5° 单元等权），
  标记为 `equal_longitude_mean`；横轴为真实 UTC 日期。
- **年内变化**：`np.diff(np.sin(deg2rad(edges)))` 面积权重下的全球均值，标记为 `spherical_cell_area_mean`，
  与 `/overview/regional-series` 同日同值。
- **纬带**：`south_polar` / `south_mid` / `tropics` / `north_mid` / `north_polar`，
  按**单元边界**严格划分且互不重叠；36 行分配为 6/6/12/6/6。
  响应返回每一个带**实际采样的**纬度极值与网格点数，阈值本身不是网格点。
- **季节极值**：各纬带日均曲线的 `max/min/peak_to_peak` 与日期；相同极值取最早日期。
- **相关**：同一天的区域或纬带均值组成成对序列（每天一个样本，不混入空间样本）。
  `n < 3` 记 `insufficient_samples`，零方差序列记 `constant_series` 且 `r = null`，
  不伪造 `r = 1`，也不给 p 值或因果结论。
- **滞后**：固定 −30…30 天，`r(k) = corr(driver(t), TO3(t+k))`；正 `k` 表示驱动领先臭氧 `k` 天。
  只用重叠样本，不环绕、不跨年补点；UI 不把最大 `|r|` 的滞后描述为物理响应时间。
- **空间距平** `A[i][j] = annual_mean[i][j] - mean_j annual_mean[i]`：年平均场相对同纬度经度均值的距平，
  不是多年气候距平；纬带 RMS 为 `sqrt(sum(w·A²)/sum(w))`，`w` 为该行单元面积权重。
- **极区**：`|latitude| >= 60°` 的日平均统计（均值、最低/最高与日期、峰谷差、逐日序列）
  以及南北极同变量序列的对比；纬度带没有网格点时返回 `null` 与 `grid_point_count: 0`，不填零。

所有响应禁止 `NaN`/`Infinity`；无定义的统计用 `null` 加原因码表达，原始物理值不因前端不支持而变成 0。
缺测不会被 `nan_to_num` 填零，也不会静默跳过日期。

同一公历年内保留 `2020-02-29`，不折成 360 点；两年样本只用于“年内变化”，不推断长期趋势或气候态。

### 错误协议

与二维总览一致：`{"detail": {"code": ..., "message": ...}}`。

| 场景 | HTTP / code |
| --- | --- |
| 未注册 ID | 404 `unknown_dataset` |
| 请求火星数据集 | 409 `dataset_overview_not_supported` |
| `expected_fingerprint` 过期 | 409 `dataset_version_changed` |
| 缺包/坏包/不可读 | 503 `dataset_unavailable`（含安全 `availability_reason`） |
| 年份不在发布范围 | 422 `year_out_of_range` |
| 未支持变量 | 422 `unsupported_variable` |
| `/insight` 传入 `planet != "earth"` | 422 `planet_not_supported` |
| `/insight` 摘要过大（>4000 元素或嵌套 >6 层） | 422 `insight_payload_too_large` |
| 缺必需参数、SHA 格式错误 | Pydantic 422 |

### AI 解读

`POST /insight` 的请求模型为 `extra="forbid"`，只接受
`planet / dataset_id / expected_fingerprint / year / variable / date / scope / question / locale / summary`；
`summary` 只允许 `units` 与 `cards[{card, notes, values}]`。
请求里的数字不会被采信：服务端按同一身份从已验证发布**重新计算**摘要，客户端内容只作为补充说明。
原始 36×72 场不会被接受。

返回内容固定携带 `source`、`planet`、`date_range`、`variable`、`units`、`scope`、`scope_label`、
`time_kind: "iso-date"`、`model` 与 `limitations`。未配置 `copilot_service`、缺少 `AI_API_KEY`
或外部接口失败时，返回内置的确定性摘要回答（`model: "builtin-digest"`），接口仍为 200。
解读文本只描述地球日平均数据，不出现 MY/Ls、火星沙尘或昼夜循环。

## 前端行为

- **时间**：三小时 Earth 使用带 `Z` 的 ISO UTC datetime；归档日频兼容分支使用 ISO 日期。日期/时间输入、滑块、上/下一时间步、播放走同一个选择函数。
  播放等当前帧确实展示成功后才前进（不跳时间步），到发布轴末端停止，不循环；“从首日重播”显式回到首个发布时间。
  切换变量、手动选点、页面隐藏时暂停。
- **年度**：年度选择与当前日期联动——日期跨年时年度分析自动跟随；切换年度把日期平移到目标年同月同日
  （`2020-02-29 → 2021-02-28`，不会落到 3 月 1 日）。
- **单位**：地球固定使用原始单位，全局设置中的火星臭氧/温度单位不作用于地球，地球也不做单位换算。
- **点选**：三维与二维点击都先判断覆盖范围，再用最近单元中心作为最终位置；区域外只提示、不发点位请求、不移动旧点位。
  全球网格上 ±180° 是同一子午线的等价写法，只做坐标规范化；区域发布（`wrap_longitude=false`）保持原样。
- **覆盖均值**：显示服务端返回的全球单元面积加权均值，不称“全球平均”以外的含义（不是大气总质量）。
- **卡片文案**：Earth 极区卡片为“极区统计”（不是“极区动力”），昼夜卡片显示尚未接入分析的能力说明。
  旧二维文案“区域数据”“空间点抽样”“不是全球平均”已不再用于 v2 全球网格。
- **错峰请求**：`useEarthResearch` / `earthResearchClient` 按 `(fingerprint, year[, variable])` 去重缓存，
  多张卡片共享一次年度请求；观测时间步播放不会重发年度数据；失败不缓存，可重试。
- **观测台控件位置**：参见本文开头的观测/分析操作说明。分析分类集中在 `AnalysisDock`，不再重复放进地球图层面板。

## 场景切换的开销与“无缝”程度

两个星球**互斥挂载**（不做 hidden 双挂载），因此切换必然经过“卸载旧场景 → 挂载新场景 → 取新数据”三步。
2026-09-24 用系统 Edge + `playwright-core`（1440×900，headless，后端托管 `frontend/dist`）实测点击到新场景可用的耗时：

| 切换方向 | 首次 | 再次 | 切换时发出的 API 数 |
| --- | --- | --- | --- |
| 火星 → 地球 | 1151 ms → **991 ms** | 906 ms → **763 ms** | 6 → **5** |
| 地球 → 火星 | 980 ms → **851 ms** | — | 3（火星自有接口，未改） |

（数值含 Playwright 点击的可操作性等待，真实感知时间约为其中的一半。）

### 已实现的三项改动

1. **后端启动预热 Earth 发布**：`main.py` 的 lifespan 在后台线程调用一次
    `registry.get_dataset(app.state.default_earth_dataset_id)`，把活动 Earth 发布的校验从“首次点击地球”
    挪到启动阶段，并在关闭时取消该任务。当前默认发布为三小时数据，缺包时保留不可用状态，不回退到日频。
    日志会打印 `Earth dataset prewarmed: id=... availability=available`；具体耗时随发布目录和机器而变。
2. **卡片数据按需加载 + 身份不含时间步**：折叠卡片不发请求，展开才拉取；卡片请求身份是
    `(planet, source, fingerprint, cardKey, variable, year)`，**不含当前时间步**。因此观测时间步播放时右侧
    不会再每个时间步闪一次“正在加载”，切地球的并发请求从 6 个降到 5 个（极区统计等展开时才请求）。
   `research-suite` 仍只在第一张卡片展开时请求一次。
3. **视角记忆**：`SphericalFieldCanvas` 新增 `poseKey`，卸载时把相机位置/四元数与球体组四元数
   存进模块级有界缓存（最多 8 条），同一 `poseKey` 重新挂载时恢复。Mars 与 Earth 各自一个键，
   切走再切回保持上次的旋转与缩放；点击“重置视角”调用 `dropCameraPose()` 删除该键并抑制随之
   而来的那一次保存（重置会重建画布，否则旧画布的卸载清理会把重置前的视角写回来）。

已用真实浏览器验证：火星拖拽后相机 `[0.20,-0.10,-3.74]`，切到地球再切回为 `[0.25,-0.12,-3.74]`（一致）；
地球拖拽后 `[0.39,0.20,-4.18]`，点“重置视角”回到 `[0,0,4.2]` 且切走再切回仍是 `[0,0,4.2]`。

### 仍不“无缝”的地方与可行建议

按“预计收益 / 改动成本”排序。以下均**尚未实现**，是可选后续项。

| # | 问题 | 可行做法 | 预计效果 | 成本 | 风险 |
| --- | --- | --- | --- | --- | --- |
| 1 | `research-suite` 太大：2020 年响应 **1.55 MiB**，其中 **79.7% 是 `seasonal`**（五变量 ×36×366），而卡片一次只显示一个变量 | 给 `/research-suite` 加 `variable=` 参数（或拆出 `/seasonal` 端点），只返回当前变量的季节矩阵；切换变量时再请求该变量 | 单次响应 **1.55 MiB → 约 0.56 MiB**，序列化与传输时间按比例下降 | 中（后端 schema + 前端请求键） | 低：只影响新参数，旧调用不传时行为不变 |
| 2 | 切换瞬间旧场景立即卸载，新场景要等首帧数据，中间是空面板 | 切换时先挂载新场景（`opacity:0`、`pointer-events:none`），等它报告 `ready` 后再交换可见性并卸载旧场景，用 200 ms `opacity` 过渡做交叉淡入；两场景同时存在的时间窗口内内存峰值 +1 个 WebGL 上下文 | 消除空白闪烁，感知上接近瞬时 | 中（页面层双缓冲状态机） | 中：需要保证窗口期结束后旧场景的 GPU 资源确实释放，并在低内存设备上可回退 |
| 3 | 每次回到地球都要重新取 `descriptor`（暖态约 100 ms）与 `context` | 后端给 `GET /api/datasets/{id}` 与 `/context` 返回 `ETag: <fingerprint>`，前端带 `If-None-Match`；命中返回 304 | 省掉重复序列化与传输 | 低 | 低 |
| 4 | ~~火星侧栏与右栏每次挂载重播入场动画（`SidebarMenu` 100 ms 延迟 + `DetailPanel` 0.6 s 滑入）~~ | 观测台已移除常驻侧栏与右栏，这两个组件不再由页面挂载 | 该问题随布局改版消失 | — | — |
| 5 | 切换瞬间图例/时间轴日期/球体颜色全空（控制器主动 `setField(null)`） | 每个星球在页面层保留最后一次成功的 field + geometry，切回来先渲染旧帧并标注“正在更新”，新帧到达再替换 | 新场景第一帧就有内容 | 中 | 中：必须沿用现有 `requestedValue`/`displayedValue` 区分，绝不能把旧帧标成新日期 |
| 6 | 切回后不恢复播放状态（`setPlaying(false)`） | 记住每星球离开时是否在播放，回来可选恢复 | 长时对比时少一次点击 | 低 | 低 |

不建议做的事：把两个星球改成 hidden 双挂载“常驻”。省下的 0.5–1 s 换来第二套 WebGL 上下文、
火星纹理与粒子层常驻内存，以及两套 requestAnimationFrame 同时跑——与“地球不加载火星资源”的既有边界冲突。

## 测试

后端（从 `AstraAtmos_backend/backend/` 执行，使用全新的纯英文 `--basetemp` 目录）：

```powershell
python -m pytest tests/test_earth_research_service.py tests/test_earth_research_routes.py -q --basetemp "$(Join-Path $env:TEMP ('.earth-research-' + [guid]::NewGuid().ToString('N')))"
```

前端（从 `frontend/` 执行）：

```powershell
node --test src/pages/DataOverviewPage/workbench/OverviewAdapter.test.js src/pages/DataOverviewPage/workbench/overviewRequestCoordinator.test.js src/pages/DataOverviewPage/workbench/overviewWorkbenchStructure.test.js src/pages/DataOverviewPage/workbench/observatoryLayout.test.js
node --test src/pages/DataOverviewPage/EarthOverview/earthResearchModel.test.js src/pages/DataOverviewPage/EarthOverview/earthResearchClient.test.js src/pages/DataOverviewPage/EarthOverview/earthMapGeometry.test.js
node --test src/components/sphericalRegionalGrid.test.js src/components/sphericalEarthBaseMap.test.js src/components/sphericalPicking.test.js
npm run build
```

本次改版新增两个测试文件：`observatoryLayout.test.js` 覆盖两档垂直预算守恒、窄屏与矮屏转文档流、
未知档位回落 `observe`、缺失测量不产生 `NaN`、观测档不预留分析区、主图选择保留原选择并解释不可用目录；
`observatoryTimelineRail.test.js` 覆盖真实日期轴（闰年 366 / 平年 365）、位置按真实日历日而非数组下标、
月份刻度单调且落在月初、年内偏离带按年内均值中心化并归一化、缺失日不参与极值与路径生成、
`null` 年份/日期不抛异常、读数查值不臆造。

覆盖重点：星球切换取消与过期回包拒绝、Earth/Mars 时间模型与单位隔离、Earth 不使用 Mars 光照/Ls、
全球地图不再裁成旧区域、卡片 `unsupported`/`error` 状态、经度接缝与两极拾取、普通点选与播放步进、
2592 个单元几何与 `key` 身份、AI 摘要字段与 `extra="forbid"` 契约。

## 验证记录

### 2026-09-26 观测轨刻度改为每轨独立

用户反馈「取点不同，全球均值曲线形状会变」。核对结论：全球均值的数值本来就只跟行星/数据集/年份/变量有关（地球取 `regionalSeries`，火星取固定参考坐标的 `series.globalMean`），变的是刻度——两条轨原先共用 `railDomain(global, point)` 的并集，点位的极值会撑开整条尺子，把全球曲线压扁。按用户选择改为**每轨独立刻度**。

- 前端 `node --test`（工作目录 `frontend/`）**490 项全部通过**，`npm run build` 生产构建成功；`git diff --check` 无空白错误。新增/改写断言：同一条全球序列在不同点位下路径逐字节相同、每条轨的极值映射到基准线 22 与最大幅度 76、同一 Ls/日期在两条轨上 y 坐标一致。
- 真实浏览器验收（Edge headless + CDP，1440×900，后端托管 `frontend/dist`，观测档默认地球）：
  - 未选点时左轨刻度 `278↔290 DU`；球面点击 `7.5°, -7.5°` 后左轨路径与刻度逐字节不变，右轨出现曲线且刻度为 `239↔282 DU`。
  - 换第二个点位 `-27.5°, 37.5°` 后左轨仍逐字节不变（`278↔290 DU`），右轨刻度更新为 `245↔304 DU`，右轨曲线横向铺满 22→76（按自身极值归一化）。
  - 日期从 `2020-01-01` 改到 `2020-01-31`：刻度轨日期更新、本轨刻度不变；全年曲线是整年序列，不随展示日期改变。
  - 火星观测档同一口径复验：左轨刻度 `2.04↔5.37 μm-atm` 在两个点位下逐字节不变，右轨分别 `0.19↔1.81`、`0.18↔2.46 μm-atm`；两轨 `data-rail-ls` 读数一致（`0.34046`），说明 Ls 轴仍严格共用。
  - 两轮均无控制台错误、无 JS 异常；轨道刻度提示与 `title` 已改为「本轨数值范围…两轨共用日期轴/Ls，数值刻度各自独立」。
- 验证用临时 CDP 脚本与输出文件不属于仓库内容，验证结束后已逐个删除。

#### 同轮追加：曲线下方按数值填充

用户要求两条轨的曲线下方按数值高低填充颜色。实现见 `railCurveFill.js`；两条轨共用同一套几何（基准线 22 / 最大幅度 54 / 高度 460）。

- 前端 `node --test` **498 项全部通过**（新增 8 项填充测试：分段闭合法、stop 顺序与色阶来源、深/浅主题取色段、本轨极值铺满、退化刻度与平坦序列不填充），`npm run build` 成功。
- 浏览器实测（Edge headless + CDP，1440×900，后端托管 `frontend/dist`）：地球与火星、选点前后的四条轨均渲染 `observatory-rail__fill`，各 16 个渐变 stop，深色主题下 `460 → rgb(167,47,94)`、`0 → rgb(252,255,164)`，渐变轴为 `userSpaceOnUse` 的 0→460，与轨道几何一致；无控制台错误。
- 三点实施中发现并修掉的问题：① 直接按「本轨刻度 → 整条色阶」填充时，inferno 低端近黑，黑底上等于没有填充（首轮截图两轨都看不出填充）；② 改成抬亮暗端后整条色阶被洗成同一片灰（`rgb(105,105,107)`），颜色不再随数值变化；③ 改成反转色阶方向后窄范围又整片落在色阶最亮的淡黄段。最终方案是「本轨极值铺满 + 只取当前主题可用色阶段」，既保证每个数值看得见，也保证低值到高值有色相差异。
- 验收截图：`_shot/fill-earth-rail-*.png`、`_shot/fill-earth-point-rail-*.png`、`_shot/fill-mars-rail-*.png`、`_shot/fill-mars-point-rail-*.png`（临时 CDP 脚本已删除）。

#### 同轮修复：填充左边界不是竖直的

用户复看后指出「色块的左边界不是竖直的」。核实结论：**是真实缺陷，出在填充路径的闭合方式上**。

- 原实现是 `M ${low} ${y0} ${curve} H ${low} Z`。`Z` 画的是「末点 → 路径起点」：起点在曲线上（x=59.43）、末点在基准线上（x=22），这条自动闭合线就是一条斜边，把色块左边界从年中斜到年末。实测该斜边覆盖 x 59.43 → 22、y 0 → 460，正是用户看到的那条斜线。
- 修法：四条边全部显式写出 —— `M low y0 H x0 <曲线> H low L low y0 Z`，不再依赖 `Z` 补边。修后基准线上的命令依次是「起点 → 曲线末行投影 → 显式竖边 → 闭合回起点」，任何时刻 x 都不小于基准线。
- 定位方式（可复现）：把浏览器里真实的 `d` 导出，按 `M` 拆成子段分别染色渲染，斜边只出现在含曲线的第二段；再用命令级解析逐点求 x 范围，确认自动闭合线是唯一越界来源。
- 回归测试：`railCurveFill.test.js` 新增「左边界必须留在基准线上」用例，逐命令断言 `x >= 22`，并检查基准线上的落点序列为 `[y0, y末, y0, y0]`。前端全套 **499 项通过**，生产构建成功。
- 同轮还发现一个**已知约定**（非缺陷）：缺日造成单点片段时（采样间隔 >1 天），填充跳过该片段不产生面积，曲线本身仍是孤立的 `M` 点；测试里已用 4 个稀疏采样点固化这一行为。

#### 同轮返工：颜色变化看不出来 → 改用数值分档实色

用户反馈「我要的颜色变化也没有实现」。核实结论：**颜色确实在变，但肉眼分辨不出**，而且填充本身是纯色。

- 逐行量渲染像素（真实填充、避开曲线描边）：整条填充的中位色只有 `rgb(89,29,56)` → `rgb(96,32,52)`，亮度 44→47。原因有两层：① 数值范围窄（臭氧 278–290 DU），色带只用到很小一段；② 填充是 SVG 渐变，实测**渲染成单一颜色**（探针矩形自上而下亮度跨度 0，与 16 stops 的渐变定义无关，2 stops 同样复现）。
- 返工方案：放弃 SVG 渐变，改为**数值分档 + 实色矩形**。轨道高度等分 14 档，每档一个实色，颜色取该变量色带在本轨数值区间上的位置；色块左边界固定基准线、右边界取该档内曲线最右点。深色主题最低亮度抬升改为**三通道同偏移**（原按比例幂变换会把 `rgb(0,0,4)` 变成 `rgb(0,0,231)` 的电光蓝且亮度仍只有 26）。
- 返工后的实测（真实应用、地球全球均值轨）：14 条色块、14 种颜色，档高各 33px 首尾相接无缝，左边界全部为基准线 22；颜色自顶向下为 `rgb(252,255,164)` → `rgb(247,151,28)` → `rgb(200,65,74)` → `rgb(111,25,109)` → `rgb(67,51,92)`，与"高值偏黄、低值偏暗"一致。
- 前端 `node --test` **498 项通过**（填充相关 8 项：档位无缝铺满、颜色随数值单调、数值区间首尾相接、宽度跟随曲线包络、亮度抬升不改色相差、退化输入不产生色块），生产构建成功。
- 验收截图：`_shot/final-band-earth-*.png`、`_shot/final-band-earth-point-*.png`、`_shot/final-band-mars-*.png`。
- 排障记录（避免重走）：定位过程中临时 CDP 脚本自身出过两个假信号 —— `rate(f)` 返回字符串却又套 `rgb()` 导致 `rgb(rgb(...))`，以及把亮蓝曲线描边当成填充像素；两次都让"渐变是否生效"的结论不可信。最终结论以**计数扫档 + 逐行中位色**的像素测量为准。

#### 同轮再返工：色条不随数据变化 → 颜色锚定绝对数值范围

用户反馈「这色条不会变化」。核实结论：**属实，而且是最根本的设计错误**。返工版仍把**本轨自己的最小/最大值**映射到色带两端，于是位置决定颜色——实测四种情形（臭氧未选点 / 臭氧已选点 / 臭氧点位轨 / 温度点位轨）的颜色**一字不差**：

```
rgb(252,255,164) → rgb(171,49,91) → rgb(60,60,64)     四条轨完全相同
```

色带因此退化成与数据无关的装饰性"彩虹楼梯"。

- 修法：颜色改由**变量色标的绝对范围**（`colorRange`，来自球体图例的 min/max）决定，`buildCurveFillBands` 增加 `colorRange` 入参，缺省时才回退本轨范围。地球从 `controller.field.colorRange` 取值，火星从 `sceneModel.layers[0].minVal/maxVal` 取值；火星的 `makeColorSpec` 不再吞掉 range，改为把 `mode` 与 `colorRange` 分开传给轨道。
- 返工后实测（真实应用）：臭氧 278–290 DU 落在色标 250–330 的下段 → `rgb(160,43,98)` 系；切换温度 286–302 K → `rgb(248,162,31)` 系（整段变橙）；同一变量下单点轨 239–282 DU → `rgb(152,39,100)`→`rgb(116,26,107)`。选点不改变全球轨颜色，因为全球序列本身没变——这是正确行为，不再是缺陷。
- 新增/改写测试：绝对范围锚定（两条轨同值同色、越界钳制到端点）、数值整体上移颜色必须改变（回归本条缺陷）、缺省 `colorRange` 的回退行为。前端 `node --test` **500 项全部通过**，生产构建成功。
- 顺带修正一处易错 API：`curveValues` / `buildCurveFillBands` 的 `domain` 必须是 `[min, max]` **数组**（与 `earthRailDomain` / `railDomain` 一致），传对象会返回空数组；参数 `low` 更名为 `baseline`，避免与"数值下限"混淆。

#### 同轮定稿：颜色锚定改为「按本轨数值范围铺满」

用户追问「那这个色表的意义在哪里」，并核对后确认绝对锚定方案在均值曲线上没有信息量，最终选择按本轨范围铺满。

- 数据依据：全球面积加权臭氧全年 **277.55–294.33 DU**（跨 13 DU），而球体图例色标 `scope=dataset` 为 **94.39–546.88 DU**（跨 452 DU）——这条轨只用到色带约 **3%**，所以绝对锚定下看上去接近单色。
- 定稿实现：`buildCurveFillBands` 的 `colorRange` 入参及两个轨道的 `colorRange` prop 全部移除（不再保留无人调用的死代码），颜色一律按本轨数值范围铺满。
- 定稿实测（真实应用）：地球/火星四条轨在渲染时各产生 **14 档、14 种不同颜色**，自顶向下亮度 **231 → 60**（`rgb(252,240,133)` → `rgb(63,56,77)`），即整条色带被用满；同一变量下全球轨与单点轨各自铺满。
- 前端 `node --test` **501 项全部通过**，生产构建成功。测试固化了本方案的取舍：两条轨各自铺满、颜色不可跨轨比较、档数值区间随数据变化、`domain` 传对象返回空。
- 验收截图：`_shot/plan1-earth-0.png`、`_shot/plan1-earth-1.png`、`_shot/plan1-mars-0.png`。

### 2026-09-26 主题看板与观测控件整理

- 前端全套 Node 测试 **481 项通过**（含两颗星球看板模型的 30 项测试），生产构建成功。模型回归覆盖日期/Ls 配对、缺测、温差、输入不变及小数常量不产生假 Z-score/相关系数。
- 浏览器检查 Earth/Mars 三个主题、变量和年份切换、全部单项入口、18 张图分别放大/返回；放大未增加分析请求。Escape 可在焦点移到筛选区后返回看板，打开数据源面板时先关闭面板。
- 两颗星球的三个主题逐项模拟 503，均能重试恢复；Mars 旧变量请求晚到的失败不覆盖新图，成功但为空的响应展示三个空态。
- 桌面 1440px 检查中文深色和英文浅色（120% 文字），纬带五个标签完整展示，热力图使用横向色标。曾完成 390px 基础检查；用户随后明确桌面优先，后续不再进行手机和平板专项适配。
- 桌面检查地球紧凑图例：有效格点数 2592、臭氧 DU 原始单位、风分量中值 0、色带低值在左高值在右。火星四种臭氧模式可在“数据源”中切换；“图层”的数据场、经纬网和火星贴图开关可独立操作，“显示”保留两个旋转/手势开关。
- README、本文与二维协议的当前入口说明已同步；后端科学接口未修改，个人上传来源与外部 AI 服务未在本轮重新验证。

### 2026-09-26 分析选择与操作修复

本轮已验证 Earth/Mars 选择季节图后不再回退、Earth 适用条件显示、关系图直接切纬带、标准化默认值与单位、不可用原因、数据源面板与关闭后的焦点恢复，以及 Earth 模拟一次 503 后点击重试恢复图表。检查了中文深色、英文浅色、120% 文字和 1440px / 390px 视口；窄屏导航不再遮挡数据源入口，页面无横向溢出。前端 Node 测试与生产构建通过。以下保留的 2026-09-25 尺寸和布局记录是当时版本的历史验证，以本文开头为当前控件约定。

同日补齐 Mars 的条件归位、条件记忆、图表布局和异常恢复，并完成以下验证：

- 前端 Node 测试 **451 项通过**，生产构建成功；后端 `/health`、前端首页和前端代理 `/api/datasets` 均返回 200。
- 逐一打开 Earth 与 Mars 各十个分析选项，确认图表或能力说明；Mars 多图结果没有嵌套滚动框。变量和纬带在切换图表、分类、视图和星球后恢复，空间诊断地图与指标使用相同变量和单位。
- 对 Mars 十张活动图表逐一模拟 HTTP 503，均显示失败提示并可点击重试恢复；另验证成功但为空的响应仍显示空数据，以及旧变量请求晚到的失败不会覆盖新变量图表。
- 在 1440px 与 390px、英文浅色及 120% 字体下检查环境因子、变量关系、空间诊断，无页面横向溢出、图表越界或嵌套滚动；桌面与窄屏的数据源面板均可开关。中文深色下检查观测三项工具及分析昼夜图的 Ls 操作。

### 2026-09-26 地球观测轨视觉统一验证

- 前端 `node --test` 490 项通过，最终局部调整后的日期纯函数与外壳结构测试 16 项通过；生产构建与 `git diff --check` 通过。
- Edge 浏览器实测日期拖动、方向键、PageUp、Home/End、2020-02-29 切至 2021-02-28、播放/暂停与从首日重播；球面选点后右轨出现曲线，左右共用范围，清除恢复占位，切换温度保留 K 单位。
- 检查中文深色、英文浅色及 120% 字号，1440×900、1440×600、390×844 视口；两轨对齐，矮屏保留三列，小屏两轨并排且无页面横向溢出。控制台无错误；后端健康、前端首页和 `/api/datasets` 代理均返回 200。

### 2026-09-25 观测台改版验证（历史记录）

2026-09-25 完成「行星观测台」改版并复验。本轮与 2026-09-24 记录的差异：布局从三栏改为观测台，
其余数据、单位、能力边界未变。

- **前端测试**：`node --experimental-test-isolation=none --test`（工作目录 `frontend/`）全量 **446 项通过**；
  `npm run build` 成功（Vite 生产构建，`dist/assets/index-*.js` ≈ 6.6 MB）。
- **真实浏览器验收**（Chrome 153 headless + CDP，1440×900，后端托管 `frontend/dist`）：Earth 观测/分析、
  图表切换（季节结构 / 影响关系 / 变量相关性 / 空间诊断）、三个设置面板（图层 / 点位 / 显示）、
  逐日步进与播放、Mars 观测/分析（变化规律 / 变量关系 / 空间诊断）、Mars 图层多源 / 验证 / 差值切换、
  Mars 数据源与显示面板、时间播放，全部无控制台错误、无 `Network.loadingFailed`。
- **尺寸预算实测**（1440×900、导航 70px）：观测档左轨 124×766px、画布 1186×766px、右轨 124×766px
  （分析区与底部时间轨道两行都不渲染，画布因此吃掉全部剩余高度）；分析档不渲染球体，
  底部水平轨道 1434×64px、分析区 1434×702px，主图 1396×496px（分析内容宽 1434px 的 97%）。
  （主图高度与下方辅助面板的排布随后按用户反馈改过，当前值见下方「分析档版式整理」一节。）
  各图类型（年内变化、季节结构、季节极值、环境因子、极区、辐射关系、相关、耦合、空间距平）
  都按同一高度渲染，不再有的图停在 320px、下方留白。
- **点位年变化轨实测**：未选点位时右栏为占位说明；在球面点击后出曲线（坐标 `-2.5°, 7.5°`、
  当前值 244 DU、年内范围 244~281），拖动左轨日期后右轨当前值与横标同步更新（261 DU），
  点月份标签可跳到该月 1 日，点关闭按钮回到占位；横向溢出始终为 0px。
- **观测刻度轨实测**：拖拽轨道可改日期（拖动到轨中 → `2020-07-02`、第 184 天、全球均值 285 DU），
  `aria-valuetext` 为「2020-07-02，第 184 天，臭氧柱总量 285 DU」；月份标签只显示季度首月；
  切到分析档后刻度轨卸载、水平轨道接管，返回观测档时日期选择保持（`2020-07-02`）。
- **观测档内容边界**：工具栏在观测档只渲染「显示」一个工具入口，分析分组 / 图层 / 点位面板不出现；
  `data-observatory-view="observe"` 时分析区实测 `0x0` 且不可见，切到分析档后三个入口与主图一起出现。
- **响应式**：390 / 768 / 1024 / 1280 / 1920 五种宽度下页面横向溢出均为 0px；
  宽度 < 1024px 或工作区 < 576px 时进入文档流布局，观测档球体 420px、分析档 280px。
- 验收截图保存在工作区 `<工作区>\_shot\`（`rail-*.png`、`no3d-*.png`、`transparent-*.png`、`v3-*.png`、`v4-*.png` 等，含分析档版式整理前后的对照）；临时 CDP 脚本不属于仓库内容，验证结束后逐个删除。
- 本轮验收发现并修复的真实缺陷：布局尺寸回灌自身高度导致工作区越排越高（改为「窗口高度 − 实测导航高度」并钉死 Grid 行高）；
  `toolsSlot` 在 `toolContent` 之前求值触发 TDZ 崩溃（地球场景整体白屏）；
  `flowSceneHeight` 常量缺失使文档流下的画布高度变成 `undefined`；
  观测模式紧凑预览的 Plotly 容器拿到百分比高度而解析为 0（改为明确的像素高度）；
  工具栏在桌面换行把画布挤小（改为不换行 + 状态摘要独立 flex 项）；
  观测档仍预留 192px 分析区并渲染紧凑预览（按用户反馈改为整行收起）；
  观测刻度轨：`yearProgress` 在 `year` 为 `null` 时穿过空值判断并对 `null` 日期调 `slice` 导致整页崩溃
  （改为先校验年份是整数）；偏离带把 `Number(null)` 当成 0，会把缺失日画成零值点（改为只接受真正的数字）；
  轨道上的 MUI Slider 被压成 4px 宽导致无法拖拽（改为铺满轨道宽度，视觉轨道由 SVG 承担）；
  **观测档底部残留一条无内容的着色横条**：底部时间轨道行高度已压成 0，但该网格单元仍在渲染并画自己的
  面板底色与上边框（改为观测档整行 `display: none`；分析区同一问题也已修）。

### 2026-09-25 分析档版式整理（同日晚些时候，按用户反馈）

用户反馈「分析页面乱七八糟」，本轮只改分析档的版式与信息层级，不动数据、单位、能力与请求路径。

- **信息各出现一次**：分析区头部只留「范围」（地球上是在哪个纬带，火星上是哪个火星年与 Ls）。
  年份与数据源在条件栏已有同名同值的选择器，不再抄进分析区；变量只由卡片里的药丸行表达，
  `CurrentVariableLine`（「当前变量: X (单位)」）整行删除，规则写进
  `overviewWorkbenchStructure.test.js`（禁止该标识与中英文案再次出现）。
- **一行放完所有辅助内容**：主图下方原本竖着堆「点位/序列」与「AI 解读」，竖排时后一块落到第一屏之外，
  折叠线上只剩半张卡片。现在两者并排（`AnalysisDock` 的 `.dock-bottom`，2 格 `minmax(0,1fr) + minmax(300px,460px)`），
  主图高度 = 分析区高度 − 181（头部 + 说明行 + 注脚）− 下方预留（两块并排 256px / 单块 140px），
  常量在 `observatoryLayout.js`（`dockMainChartHeight(dockHeight, { reserve })`）。
- **序列面板压缩**：摘要从卡片改成一行（变量名 + 当前点位值 + 覆盖区域平均 + 已展示日期），
  两条曲线高度 200 → 132px，因此整块能落进预留行；点位未选时仍是占位说明，不臆造曲线。
- **Mars 的 AI 解读改成内嵌面板**：`AICopilotWidget` 新增 `embedded` 形态（与地球 AI 面板同位置同版式）。
  它原本在分析区里是 `position: fixed` 的浮动入口，而锚点变量 `--overview-scene-right` 已经不存在，
  整块退化成 0 高度、按钮飘到画面左下角；内嵌形态不再做 fixed 定位，也不在无球体时自动弹气泡。
  面板里的目标图表名也从「未命名图表」修正为当前主图（页面把 `selectedCard` 作为 `cardKey` 传入，
  上下文的 `expandedCard` 已无组件写入）。
- **主图模式栏**：Plotly 默认把模式栏竖成一列压在图的右缘，且 `bottom` 定位会落进说明行、压住「原始单位」药丸。
  现在统一收成图内右上角横向一行、低对比度（`overviewWorkbench.css` 的 `[data-dock-main] .modebar`）。
- **浅色/英文复查**：`theme=light` + `language=en` 下分析区头部为 `Scope Global`、AI 面板与图例为英文，
  横向溢出 0px；窄屏 1000×820 进入文档流后分析区不再钉高度（正文 970×807，溢出 0px），
  下方两格并排改为按内容高度排布。

实测（Chrome 153 headless + CDP，1440×900，深色中文、Earth 分析档）：

| 部位 | 实测 |
| --- | --- |
| 分析区头部 | 1402×81（固定），右侧只有「范围 全球」与「返回观测 ↙」 |
| 主图 | `[data-dock-main]` 1402×347，Plotly 1396×265，模式栏 194×24 落在图内右上角 |
| 下方一行 | 1402×244，两格：序列 932×244、AI 460×244 |
| 分析区正文 | 611/611，滚动溢出 **0px**（点位未选与已选点位两种状态都是 0） |
| Mars 分析档 | 主图 1402×437（Plotly 381），AI 内嵌面板 1402×154，正文溢出 **0px** |
| 横向溢出 | 1440×900 与 1000×820 都是 0px |

本轮缺陷：Mars 分析档的 AI 入口整块 0 高度并飘到左下角（`--overview-scene-right` 未定义使 `right` 无效）；
分析区头部重复表达年份与数据源；主图下方竖排导致折叠线切在卡片中间；
序列面板 200px 曲线 + 卡片式摘要把预留行撑高；Plotly 模式栏竖排压住图与说明行。

### 2026-09-24 记录（布局改版前的三栏工作台）

2026-09-24 在本地实际执行，使用 conda 环境 `AstraAtmos` 的解释器与系统 Edge（`playwright-core`，headless）。

- **后端测试**：`tests/test_earth_research_service.py` + `tests/test_earth_research_routes.py` **92 passed**；
  `tests/test_earth_overview_service.py` + `tests/test_earth_overview_routes.py` + `tests/test_dataset_registry.py` + `tests/test_dataset_routes.py` **152 passed**（合计 244）。
- **真实 v2 数据只读核对**（`earth_merra2_daily_v2`，发布指纹 `8871f7ef865c…`）：
  `/research-suite` 2020 返回 366 天、纬度 36 行、季节矩阵 36×366、纬带行数 6/6/12/6/6、作用域 global+5 带；
  2021 返回 365 天；`/spatial-diagnostics` 返回 36×72 距平且色带以 0 对称；`/polar-dynamics` 返回南北极各 6 行。
  单个年度 suite 的 JSON 约 **1.5 MiB**（服务端实测 1 529 493 字节，低于 4 MiB 目标）。
- **HTTP 协议核对**：`/context`、`/research-suite`、`/spatial-diagnostics`、`/polar-dynamics` 均返回 200；
  `/insight` 在未配置外部 AI 时返回 200 且 `model=builtin-digest`，回答包含星球、数据源、真实日期范围、变量与单位，且不含 MY/Ls 或沙箱字样；
  `planet=mars` → 422 `planet_not_supported`，`year=2019` → 422 `year_out_of_range`，
  过期指纹 → 409 `dataset_version_changed`，火星数据集 ID → 409 `dataset_overview_not_supported`。
- **前端测试与构建**：`node --test` 全量 **415 项通过**；`npm run build` 成功。
- **真实浏览器验收**（Edge + `playwright-core`，1440×900，headless，后端托管 `frontend/dist`）：**15/15 项通过**，
  该次记录覆盖当时默认火星、Mars→Earth→Mars 连续切换（互斥挂载）、Earth 三栏工作台、五个变量与原始单位；当前入口默认地球，
  2020-01-01→2021-12-31 播放（含跨年）、赤道/经度接缝 ±179.9°/南北极点选返回最近单元中心（同距取较小下标）、
  三维球体点击选点、三种分析模式、2020 与 2021 年度分析各渲染图表、极区统计范围与日平均限制、
  昼夜卡片能力说明、AI 解读、Earth 阶段不请求 Mars 分析接口/火星纹理/在线底图、切回火星无回归、
  地球阶段控制台无新增错误。该次验收针对**三栏布局**，其固定栏宽断言已由本计划替换为观测台行为断言。
- **切换计时与视角记忆**：`switch-timing.mjs`（点击到新场景可用）与 `pose-debug.mjs`
  （相机位置在切走切回前后一致、重置视角后不再复活旧视角），见上文“场景切换的开销与无缝程度”。
- **真实 v2 数据规模**：`probe_suite_size.py` 实测 `/research-suite` 2020 响应 1.55 MiB，
  `seasonal` 占 79.7%，服务端计算仅 63 ms —— 瓶颈是序列化与传输，不是 NumPy。
- 验收脚本与截图保存在工作区 `<工作区>\.workbench-verification\`（`workbench-verify.mjs`、`report.json`、`screenshots/`），运行产物不属于仓库内容。
- 该轮验收发现并修复的真实缺陷：共用控制器缺少 `cardUnsupported` 导入导致地球场景崩溃；
  分析栏 flex `min-height:auto` 导致卡片列表覆盖下方面板；底部时间轴跨满视口遮挡右侧按钮；
  重复选择同一变量会清空场景；Plotly 以日期字符串作为散点颜色导致 `<rect height="NaN">`；
  **单元顶点色被重复归一化（0..1 又除以 255）导致三维地球只有近黑底色**；
  AI 摘要嵌套超过服务端 6 层限制被 422 拒绝。

### 历史验证的边界与待验项目

本节保留 2026-09-25 验证边界；2026-09-26 对分析档及观测轨补充的验证以上方记录和本文开头为准。

- 真实摄像头手势交互（首期未接入识别，仅预留 `gestureActions`）；本轮只验证了手势面板与 `sceneRef` 真实矩形映射的代码路径，未接入摄像头。
- 外部 AI 服务的真实回答质量与可用性（本环境未配置 `AI_API_KEY`，验证的是内置摘要回答与协议）。
- 本轮未登录账号，因此个人上传来源（MCD / OpenMARS / NOMAD 的「个人」分组）只验证了面板与禁用说明，未验证真实个人文件加载。
- 浅色主题与英文界面的逐屏截图（本轮以默认深色 + 中文为主，另有 200% 文字缩放下无横向溢出的检查）。
- 旧版地球竖轨未适配窄屏、火星仍使用横向时间轴的限制已由 2026-09-26 两侧观测轨替换：两星球在 390px 下均采用球体在上、双轨在下的排布，分别保留日期与 Ls 语义。

### 彩色地球底图

总览入口的星球按钮按“地球 / 火星”排列，首次进入或刷新默认展示地球。历史验收记录中的“默认火星”对应当时版本。

2026-09-24 接入本地 `frontend/public/earth/blue-marble-2048.png`（NASA Blue Marble，2048 × 1024）。影像呈现蓝色海洋、绿色植被、棕黄色干旱地表与白色冰雪，仅作静态地理参考，不代表当前数据日期的地表状态。浅色、深色主题共用同一自然色影像；地球球面的纹理 U 方向适配经度正方向，与数据单元及海岸线对齐。三维海岸线默认显示，可通过左栏开关隐藏；二维降级地图也保留海岸线。影像加载失败时保留渐变海洋，独立海岸线仍可显示。来源、许可与校验值见 [地球底图资源](../frontend/public/earth/README.md)。

## 后续计划

- 手势交互接入摄像头识别，复用控制器已预留的 `gestureActions`（旋转/缩放/播放/步进/选点/重置）。
- Earth 训练与预测页（独立交付项，见 [DLinear 地球训练方案](plans/2026-09-23-earth-dlinear-training.md)）。
