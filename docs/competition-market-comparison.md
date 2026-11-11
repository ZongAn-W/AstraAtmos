# AstraAtmos 相似市场产品调研与答辩对标报告

> 用途：比赛/项目答辩时回答「有没有相似产品」「和它们比你们有什么不同」「为什么不直接用现成工具」。
> 检索日期：**2026-10-29**（Asia/Shanghai）。本文所有外部产品信息均来自当日可访问的官方站点/官方博客，**不写价格、不写用户数（机构自述除外）、不写精度数字**。
> 标注约定：**【事实】**＝官方页面或可核来源明确陈述；**【推断】**＝基于事实的对比判断，答辩时应说明是推断；**【待核实】**＝本次未取得可靠证据，答辩前需人工复核，不得当作结论。
> 基线：AstraAtmos 能力与边界以 [`README.md`](../README.md)「主要功能」「当前功能边界」为准；本报告不重复其实现细节。

---

## 0. 一页结论（答辩开场 30 秒）

**一句话定位**

> 官方产品分别占据「气候态数据（MCD）」「再分析数据（OpenMARS）」「模型生产（MCMC/MarsWRF/LMD GCM）」「地表 GIS 与三维浏览（JMARS/Treks/Eyes）」「数据归档（PDS/PSA）」；AI 气象大模型占据「固定契约的推理服务」；ML 工具链占据「单点实验追踪/渲染」。**没有一个产品把「多源行星大气数据治理 → ML 训练 → 预测/回测 → 严格对比与诊断」放进同一个 Web 工作台。** AstraAtmos 补的是这条链路，定位是**实验层/集成层**，不是数据层、模型层或业务预报层。

**三层市场结论**

| 层 | 代表产品 | 它们强在哪 | 它们不做什么（我们的空位） |
|---|---|---|---|
| 第一层 行星/火星 | MCD、OpenMARS、JMARS、Solar System Treks、Eyes、MCMC、PDS、ESA PSA | 权威数据、官方归档、行星可视化与 GIS | 无训练/预测/回测/诊断闭环；无「数据身份」治理；无多用户实验工作台 |
| 第二层 地球气象 + AI | Worldview、Giovanni、CDS/CAMS、GEE、AIFS、WeatherNext、Pangu、FengWu、FuXi、Earth-2、Aurora、DestinE、WeatherBench 2 | 数据分发、业务级 AI 预报、评测榜单 | 全部地球专属；AI 只给推理/权重；评测只给榜单，不提供换数据/换目标的实验管理 |
| 第三层 平台化工具 | W&B、MLflow、DVC、ClearML、ParaView、Panoply、VAPOR、PIE-Engine、GEOVIS Earth、ModelArts/PAI | 单点能力（追踪、版本、渲染、云算力）成熟 | 无领域语义（Ls/UTC、MCD/OpenMARS/MERRA-2）、无数据身份约束、无「同口径多模型比较」闭环 |

**三个可主张的差异点（均有产品基线支撑，不空泛）**

1. **训练—预测—回测—评估闭环**：内置时空预测模型库 + 上传自定义 PyTorch 模型契约校验/隔离执行 → 单/多模型预测 → Earth 历史回测（UTC + origin_split + DU）、Mars Ls 最近邻窗口 → RMSE/MAE/R²/MAPE/SMAPE + 误差分布 + 残差 + PFI 诊断 → 严格兼容的多模型比较。**直接同类产品 0 家提供。**
2. **可复现的数据身份与比较口径**：数据集注册表（版本、发布指纹、能力声明、`409 dataset_version_changed` 身份校验）→ 上传 `.nc` 校验与审核 → checkpoint 冻结数据身份/归一化/`split_meta` → 多模型比较强制同指纹、同测试范围、同窗口、同指标口径（不兼容返回 `409 earth_comparison_incompatible`）。这是 MCD「不担保可靠性」、OpenMARS/PDS/PSA「只交付文件」所不覆盖的**实验可信层**。
3. **中文 Web 一体化 + 教学/研究双用**：浏览器即用，三维行星球体、观测台、分析工作台、AI 解读在同一产品内；账号/审核/通知/中英文界面；开源技术栈、Windows 分发与 Linux 部署、可私有化。对照 JMARS 桌面 Java、MCD Fortran 库、ParaView 桌面单机、云平台按量计费。

**三条红线（答辩不要说）**

- 不说预测精度/收敛速度/准确率优势——真实精度与长 horizon 浏览器性能**未验收**；
- 不把「三维行星可视化」当原创——Treks/Eyes/JMARS 早已有，且是 NASA 品牌；
- 不把「有火星/地球数据」当卖点——OpenMARS/MCD/NOMAD/MERRA-2 均公开；卖点是治理与实验闭环，不是数据本身。

---

## 1. 调研范围、方法与局限

**三层范围（与用户确认的口径一致）**

1. **直接同类**：行星大气/火星气候数据与可视化平台；
2. **相邻同类**：地球大气/气象数据平台 + AI 气象预报大模型 + 数字孪生/评测基准；
3. **平台化能力**：ML 实验管理/实验追踪、科学数据可视化、国内遥感地信气象平台、云训练平台。

**条目规模**：第一层 13 个、第二层 17 个、第三层 24 个，共 54 个对象（含少量仅作参照的模型/工具）。逐条细节笔记在 `_market_research/`（工作区内部调研笔记，不随项目发布）。

**信息处理规则**

- 优先官方站点、官方博客、官方文档；媒体与第三方记录只用于交叉印证；
- 每条区分【事实】【推断】【待核实】；无法核实一律写「未公开/待核实」，**不编造价格、用户量、精度、客户数**；
- 机构自述的用户规模按「机构自述」处理，不作为市场规模证据；
- 网页为动态渲染或抓取失败的对象（如 NASA MCMC 子页、部分国内站点）明确标注，答辩前人工复核清单见 §9.2。

**局限**

- 部分产品正文无法抓取（MCMC Webserver/MGCM Visualizer、NASA Eyes 当前形态、月球与行星探测数据发布系统 URL），只能以标题级证据判断，结论保守；
- 不涉及采购金额、授权合同、真实客户案例——公开渠道无法核实；
- 本报告不评价任何产品的科学精度，只比较功能形态与定位。

---

## 2. 对标锚点：AstraAtmos 能力基线

### 2.1 一句话

AstraAtmos 是一个**行星大气机器学习实验平台**：以火星（OpenMARS 再分析 + MCD 气候模式 + NOMAD 观测）和地球（MERRA-2 三小时再分析）大气数据为对象，在浏览器里完成数据治理、模型训练、预测回测、误差诊断、多模型严格比较与科研出图。

### 2.2 能力清单（对标矩阵的事实来源）

| 能力域 | 已实现（README 可核） | 答辩时的表述口径 |
|---|---|---|
| 数据 | 数据注册表（4 个活动数据集）、版本、发布指纹、能力声明、`409 dataset_version_changed`；上传 `.nc/.nc4` 校验与管理员审核；Mars OpenMARS 27 文件/6480 时间步、MCD 气候态、NOMAD；Earth MERRA-2 三小时 2020–2021 共 5848 UTC 步、240×480/0.75°（另有 5° 训练包） | 「数据身份可追溯的接入与治理」 |
| 可视化 | Three.js 三维球体、火星 Ls 时间轴 + MCD 全球均值、地球 UTC + 面积加权均值、点位曲线；分析工作台（季节、极区、相关、距平、昼夜线） | 「行星原生 Web 三维 + 大气时空分析」 |
| 训练 | 实验中心（窗口/步长/通道/超参/划分/早停/迁移/队列/标签/日志/指标）；模型库登记 29 个标识（通用时空骨干 + 火星结构化变体，README 列 8 个代表模型）；上传 PyTorch 单文件模型契约校验与隔离 dry-run | 「选模型、换数据、设实验」 |
| 预测 | 单/多模型预测；Earth 历史回测（UTC origin_split + DU）；Mars 在 Ls 实际时间轴上最近邻选窗，步长 1–30 | 「回测而非无真值外推」 |
| 评估 | 误差分布、残差、PFI（Mars ΔR²、Earth ΔRMSE）、多模型严格兼容比较（同指纹/同测试范围/同窗口/同指标口径，不兼容 409） | 「同口径比较，可复现」 |
| 诊断/导出 | 科研图导出（PNG）并附 MATLAB 重建脚本；上限 8 模型/100 万格点/25 万散点/2400 万像素 | 「图可重建，结果可复核」 |
| 协作/交付 | 账号/JWT/邮件验证码/通知/反馈/管理员审核；中英文界面；Windows 分发、Linux 部署；SQLite | 「中文、可私有化、教学科研双用」 |
| AI | AI 对话 + Copilot（外部模型接口，无 key 时降级兜底） | 「AI 解读是辅助，不冒充预测能力」 |

### 2.3 当前边界（必须主动说明，避免答辩翻车）

- 真实预测精度、长期收敛、全分辨率长 horizon 浏览器性能**尚未验收**；演示数据不代表精度；
- 未开放无参考真值的未来外推、Earth/Mars 混合比较；PFI 是模型解释（置换重要性），**不是因果推断**；
- AI 能力依赖外部模型接口配置；无 RAG、无实时卫星接入、无 VR；
- 官方 MCD 数据发布入口未启用（`DisabledOfficialMcdSourcePublisher`）；上传数据≠默认可训练；
- 不自研 GCM，不发布业务预报。

---

## 3. 第一层：行星/火星大气数据与可视化平台

### 3.1 产品地图

| 产品 | 机构/国家 | 形态 | 与 AstraAtmos 的关系 | 威胁度 |
|---|---|---|---|---|
| **MCD v6.2** 火星气候数据库 | LMD（法）牵头，ESA/CNES/IAA/Oxford AOPP/LATMOS 合作 | Web 查询 + netCDF 下载 + Fortran/Python 库 | **数据源之一** + 数据层直接对标 | ★★★★★ |
| **OpenMARS** 再分析 | The Open University（英） | 数据发布（DOI、netCDF） | **数据源之一** | ★★（上游） |
| **JMARS** | ASU Mars Space Flight Facility（美） | 桌面 Java 行星 GIS | 行星 GIS 与三维图层最像 | ★★★★ |
| **NASA Solar System Treks**（含 Mars Trek） | NASA/JPL（美） | 浏览器 Web 3D + WMTS 瓦片服务 | Web 三维行星浏览最像 | ★★★★ |
| **NASA Eyes on the Solar System** | NASA/JPL（美） | 浏览器/客户端 3D 科普 | 公众 3D 观感对比 | ★★★ |
| **NASA MCMC**（含 MGCM Webserver/Visualizer） | NASA Ames（美） | 机构建模中心 + 模型数据服务 | 「火星气候建模」名号撞词 | ★★★★（待核实） |
| **NASA PDS**（含 Atmospheres Node） | NASA（美） | 数据归档（PDS3/PDS4） | 数据上游/治理对照 | ★★ |
| **ESA PSA**（含 NOMAD） | ESA（欧洲） | 数据归档 + 检索 | 数据上游 | ★★ |
| MarsWRF/planetWRF、LMD Mars GCM、ExoPlaSim | 科研机构/社区 | 模型代码/库 | 模型生产工具，非竞品 | — |
| SpaceEngine、OpenSpace | 商业/开源社区 | 科普可视化 | 观感参照 | — |

### 3.2 直接同类条目

#### A1. MCD（Mars Climate Database v6.2）

- **是什么**：火星气候数据库，LMD 牵头，ESA、CNES、IAA、Oxford AOPP、LATMOS 等合作；官网首页显示当前版本 **v6.2**。【事实】
- **形态**：① Web 查询界面（官网明示 "intended for moderate use only"）；② 完整版免费下载（netCDF + Fortran 访问软件 + 多语言接口示例）；③ 附带日期转换器、沙尘气候学、合成沙尘情景等工具页。【事实】
- **能力**：三维网格上的**气候平均量**（12 个年时段 × 12 个地方时的平均），含温度、风、气压、沙尘光学厚度等；无用户建模、无预测/外推接口、无评估平台。【事实/推断】
- **开放与条款**：科学用途免费，要求引用；**未经授权不得商用**；官网免责声明明确**不担保可靠性**，并警告气候平均无法代表完整变率、线性插值会低估极值。【事实】
- **与我们的关系**：AstraAtmos 火星线直接用 MCD 作为数据源，并在观测台叠加 MCD 全球均值曲线。
- **答辩用途**：这是评委最可能拿来问「有 MCD 为什么还需要你」的产品。回答框架：MCD 是**气候平均场数据库**，不是训练/预测/评估平台；我们把它当基准数据源之一，并为其数据保留可复核的身份（版本/指纹/冻结），让「在 MCD 数据上做实验」这件事可复现——这恰是 MCD 不覆盖的层。
- **不要比**：不要比气候态权威性，不要暗示 MCD 数据「不权威」；只比「实验工作流」。
- **URL**：<https://www-mars.lmd.jussieu.fr/> · 访问与条款：<https://www-mars.lmd.jussieu.fr/mars/access.html>

#### A2. OpenMARS 再分析数据库

- **是什么**：Open University 发布的火星大气再分析（**Open** access to **Mars** Assimilated Remote Soundings），把 TES/MGS 的柱沙尘光学厚度与温度廓线观测与 OU 火星 GCM 同化，标准库含 **2 小时步长**的纬向/经向风、温度（随高度）、柱沙尘光学厚度等，覆盖多个火星年（MY24-27、连续 MY28-35 等，带 DOI）。【事实】
- **能力**：只有数据 + 论文；无可视化、无建模、无预测/评估、无账号治理。【事实/推断】
- **与我们的关系**：AstraAtmos 火星线的**直接数据源**；我们把 OpenMARS 纳入注册表、指纹与上传审核，并在其上做 Ls 时间轴观测台、三维球体与训练。
- **答辩用途**：说明「公开数据不是卖点」；我们的增量是数据治理与实验闭环。若评委问数据版权，答：数据来自公开学术发布（DOI），平台只做接入与实验，不转售数据。
- **URL**：<https://data.open.ac.uk/page/oro/62713> · <https://ordo.open.ac.uk/articles/dataset/OpenMARS_continuous_MY28-35_standard_database/24573205>
- **检索状态**：openmars.org.uk 直连抓取失败，本报告以 Open University 官方记录与 DOI 为依据；数据本体许可证待核实。

#### A3. JMARS（Java Mission-planning and Analysis for Remote Sensing）

- **是什么**：ASU Mars Space Flight Facility 的行星 GIS，桌面 Java 客户端（Win/Mac/Linux），可含 Mars、Earth、Moon、Mercury、Venus 等多天体；需注册账号。【事实】
- **能力**：多天体遥感图层（THEMIS、HiRISE/DTM、CTX、CRISM、MOC、Viking、OMEGA、TES、HRSC、SHARAD 等）、二维地图 + 3D 图层、经纬网、剖面线、散点图、阈值/掩膜、陨石坑计数、数据导出到 ArcGIS/QGIS。【事实】
- **成熟度**：自称 2003 年起公开、100+ 国家使用、**6,000+ 活跃用户（机构自述，未独立核实）**；2026-10-01 发布 5.7.3。【事实/自述】
- **与我们的关系**：观感上最接近「行星三维/图层可视化 + 点位读数 + 地理分析」的产品；但 JMARS 是**地表/遥感 GIS**，不做大气变量时间序列、GCM 数据、模型训练、预测、回测。
- **答辩用途**：承认它在多天体遥感图层与任务规划上远超我们；差异在「大气时空数据 + ML 实验闭环 + Web 多用户」。**不要比多天体覆盖**。
- **URL**：<https://jmars.asu.edu/>

#### A4. NASA Solar System Treks（含 Mars Trek）

- **是什么**：NASA/JPL 的浏览器 Web 3D 行星门户（十余个天体），并提供 **OGC WMTS 地图瓦片服务**、VR 漫游、3D 打印模型导出。【事实】
- **能力**：真实航天器数据图层、3D 可视化、实时 3D 飞越、**距离/路径量测、高程剖面（可导出 CSV）、太阳高度角/方位角计算**、多投影/视角。【事实】
- **能力边界**：以**天体表面**地形/影像为主，不含大气环流/气候场、无模型训练、无预测、无指标评估。【推断：功能页目录无相关项】
- **与我们的关系**：这是「Web 三维行星可视化」卖点的正面重叠者，且是 NASA 品牌。
- **答辩用途**：把「三维可视化」表述为**整合手段**而非原创差异：Treks 做地表浏览与量测，我们做**大气变量场**的三维播放/剖面/点位，并把它接到训练与评估链路上。**不要比 VR/WMTS/多天体**。
- **URL**：<https://trek.nasa.gov/> · <https://trek.nasa.gov/mars/>

#### A5. NASA Eyes on the Solar System

- **是什么**：NASA/JPL 的沉浸式 3D 太阳系/任务可视化应用（科普为主）。【事实：页面性质与标题；当前 Web/客户端形态待核实】
- **能力**：3D 太阳系、航天器轨迹、天体浏览；无大气数据字段分析、无科研工作流、无模型训练/评估。【推断】
- **答辩用途**：用于区分「科普可视化炫技」与「科研/教学实验台」；**不要以「3D 好看」作为首要差异**。
- **URL**：<https://eyes.nasa.gov/apps/solar-system/>

#### A6. NASA MCMC（Mars Climate Modeling Center）

- **是什么**：NASA Ames 的火星气候建模中心；官方主页可访问，NASA 站内检索可见 **MCMC Webserver** 与 **MGCM Visualizer** 子页。【事实：主页 200 + 检索标题；子页正文 429/404 未核实】
- **能力**：GCM 模型研发与运行（Ames MGCM / MarsWRF 谱系），面向 NASA 任务支持与科研社区；是否向公众提供训练/评估/对比工具**未公开/待核实**。【推断】
- **与我们的关系**：定位撞词——「火星气候建模中心」vs 我们的「实验中心」。必须把口径改为「**大气数据上的机器学习实验与评估平台**」，不声称「做气候建模」。
- **答辩用途**：MCMC 是**模型生产方**，我们是**实验消费者/编排者**；我们不运行 GCM，不与之竞争模型研发。
- **URL**：<https://www.nasa.gov/space-science-and-astrobiology-at-ames/division-overview/planetary-systems-branch-overview-stt/mcmc/>

#### A7. NASA PDS 与 ESA PSA（含 NOMAD）

- **PDS**：NASA 行星数据系统（含 Atmospheres Node，NMSU 托管），PDS3/PDS4 标准的长期归档，免费公开。【事实】
- **PSA**：ESA 行星科学档案（psa.esa.int），火星相关含 Mars Express、ExoMars TGO/NOMAD 等数据；PDS3/PDS4 双标准。【事实】
- **与我们的关系**：数据上游与归档层；AstraAtmos 火星线的 NOMAD 观测即此类归档数据。
- **答辩用途**：证明「数据接入能力」的上游是官方归档，我们的增量在「用数据做实验」而非「有没有数据」；**不要比数据量**。
- **URL**：<https://pds.nasa.gov/> · <https://pdsatmospheres.nmsu.edu/> · <https://www.cosmos.esa.int/web/psa>

### 3.3 GCM 与科普生态（非竞品，答辩参照）

- **MarsWRF / planetWRF**：行星大气 GCM/中尺度研究模型代码（NASA Ames 系团队），用于 Mars、Titan、Venus 等；**不是平台**。答辩价值：说明我们「站在 GCM 生态之上做实验，而不是再造 GCM」。
- **LMD Mars GCM**：MCD 的母模型；说明 MCD 数据是**模型输出**而非观测，答辩时不能把 MCD 当「观测真值」。
- **ExoPlaSim**：`pip install exoplasim` 的 GPL 开源 3D 气候模式（系外行星），证明「GCM 可以 pip 装」但仍是代码/库，**没有 Web 多用户实验平台**——正是形态差异。
- **SpaceEngine / OpenSpace**：科普/天文馆可视化参照，无科研数据分析工作流。

### 3.4 本层结论

- **威胁排序**：MCD（最权威，必答题）> JMARS（NASA/ASU 生态、20+ 年、行星 GIS 事实标准）> Solar System Treks（NASA 免费 Web 3D）> MCMC（名号撞词）> Eyes（科普观感）；OpenMARS/PDS/PSA 是上游数据源，威胁低。
- **定位句**：我们在这些产品的**下游/上层**——它们交付数据与模型，我们交付「可训练、可回测、可对比、可复现」的实验工作台。
- **答辩禁忌**：不与 JMARS 比多天体覆盖、不与 MCD 比气候态权威性、不与 PDS/PSA 比数据量、不把三维可视化当原创。

---

## 4. 第二层：地球气象数据平台与 AI 气象大模型

### 4.1 数据与可视化平台

| 产品 | 机构 | 形态与能力 | 与我们的差异 |
|---|---|---|---|
| **NASA Worldview** | NASA | 卫星影像/产品浏览与时间对比，全球近实时 | 只浏览，无上传数据、无训练、无预测/评估 |
| **NASA GES DISC Giovanni** | NASA | MERRA-2 等再分析的**无代码在线分析/出图**，限定数据集 | 最像「地球分析台」，但无训练/预测/回测/诊断，仅地球 |
| **Copernicus CDS / CAMS** | ECMWF/欧盟 | ERA5 0.25° 逐小时（1940–今）等数据分发；CAMS 大气成分 | 数据分发中心，不做实验管理 |
| **Google Earth Engine** | Google | PB 级遥感目录 + 云端计算 | 遥感对地观测为主；无行星、无实验台账与严格比较 |
| **Microsoft Planetary Computer** | Microsoft | STAC 数据目录 + Azure 计算环境 | 数据+算力基础设施，非领域实验平台 |
| **Windy / Ventusky** | 商业 | 消费级预报可视化（freemium） | 只做预报展示，无训练/评估 |
| **earth.nullschool.net** | 开源 | 单页全球风场等可视化 | 轻量可视化参照 |
| **中国气象数据网** | 国家气象信息中心/CMA | 气象数据检索下载，部分需实名/审批 | 纯数据分发，无实验能力 |

**本组结论**：地球数据平台「有数据、有图、有算力」，但**没有「一次实验从数据身份到对比结论」的管理层**。Giovanni 是最容易被拿来问「这不就是 Giovanni 吗」的产品——回答：Giovanni 是**固定数据集的在线分析出图**，不接受上传数据、不做模型训练、不做历史回测与多模型严格比较，也不覆盖火星。

### 4.2 AI 气象预报大模型/服务

| 产品 | 机构 | 开放形态 | 输入/输出口径（官方公开） | 与 AstraAtmos 的关系 |
|---|---|---|---|---|
| **AIFS** | ECMWF | 开放数据（GRIB2），内部训练，**不提供用户训练平台** | 0.25°、6h 步长、至 360h、每日 00/06/12/18 UTC；AIFS Single 确定性 + AIFS ENS 集合 | 可作「更权威的 AI 推理基线」，不是实验台 |
| **WeatherNext（WN2/WN3）** | Google DeepMind/Google Research | 数据 + 托管推理，部分模型开源（WN2-mini） | WN3（2026-08-31）：实时卫星输入、5km 多分辨率、逐小时、15 天/360h、64 成员集合；WN2（2025-06）：0.25°、6h、64 成员 | 「AI 能力天花板」参照；目标是换数据/换目标的实验，不适用 |
| **Pangu-Weather** | 华为云 | ONNX 权重，**CC BY-NC-SA 4.0（禁止商用）** | 0.25° 721×1440，4 地面变量 + 5 高空变量×13 层，1/3/6/24h 迭代；官方仓库自述全量训练约 **60TB ERA5 + 3000 GPU-days（V100）** | 硬证据：为什么个人/教学团队不能自己训一个 |
| **FengWu（风乌）** | 上海 AI Lab | 0.25°、69 特征、两帧 6h 输入，ONNX 权重（许可待核实） | 中短期全球预报 | 国内 AI 气象参照 |
| **FuXi（伏羲）** | 复旦大学 | 级联 15 天确定性预报，0.25°/6h，权重 CC BY 4.0（Zenodo） | 15 天全球预报 | 开源权重参照 |
| **NVIDIA Earth-2 / Earth2Studio** | NVIDIA | Apache 2.0 **开源推理工具包**；统一 API 跑 AIFS、GraphCast、WeatherNext2、FuXi-S2S、CorrDiff、Aurora 等；训练配方在 PhysicsNeMo | 纯代码库 | 技术评委最可能问「为什么不直接用 Earth2Studio」：它是**推理与训练库**，无 Web 工作台、无数据治理、无实验编排与诊断 |
| **Microsoft Aurora** | Microsoft | 地球系统基础模型（Nature 2025），可微调；Azure Foundry 托管；Aurora 1.5 逐小时 + 集合 | 多任务（天气/高分辨率/空气污染/海浪） | 「模型即服务 + 微调」路线；绕过平台的另一种路径，需在答辩中承认 |

**「为什么不用现成气象大模型」证据链（答辩可直接用）**

1. **契约固定**：现成模型是「固定输入变量/网格/时效 → 固定输出」，无法支撑「换自己的数据、换目标变量、换窗口与步长」的教学科研实验；
2. **只给推理/权重**：AIFS 由 ECMWF 内部训练、对外只开放推理数据；WeatherNext 开放的是数据与托管推理；Earth2Studio 是代码库；**没有一家提供面向多用户的数据治理 + 实验编排 + 回测诊断工作台**；
3. **训练门槛**：Pangu 官方仓库自述全量训练约 60TB ERA5 + 3000 GPU-days（V100），权重还禁止商用——个人、课程、小团队无法复制；
4. **定位差异**：AstraAtmos 做的是「**选模型、换数据、设实验、比结果、看诊断**」，现成模型可以作为**可选基线**（对照对象），但我们不发布业务预报、不宣称精度超越。

### 4.3 数字孪生与评测基准

- **EU DestinE**（欧盟 + ECMWF 等）：Climate DT + Extremes DT + 数字孪生引擎 + 数据湖，2026-02 官方确认第三阶段为 2026-06～2028-06，目标 2030 年地球数字副本。【事实】量级上是国家级工程，**不可正面对标**；答辩价值：反证「轻量、可私有化、面向教学的实验台」有独立空间。
- **WeatherBench 2**（Google Research 等）：AI 气象事实标准评测榜，指标 RMSE/CRPS/SEEPS/能量谱，1–15 天，ERA5 0.25° 真值，榜单收录 Pangu/GraphCast/FuXi/GenCast/NeuralGCM/Aurora 等。【事实】它是**榜单**不是平台：不提供数据治理、不给用户换数据训练、不做领域诊断。若评委问「评测为什么不用 WB2 口径」，回答：WB2 是公开基准与榜单，我们面向**自管数据与自定义模型**，在其之外提供实验台账与严格兼容比较；指标口径可对照，但 WB2 不解决「同一数据身份下多模型是否可比」。

### 4.4 本层结论

- 第二层**全部 17 个对象都是地球专属**——AstraAtmos 的「地球 + 火星同一套数据治理/训练/三维/评估」是这一层没有的维度；
- 面对 AI 大模型，姿态是「**承认天花板、讲清分工**」：它们是推理服务/权重/榜单，我们是实验台；把现成模型当基线是加分项，不是竞争关系。

---

## 5. 第三层：平台化能力对标（ML 实验管理 / 科学可视化 / 国内平台 / 云训练）

### 5.1 ML 实验管理

| 工具 | 机构/形态 | 强项 | 与 AstraAtmos 的差异 |
|---|---|---|---|
| **Weights & Biases** | CoreWeave（2025-05-05 完成收购），SaaS/私有化 MLOps | 实验追踪、Sweeps、Artifacts、Model Registry、Reports、LLM 观测（Weave） | 无数据发布治理/身份校验、无行星时空语义、无三维、无回测/PFI/严格兼容比较；需 SDK 埋点 |
| **MLflow** | Databricks 发起，2020-06 起 Linux Foundation 托管，开源 | Tracking、Model Registry、Projects、可自托管 | 库+服务，不是领域平台；数据治理与评测口径需自建 |
| **TensorBoard** | Google，开源 | 指标曲线/图/嵌入 | 单机看板，不构成实验管理平台 |
| **Neptune** | 波兰，2025 年被 OpenAI 收购（产品后续待核实） | 大规模实验记录与对比 | 同上；供应稳定性受影响（推断） |
| **ClearML** | 以色列，开源 MLOps + 企业版 | 实验管理、队列/Agent 编排、Data、Serving、私有化 | 无发布审核/身份校验、无行星三维、无回测诊断 |
| **Comet** | 美国，SaaS + 自托管 | 追踪、对比、LLM 评估 | 无领域数据治理/三维/回测 |
| **Kubeflow** | CNCF，开源 | K8s 上的 Pipeline/训练编排 | 需自建 K8s，运维门槛高；无开箱界面与领域评测 |
| **DVC** | 2025 年加入 lakeFS（原 Iterative），开源 | 数据/模型版本、流水线复现 | 命令行；无业务审核、无 Web 实验中心 |
| **Hugging Face Hub（含 Trackio）** | Hugging Face，社区平台 | 模型/数据集托管、Spaces、推理；2025-07-29 发布 Trackio 追踪 | 模型中心导向，非实验业务平台；无数据治理/回测 |

**结论**：实验追踪环节它们确实更成熟（要承认）；但「数据身份 → checkpoint 冻结 → 同口径比较」的链条没有等价物。

### 5.2 科学数据可视化

| 工具 | 机构 | 强项 | 与 AstraAtmos 的差异 |
|---|---|---|---|
| **ParaView** | Kitware（2025-08-12 发布 6.0 / VTK 9.5） | 大规模三维科学可视化、并行渲染、Python 自动化 | 桌面单机/自备算力；无 Web 多用户、无数据治理/训练/回测；无行星底图业务交互 |
| **NASA Panoply** | NASA GISS（5.10.2，2026-10-01） | netCDF/HDF/GRIB 二维图、200+ 投影、动画导出、OPeNDAP/THREDDS/S3 | 桌面 2D 查看器；无三维球体、无训练/治理/比较 |
| **VAPOR** | NSF NCAR | 大气/海洋三维体渲染、时间动画 | 桌面后处理工具；无 Web 平台/训练/回测 |
| **ncview** | Scripps/UCSD | netCDF 快速浏览 | 纯查看器 |
| **CDO / NCO** | MPI-M / UC Irvine | 命令行气候数据算子、netCDF 处理 | 数据准备环节，无平台 |

**结论**：单点渲染与格式支持不值得比；我们的价值是「大气变量与训练/评估数据在同一个 Web 工作台里联动」。

### 5.3 国内遥感/地信/气象/天文平台

| 产品 | 机构 | 形态 | 与 AstraAtmos 的差异 |
|---|---|---|---|
| **航天宏图 PIE-Engine** | 航天宏图（688066） | 遥感云 SaaS：在线数据 + 云算力 + 在线开发/API + AI 解译 | 对地观测/遥感行业导向，非行星大气实验；无实验管理闭环（待核实 AI 训练边界） |
| **中科星图 GEOVIS Earth** | 中科星图（688568） | 数字地球在线平台：三维地球浏览、数据服务、API | 三维地球浏览强；无模型训练/回测/诊断、无火星业务线 |
| **超图 SuperMap** | 超图软件（300036） | GIS 基础平台（SuperMap GIS 2026 已发布），可私有化 | GIS 基础设施而非科研实验平台 |
| **中国气象数据网** | 国家气象信息中心/CMA | 气象数据门户 | 数据分发，无训练/可视化/实验管理 |
| **国家地球系统科学数据中心** | 中科院地理资源所牵头 | 地球科学数据共享 | 数据分发；主题偏陆地表层 |
| **国家天文科学数据中心（NADC）** | 国家天文台 | 天文数据发布/VO 服务 | 数据归档，无大气实验能力 |
| **月球与行星探测工程科学数据发布系统** | 国家航天局体系/国家天文台 | 探月与行星探测科学数据发布 | 数据发布为主；URL/系统名待核实 |

**结论**：国内平台强在「行业数据 + 云算力 + 三维地球」，弱在「科研实验管理」；它们既不是同题竞品，也说明国内少有「行星大气 ML 实验台」这一形态。

### 5.4 云训练平台

华为云 ModelArts、阿里云 PAI、百度飞桨/BML：通用 AI 开发/训练/部署云，强项是算力与工程基础设施；无领域数据治理、无行星三维、无回测/诊断/领域比较。**答辩话术**：它们是「我们可选的算力底座」，不是同类产品。

### 5.5 「现成工具组合能否替代 AstraAtmos」专节（答辩核心）

**结论：能替代「零件」，不能替代「闭环」。**

| AstraAtmos 环节 | 最合适的现成工具 | 可替代程度（推断） | 组合缺口 |
|---|---|---|---|
| 训练指标记录与曲线 | W&B / MLflow / ClearML / Trackio / TensorBoard | 高（成熟度高于我们） | 只记录 run，无数据身份与业务口径 |
| 数据/模型版本 | DVC(+lakeFS) / MLflow Registry / ClearML Data | 中高 | 无发布指纹、审核、身份校验 |
| 训练编排/算力 | Kubeflow / 云训练平台 | 中 | 需 K8s 运维或云费用；无领域任务模板 |
| 三维科学可视化 | ParaView / VAPOR | 高（渲染远超我们） | 桌面单机；与训练/评估数据无联动 |
| 大气二维图/动画 | Panoply | 高（格式支持强） | 桌面；无平台/无行星双球 |
| 数据来源 | 气象数据网 / 地球系统数据中心 / PDS / PSA | 高（作为数据源） | 纯下载，无治理与实验 |
| 模型托管/分享 | Hugging Face Hub | 中高 | 模型中心，无领域评测 |
| **数据治理（发布指纹/身份校验/审核）** | **无现成等价物** | **低** | 需自研 |
| **预测-回测-诊断-对比闭环** | **无现成等价物** | **低** | 需自研（回测、残差 DU、PFI、严格兼容比较） |
| **三维行星 Web 工作台（地球/火星）** | **无现成等价物** | **低** | 需自研三维行星渲染与数据管线 |

**组合方案的 6 个断点**

1. **数据身份断裂**（最核心）：W&B/MLflow 记 run 元数据，不知道数据集发布指纹/审核状态；DVC 记版本但不承载业务审核与身份校验。组合里「同数据身份才能比」只能靠团队纪律，答辩现场无法自证。
2. **领域语义缺失**：Ls/UTC 时间轴、MCD/OpenMARS/NOMAD/MERRA-2 的变量与网格规范、极区/季节/昼夜视图、地球/火星双球，通用工具都没有。
3. **评测链断裂**：历史回测、残差 DU、PFI、严格兼容的多模型比较、科研图 + MATLAB 重建脚本，工具链无现成等价物；ParaView/Panoply 的图与训练指标无关联。
4. **多用户/教学场景缺失**：ParaView/VAPOR/Panoply 是桌面单机；W&B 有账号但无数据发布审核与中文教学界面；云平台有账号但无领域工作台。
5. **供应/运维不确定性**：W&B→CoreWeave（2025-05-05 完成）、Neptune→OpenAI（2025）、DVC→lakeFS（2025）——赛道持续整合，定价与中立性存在不确定性（推断，服务状态需区分「收购」与「停服」）；Kubeflow 要 K8s 运维。
6. **胶水层成本**：每个衔接点都要写代码（导出指标、同步数据版本、对齐时间轴），维护与培训成本落在团队而非工具上。

**答辩话术**

> 先承认：实验追踪用 W&B/MLflow 更成熟，三维渲染 ParaView 更强，二维大气图 Panoply 更全，数据下载有国家数据中心——单点能力我们不否认，也不主张替代它们。
> 再定位：它们各自解决「一个环节」，我们解决「一条链路」——数据治理、行星时空语义、训练、回测、诊断、对比、汇报串在一个 Web 工作台，且评测口径贯穿（数据身份 → checkpoint 冻结 → 比较校验）。
> 后举证：组合方案的真实成本在胶水层；我们可以现场演示「同身份多模型严格比较」，而临时拼装的组合做不到。
> 边界：不宣称精度/速度；定位是实验/研究/教学平台，不是业务预报系统，也不是 GCM 替代品。

### 5.6 2024–2026 行业变化时间线（答辩前对齐事实）

| 时间 | 变化 | 来源性质 |
|---|---|---|
| 2020-06-25 | MLflow 加入 Linux Foundation（常被误传为 2024–26 事件） | 官方博客【事实】 |
| 2025 年 | OpenAI 收购 Neptune（交割与产品后续待核实；**不要说「已停服」**） | 官方公告/媒体【事实：收购】 |
| 2025-05-05 | CoreWeave 完成收购 Weights & Biases | 官方博客/新闻稿【事实】 |
| 2025-07-29 | Hugging Face 发布 Trackio（轻量开源实验追踪） | 官方博客【事实】 |
| 2025-08-12 | ParaView 6.0 / VTK 9.5 发布 | 官方博客【事实】 |
| 2025 年 | DVC 加入 lakeFS | 官方博客【事实：加入；日期待核实】 |
| 2026-10-01 | NASA Panoply 5.10.2 | 官方页【事实】 |
| 2026-08-31 | Google WeatherNext 3 发布（实时卫星输入、5km、逐小时、15 天、64 成员） | 官方博客【事实】 |

> 含义（推断）：实验管理赛道被基础设施/大模型公司整合，通用工具的定价与中立性存在不确定性；「领域一体化平台」的价值反而更清晰——但答辩时要把「收购」与「停服」严格区分。

---

## 6. 对比矩阵（答辩一页表）

图例：● 具备　◐ 部分/受限　○ 间接（依赖上游）　✕ 无

| 产品 | 数据治理/身份 | 多源数据接入 | 三维/Web 可视化 | ML 训练 | 预测/回测 | 严格对比/评估 | 诊断/可解释 | 多用户/账号 | 中文/教学 | 行星维度 |
|---|---|---|---|---|---|---|---|---|---|---|
| **AstraAtmos** | ● | ●（OpenMARS/MCD/NOMAD/MERRA-2） | ● | ● | ● | ● | ●（PFI/残差，非因果） | ● | ● | ●（地+火） |
| MCD v6.2 | ✕（不担保可靠性） | ○（自有气候态库） | ◐（Web 查询） | ✕ | ✕ | ✕ | ✕ | ✕ | ✕ | ● |
| OpenMARS | ✕ | ○（单一再分析） | ✕ | ✕ | ✕ | ✕ | ✕ | ✕ | ✕ | ● |
| JMARS | ✕ | ○（用 PDS 数据） | ●（桌面 GIS/3D） | ✕ | ✕ | ◐（量测/导出） | ✕ | ◐（账号） | ✕ | ●（多天体） |
| Solar System Treks | ✕ | ○（任务数据） | ●（Web 3D+VR） | ✕ | ✕ | ◐（距离/高程/太阳角） | ✕ | ✕ | ◐ | ● |
| NASA Eyes | ✕ | ✕ | ●（3D 科普） | ✕ | ✕ | ✕ | ✕ | ✕ | ◐ | ● |
| NASA MCMC | ◐（待核实） | ◐ | ◐（Visualizer 待核实） | ✕（GCM 生产非用户训练） | ◐（模型输出） | ✕ | ✕ | ✕ | ✕ | ● |
| PDS / ESA PSA | ◐（归档标准） | ●（归档） | ✕ | ✕ | ✕ | ✕ | ✕ | ✕ | ◐ | ● |
| NASA Giovanni | ✕ | ○（限定数据集） | ◐（在线出图） | ✕ | ✕ | ✕ | ✕ | ◐（账号） | ✕ | ✕（地球） |
| CDS/CAMS、GEE 等数据平台 | ✕ | ●（数据目录） | ◐ | ✕ | ✕ | ✕ | ✕ | ◐ | ◐ | ✕（地球） |
| AIFS | ✕ | ✕ | ✕ | ✕（内部训练） | ●（推理） | ✕ | ✕ | ✕ | ✕ | ✕（地球） |
| WeatherNext / Pangu / FuXi / Aurora | ✕ | ✕ | ✕ | ✕（给权重/微调） | ●（推理） | ◐（WB2 榜单） | ✕ | ◐ | ✕ | ✕（地球） |
| Earth2Studio / PhysicsNeMo | ✕ | ◐ | ✕ | ◐（配方/库） | ◐（推理） | ✕ | ✕ | ✕ | ✕ | ✕（地球） |
| DestinE | ◐ | ● | ●（数据湖/DT） | ✕ | ●（数字孪生） | ◐ | ✕ | ✕ | ◐ | ✕（地球） |
| WeatherBench 2 | ✕ | ◐ | ✕ | ✕ | ◐（基准数据） | ●（榜单指标） | ✕ | ✕ | ✕ | ✕（地球） |
| W&B / MLflow / DVC / ClearML | ✕ | ✕ | ✕ | ◐（追踪/编排） | ✕ | ◐（指标对比） | ✕ | ● | ◐ | ✕ |
| ParaView / VAPOR / Panoply | ✕ | ✕ | ●（桌面渲染） | ✕ | ✕ | ✕ | ✕ | ✕ | ◐ | ◐（地球/通用） |
| PIE-Engine / GEOVIS Earth / SuperMap | ◐ | ●（行业数据） | ●（数字地球/GIS） | ◐（待核实） | ✕ | ✕ | ✕ | ● | ● | ✕（地球） |
| ModelArts / PAI / 飞桨 | ✕ | ◐ | ✕ | ● | ◐（部署） | ✕ | ✕ | ● | ● | ✕（地球） |

**读表要点**：只有 AstraAtmos 一行在「训练 + 回测 + 严格对比 + 诊断 + 多用户 + 行星维度」同时为 ●；单点最强的列（三维渲染、追踪、AI 推理）分别被成熟产品占据——这正是「集成/实验层」定位的依据。

---

## 7. 差异化定位与创新点

### 7.1 三个层次的定位句

- **对行星科学评委**：我们不是又一个 MCD/JMARS，而是把公开行星大气数据变成「可训练、可回测、可对比」的实验台；我们承认数据来自 OpenMARS/MCD/NOMAD。
- **对 AI/气象评委**：我们不做业务预报、不与盘古/GraphCast 比精度；我们提供「换数据、换目标、设实验、比结果、看诊断」的实验管理——现成大模型在这里只能当基线。
- **对工程/产品评委**：我们不重复造 W&B、ParaView、Earth Engine；我们补的是它们之间的「数据身份 + 领域语义 + 评测口径」胶水层，并且已产品化。

### 7.2 三个可主张差异（证据 / 对照 / 边界）

| # | 差异点 | 对照对象 | 证据（README 可核） | 边界（不吹） |
|---|---|---|---|---|
| 1 | **训练-预测-回测-评估闭环** | MCD/OpenMARS/JMARS/Treks/MCMC | 实验中心（窗口/步长/通道/超参/划分/早停/迁移/队列/标签/日志/指标）；Earth 回测 UTC+origin_split+DU；Mars Ls 最近邻选窗；单/多模型预测；误差分布/残差/PFI；严格兼容多模型比较 | 真实精度未验收；PFI 非因果 |
| 2 | **可复现的数据身份与比较口径** | MCD（不担保可靠性）、OpenMARS/PDS/PSA（只发数据）、W&B/MLflow/DVC（只记 run/版本） | 注册表版本/发布指纹/409 身份校验；上传 .nc 校验与审核；checkpoint 冻结数据身份+归一化+split_meta；比较强制同指纹/范围/窗口/指标，差异 409 | 仅已实现并验证范围；不覆盖任意第三方数据 |
| 3 | **中文 Web 一体化 + 教学/研究双用** | JMARS 桌面 Java、MCD Fortran 库、ParaView 桌面、云平台按量计费 | Three.js 三维球体/观测台/分析工作台/AI 解读同一产品；中英文界面、账号/审核/通知；Windows 分发、Linux 部署、可私有化 | 长 horizon 浏览器性能未系统验收；无 VR；AI 依赖外部接口 |

### 7.3 诚实清单：这些**不是**我们的原创差异

- 三维行星可视化（Treks/Eyes/JMARS）、行星 GIS 图层（JMARS）、气候态数据（MCD）、再分析数据（OpenMARS）、行星归档（PDS/PSA）；
- 实验追踪/看板（W&B/MLflow/TensorBoard）、数据版本（DVC）、三维科学渲染（ParaView/VAPOR）、大气二维图（Panoply）、AI 预报模型（AIFS/WeatherNext/Pangu 等）；
- 「有地球/火星数据」本身——全部公开可得。

### 7.4 创新点怎么讲（给评委的准确表述）

> 我们的创新不在「发明一个更强的模型」，而在**把行星大气研究里分散的数据、模型、评估与出图流程，收敛成一个可复核的 Web 实验台**，并为此实现了三件事：①数据身份贯通（发布指纹 → checkpoint 冻结 → 比较校验）；②领域时间与坐标语义（Mars Ls 最近邻选窗、Earth UTC origin_split 回测、DU 单位、球面网格契约）；③同口径多模型严格比较与可重建科研图（PNG + MATLAB 脚本）。这是**领域化集成创新 + 工程治理创新**，不是算法首创。

---

## 8. 答辩 Q&A（18 问）

> 每题：评委意图 → 30 秒回答 → 证据/可演示 → 不要说什么。

**Q1「有 MCD 了，为什么还需要你们？」**
- 意图：质疑必要性。
- 回答：MCD 是气候平均场数据库，交付的是数据与查询器；它明确不担保可靠性、不支持训练/预测/评估。我们把它当基准数据源之一，并补齐「在它上面做实验如何可复现」——数据身份、训练、回测、同口径比较。它是上游，我们是实验层。
- 证据：MCD access 页条款；我们观测台叠加 MCD 全球均值曲线。
- 不要说：MCD 数据不权威/我们要替代 MCD。

**Q2「和 JMARS 比呢？」**
- 回答：JMARS 是 20+ 年积累的行星 GIS，强在遥感图层、任务规划与多天体；它不做大气变量时间序列、不做模型训练与评估。我们做的是大气数据的实验闭环，三维只是入口。
- 不要说：我们三维比 JMARS 好；我们能覆盖更多天体。

**Q3「三维行星可视化 NASA Treks 早做了，你们是不是重复造轮子？」**
- 回答：是，三维浏览是成熟能力，我们不主张原创。区别在对象与链路：Treks 做地表地形/影像与量测；我们做**大气变量场**（温度/风/臭氧/沙尘）的多时间步播放、点位序列、剖面与训练/评估联动。
- 不要说：我们的 3D 更先进；我们有 VR/WMTS（没有）。

**Q4「为什么不用 W&B + DVC + ParaView 组合？」**
- 回答：组合能替代零件，替代不了闭环。断点在数据身份：W&B 记 run、DVC 记版本，但都不承载发布指纹/审核/身份校验；领域语义（Ls/UTC、四套数据规范）与评测链（回测/残差 DU/PFI/严格兼容比较）也无等价物。组合的真实成本在胶水层，而胶水层正是我们已实现的部分。
- 可演示：同数据身份下两个模型比较；改数据后返回 409 强制重训。
- 不要说：W&B/DVC 不好用；我们功能比它们全。

**Q5「和 MLflow 比，你们的实验中心有什么不同？」**
- 回答：MLflow 是通用实验追踪/模型注册的标准，可自托管，这点比我们成熟。我们的实验中心是**领域化**的：任务模板（窗口/步长/通道/早停/迁移）、行星时间语义、checkpoint 冻结数据身份、预测/回测/诊断/比较一体化，面向教学与非工程用户。
- 不要说：我们替代 MLflow；我们追踪功能更强。

**Q6「为什么不直接用盘古/GraphCast/AIFS？」**
- 回答：它们是固定输入契约的推理服务/权重：AIFS 只开放推理数据、WeatherNext 只开放数据与托管推理、盘古权重禁止商用且官方自述全量训练约 60TB ERA5 + 3000 GPU-days。它们不能换数据、换目标、设对照实验、做严格比较与诊断。我们做实验台，现成模型可作基线。
- 不要说：我们精度接近它们；我们已经集成它们（未集成）。

**Q7「你们的预测精度是多少？和 GraphCast 比呢？」**
- 回答：诚实说：当前版本尚未完成真实精度验收，我们不发布精度数字，也不与业务级模型比精度。我们比的是「实验是否可复现、比较是否公平、诊断是否可解释」；精度是我们下一阶段要补的验收项。
- 不要说：任何未经验证的 RMSE/准确率数字；不要用演示数据当精度证据。

**Q8「你们和 NASA MCMC/自研 GCM 什么关系？」**
- 回答：MCMC 是模型生产方（Ames MGCM/MarsWRF 谱系），我们是模型消费者与实验编排者。我们不运行 GCM、不自研 GCM；我们用 MCD/OpenMARS 这类 GCM 产品与再分析作为数据源，在其上做 ML 实验。
- 不要说：我们做火星气候建模（会撞词）；我们替代 GCM。

**Q9「数据从哪来？版权/许可能商用吗？」**
- 回答：火星来自 OpenMARS（Open University，DOI 发布）、MCD（LMD/ESA，科学用途免费、未经授权不得商用）、NOMAD（ESA PSA 归档）；地球 MERRA-2（NASA）。平台做接入与实验，不转售数据；MCD 的商用限制我们遵守，产品定位为科研/教学。具体许可证以各官方条款为准，答辩材料标注来源与检索日期。
- 不要说：数据是我们的；所有数据可自由商用。

**Q10「这和 Giovanni / Earth Engine 有什么区别？」**
- 回答：Giovanni 是限定数据集的在线分析出图，Earth Engine 是遥感数据目录 + 云算力；都不接受上传数据、不做模型训练/回测/严格比较，也没有火星。我们是「实验管理」而不是「数据浏览/云计算」。
- 不要说：我们是更快的 Giovanni；我们数据比 GEE 多。

**Q11「和 DestinE 数字孪生比，你们是不是太小了？」**
- 回答：是，量级不同，不能正面对标。DestinE 是国家级地球数字孪生工程（第三阶段 2026-06～2028-06）；我们是轻量、可私有化、面向课程与小团队的实验台——这恰好是大型工程的空白侧。
- 不要说：我们能做数字孪生；我们和 DestinE 同级。

**Q12「为什么做火星+地球双线？是不是噱头？」**
- 回答：双线共用同一套数据治理、训练、三维与评估基础设施，验证的是**领域无关的实验闭环**：火星有 Ls 时间轴与球面相位，地球有 UTC 三小时与 DU 单位；一套平台两种时间语义，说明架构不是为单一数据集硬编码。第二层 17 个地球产品全是地球专属，没有多行星维度。
- 不要说：火星数据是我们独有；双线是为了好看。

**Q13「单点都不如成熟工具，创新点到底是什么？是不是只是包装开源？」**
- 回答：我们承认单点差距（追踪不如 W&B、渲染不如 ParaView、AI 不如业务模型）。创新在三处领域化：①数据身份贯通；②行星时间/坐标语义（Ls 最近邻选窗、UTC origin_split 回测、DU、输出网格契约）；③同口径严格比较 + 可重建科研图。包装开源不会产生这些约束。
- 可演示：不兼容比较返回 409 的现场演示。
- 不要说：我们全部自研；我们没有依赖开源。

**Q14「大厂/大模型会不会很快覆盖你们？护城河在哪？」**
- 回答：大厂在做模型与算力（AIFS/WeatherNext/Earth-2/Aurora），不做「面向教学与科研的领域实验管理」；这个层需要数据身份、领域语义与教学交互，商业回报小、工程琐碎。我们的护城河是领域工作流与数据治理的耦合，不是某个模型。
- 不要说：大厂做不了；我们有技术壁垒（不准确）。

**Q15「用户是谁？商业模式/市场规模多大？」**
- 回答：目标用户是行星/大气课程的教学团队、做小样本时空预测实验的研究生与交叉学科研究者、以及需要私有化演示环境的实验室。商业模式本次不做预测（不编造市场数字）；可探索私有化部署、课程授权、科研合作。市场规模需另做调研，答辩不虚构。
- 不要说：任何未核实的用户数/收入/市场规模。

**Q16「如何保证多模型比较公平、可复现？现场怎么验证？」**
- 回答：比较前强制同发布指纹、同测试范围、同窗口、同指标口径，不兼容直接 409；checkpoint 冻结数据身份、归一化与 split_meta；所有指标来自同一冻结测试集。现场可演示：①同一数据集两个模型比较成功；②换数据版本后旧比较被拒绝（409）并提示重训。
- 不要说：我们自己写脚本保证公平（要讲产品机制）。

**Q17「AI 对话/Copilot 是不是只是调 API？有技术含量吗？」**
- 回答：是外部模型接口，且无 key 时降级兜底；它的定位是**解读辅助**，不是预测能力。技术含量在「把结构化分析结果（图表/指标/PFI）组织成可解释上下文」，而不是模型本身。我们不宣称 RAG 或自主智能体。
- 不要说：我们有自研大模型；AI 能替代分析师。

**Q18「项目最大的风险是什么？」**
- 回答：三点：①精度与性能验收未完成——这是下一步最硬的工作；②数据许可与再分发边界（MCD 非商用、部分数据许可待核实）；③范围膨胀风险——双线 + 29 个模型标识 + 大量功能，需要用「实验闭环」主线收敛演示。已有缓解：边界文档化、能力开关、演示数据明示不代表精度。

---

## 9. 风险、待核实清单与演示建议

### 9.1 不可宣称清单（答辩红线）

- 精度/准确率/收敛速度优势；「达到业务预报水平」；「实时卫星接入」；「VR」；「自研 GCM」；「RAG 知识库」；
- 市场规模、用户数、收入、客户案例（公开渠道无法核实）；
- 已集成 AIFS/WeatherNext/Pangu（事实是未集成，只能表述「可对照/可接入」）；
- 把 MCD 当观测真值；把 PFI 当因果结论；把演示数据当精度证据。

### 9.2 答辩前需人工复核（按优先级）

| 优先级 | 事项 | 原因 |
|---|---|---|
| 高 | NASA MCMC 的 MCMC Webserver / MGCM Visualizer 正文（429/404） | 决定「模型数据可视化」威胁是否被低估 |
| 高 | MCD Web 界面（mcd_python）可查询变量/图表清单 | 量化与 MCD 的重叠度，准备 Q1 细节 |
| 高 | OpenMARS 数据本体许可证、MCD 商用授权路径 | 数据合规问答 |
| 中 | JMARS 开源范围/许可证；用户数为机构自述的引用口径 | 避免误引 |
| 中 | NASA Solar System Treks / Eyes 当前功能与形态 | 三维对比时准确 |
| 中 | W&B/CoreWeave 收购后教育版/私有化政策；Neptune 收购后服务状态 | 组合替代问答；不要说「停服」 |
| 中 | AIFS 正式 operational 公告措辞、ENS 成员数；WeatherNext 2 精确时效上限 | AI 基线口径 |
| 低 | 月球与行星探测工程科学数据发布系统 URL/系统名；PIE-Engine AI 训练边界；GEOVIS Earth 能力清单 | 国内平台表述 |
| 低 | VAPOR/ParaView/MLflow 等 2026 最新版本号 | 时间线准确性 |

### 9.3 现场演示 60 秒脚本（配合答辩）

1. 10s 定位：打开首页 → 说明「行星大气 ML 实验平台，地球+火星」；
2. 15s 数据治理：数据总览 → 数据集详情显示版本/发布指纹/能力声明；
3. 15s 实验闭环：实验中心 → 一条已完成任务 → 指标与日志；
4. 10s 严格比较：多模型比较 → 说明同指纹/同范围/同口径；若不兼容，展示 409 提示；
5. 10s 诊断与出图：残差/PFI → 科研图导出（PNG + MATLAB 脚本）→ 一句话收尾「数据身份贯穿，比较可复现」。

---

## 10. 来源清单（检索日期 2026-10-29）

**第一层 行星/火星**
- MCD：<https://www-mars.lmd.jussieu.fr/> · <https://www-mars.lmd.jussieu.fr/mars/access.html>
- OpenMARS：<https://data.open.ac.uk/page/oro/62713> · <https://ordo.open.ac.uk/articles/dataset/OpenMARS_continuous_MY28-35_standard_database/24573205>
- JMARS：<https://jmars.asu.edu/> · <https://jmars.asu.edu/open_source>
- NASA Solar System Treks：<https://trek.nasa.gov/> · <https://trek.nasa.gov/mars/> · <https://science.nasa.gov/sciact-team/solar-system-trek/>
- NASA Eyes：<https://eyes.nasa.gov/apps/solar-system/>
- NASA MCMC：<https://www.nasa.gov/space-science-and-astrobiology-at-ames/division-overview/planetary-systems-branch-overview-stt/mcmc/>
- NASA PDS / Atmospheres Node：<https://pds.nasa.gov/> · <https://pdsatmospheres.nmsu.edu/>
- ESA PSA / NOMAD：<https://www.cosmos.esa.int/web/psa> · <http://psa.esa.int> · <https://data.aeronomie.be/dataset/calibrated-pds4-data-from-nomad-on-the-exomars-trace-gas-orbiter>
- planetWRF：<http://planetwrf.com/> · ExoPlaSim：<https://github.com/alphaparrot/ExoPlaSim>
- SpaceEngine：<https://spaceengine.org/> · OpenSpace：<https://www.openspaceproject.com/>

**第二层 地球气象与 AI**
- Worldview：<https://worldview.earthdata.nasa.gov/>
- Giovanni：<https://giovanni.gsfc.nasa.gov/giovanni/>
- CDS/CAMS：<https://cds.climate.copernicus.eu/> · <https://ads.atmosphere.copernicus.eu/>
- Earth Engine：<https://earthengine.google.com/> · Planetary Computer：<https://planetarycomputer.microsoft.com/>
- Windy：<https://www.windy.com/> · Ventusky：<https://www.ventusky.com/> · earth.nullschool：<https://earth.nullschool.net/>
- ECMWF AIFS：<https://www.ecmwf.int/en/about/media-centre/aifs-blog> · 开放数据：<https://data.ecmwf.int/forecasts/>
- WeatherNext：<https://deepmind.google/technologies/weathernext/> · GraphCast：<https://deepmind.google/technologies/graphcast/>
- Pangu-Weather：<https://github.com/198808xc/Pangu-Weather>
- FengWu：<https://github.com/OpenEarthLab/FengWu>
- FuXi：<https://github.com/OpenEarthLab/FuXi> · 权重：<https://zenodo.org/records/8100201>
- NVIDIA Earth-2 / Earth2Studio：<https://www.nvidia.com/en-us/high-performance-computing/earth-2/> · <https://github.com/NVIDIA/earth2studio>
- Aurora：<https://www.microsoft.com/en-us/research/project/aurora/> · Nature 2025：<https://www.nature.com/articles/s41586-025-09005-y>
- DestinE：<https://destination-earth.eu/> · WeatherBench 2：<https://sites.research.google/weatherbench/>

**第三层 平台化工具**
- W&B：<https://wandb.ai> · CoreWeave 收购完成：<https://www.coreweave.com/blog/coreweave-completes-acquisition-of-weights-biases>
- MLflow：<https://mlflow.org> · TensorBoard：<https://www.tensorflow.org/tensorboard>
- Neptune：<https://neptune.ai> · ClearML：<https://clear.ml> · Comet：<https://www.comet.com>
- Kubeflow：<https://www.kubeflow.org> · DVC：<https://dvc.org> · Hugging Face Hub：<https://huggingface.co> · Trackio：<https://huggingface.co/blog/trackio>
- ParaView：<https://www.paraview.org> · Panoply：<https://www.giss.nasa.gov/tools/panoply/> · VAPOR：<https://www.vapor.ucar.edu> · ncview：<https://cirrus.ucsd.edu/~pierce/ncview_home_page.html> · CDO：<https://code.mpimet.mpg.de/projects/cdo> · NCO：<http://nco.sourceforge.net>
- 航天宏图 PIE-Engine：<https://engine.piesat.cn> · 中科星图 GEOVIS Earth：<http://geovisearth.com.cn> · 超图：<https://www.supermap.com>
- 中国气象数据网：<http://data.cma.cn> · 国家地球系统科学数据中心：<http://www.geodata.cn> · 国家天文科学数据中心：<https://nadc.china-vo.org> · 中国探月与深空探测网：<http://www.clep.org.cn>
- 华为云 ModelArts：<https://www.huaweicloud.com/product/modelarts.html> · 阿里云 PAI：<https://www.aliyun.com/product/pai> · 飞桨：<https://www.paddlepaddle.org.cn> · BML：<https://cloud.baidu.com/product/bml>

> 说明：以上链接为 2026-10-29 可访问的官方入口；部分产品页面为动态渲染，正文内容以当日检索到的官方陈述为准，未逐字引用。引用本报告时请保留检索日期与「事实/推断/待核实」标注。
