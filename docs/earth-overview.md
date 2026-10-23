# 二维地球数据总览

当前地球总览只使用 `earth_merra2_3hourly_v1`：UTC datetime 轴、降采样地图、原生点位序列、面积加权均值和三小时源日聚合年度分析。观测档首次打开或刷新默认显示三维球体，不因三小时频率切换到二维；二维地图可在“显示”面板手动选择，也是 WebGL 不可用时的自动回退视图。开发与生产默认一致，日频 v1/v2 的总览和分析接口返回 409 `dataset_retired`，下文日频协议及交付记录均为历史资料。详见[停用约定](earth-dataset-retirement.md)及下方“三小时 UTC 总览”；`cell_bounds` 和 `wrap_longitude` 以 descriptor 为准。

实施方案见 [二维地球总览实施方案](plans/2026-09-23-earth-overview-2d.md)；第一阶段（数据集注册与训练任务身份）见 [数据集注册表](dataset-registry.md)；数据包协议见 [地球小数据包](earth-compact-dataset.md)。

用户已确认的[火星 / 地球共用分析工作台](plans/2026-09-23-earth-shared-analysis-workbench.md)首期范围**已经实现**：三维地球、年度分析、极区统计与图表 AI 解读见[共用分析工作台](earth-analysis-workbench.md)。本文仍是二维基础层的协议文档；页面现在默认挂载“行星观测台”布局（四个区域见下方交付说明），`EarthOverviewScene` 作为兼容保留的二维实现留在仓库中，二维地图与时间轴组件被观测台直接复用。

文档最近核对日期：**2026-10-10**；日频入口已关闭，三小时数据状态以当前 catalog 的 `availability` 为准。缺失发布不回退到日频。

> **当前界面（2026-10-10）**
>
> 数据总览默认挂载行星观测台。观测档使用三维球体与左右两条观测轨；分析档隐藏球体，默认进入主题组合看板，宽屏左侧为主题与条件竖栏，右侧一张主图加两张辅助图，支持单图放大与返回，窄屏顺序排列。年份、适用变量与纬带在左侧选择，UTC 时间、读数、点位序列和坐标输入统一在观测档使用。“单项深入分析”保留原有全部图表与 AI 解读，其中多变量年内趋势仍可切换 Z-score 和原始单位。数据源面板、错误重试与昼夜不可用原因均提供可操作入口或明确说明。详细控件约定和验证记录见[共用分析工作台](earth-analysis-workbench.md)。

## 已实现范围

| 能力 | 状态 |
| --- | --- |
| 总览“地球 / 火星”切换（默认地球） | 已实现 |
| 地球数据集元信息、可用状态、日期/UTC 时间与变量选择 | 已实现 |
| 全球经纬度底图上的全球热力图与真实单元边界 | 已实现 |
| 三小时 UTC 时间选择与播放/暂停 | 已实现；日频选择已移除 |
| 五个变量：`TO3`、`U10M`、`V10M`、`T2M`、`SWGDN`（原始单位） | 已实现 |
| 点击有效网格查看当前值与对应频率的完整点位时间序列 | 已实现 |
| 全球单元面积加权均值；三小时年度图按 UTC 每日八样本聚合 | 已实现 |
| 本地海岸线底图（运行时不依赖外部 CDN） | 已实现 |
| 中英文、深浅主题、390/900/1440px 布局、请求过期保护 | 已实现 |
| 三维全球球体、行星观测台（顶部条件栏 + 全幅画布 + 时间轨道 + 底部分析区）、`observe`/`analyze` 两档视图、年度分析、极区统计、图表 AI 解读 | 已实现，见 [共用分析工作台](earth-analysis-workbench.md) |
| 地球昼夜变化 | 未实现；三小时发布包含日内样本，当前仅未接入该分析，不回退到火星昼夜接口 |
| 风场粒子、派生风速 | 未实现 |
| 地球训练、地球预测 | 三小时网页与后端已实现，日频已停用，见[训练与回测](earth-training.md) |
| 自选多边形区域、重网格、平滑/插值、臭氧单位换算、导出、PFI | 未实现 |
| Earth/Mars 数值叠加与跨星球比较 | 未实现（首期不做） |

Earth 只使用 `earth_merra2_3hourly_v1` 的带 `Z` UTC ISO datetime、3 小时步长、0.75°×0.75°、240×480 网格，变量为 TO3、U10M、V10M、T2M、SWGDN。数据集切换会清理上一身份的场、时间、点位和年度缓存，不把旧日频任务或结果解释为三小时数据。

三小时“全球平均”使用全球单元球面面积权重。归档日频 v2 的 5° 全球单元均值与 v1 的区域抽样点均值仅是历史协议。

## 日频只读发布快照（历史实现）

总览不直接打开 NetCDF，而是复用第一步的注册表：

- `services/earth_dataset_metadata.py` 新增 `VerifiedEarthRelease`（`metadata`、`signature`、`dates`、`latitude`、`longitude`、`fields`）与 `read_earth_release()`；`read_earth_metadata()` 保留原签名，改为返回 `read_earth_release(...).metadata`。
- 五个 field 都是 `[T, H, W]` float32，日期是 `datetime64[D]`，坐标保持原始顺序；数组设为**不可写**，`fields` 用只读 mapping 包装。
- `DatasetRegistry.get_earth_overview_snapshot(dataset_id, expected_fingerprint)` 是唯一入口：registry 负责路径、发布 SHA、锁与校验，调用方拿不到 `_earth_package_dir` 等私有字段。
- 描述符查询与快照共享同一次校验结果与同一把锁：**每个进程按发布 ID 最多缓存一份五场数组**，关闭文件句柄后仍可读；不创建每日期/变量副本，不引入 Redis 或总览缓存表。
- 缓存标签是**该次校验自己确认的**文件签名，不会把较新的 `stat()` 结果贴到旧数组上。包缺失、校验失败或校验中变化时，旧快照立即失效，目录按第一步协议仍返回状态。
- 调用方传入的 `expected_fingerprint` 与当前已验证指纹不一致时返回 409，不返回“新数据配旧元信息”。

## 总览接口

### 三小时 UTC 总览

`earth_merra2_3hourly_v1` 通过同一组 overview 路由提供独立的 UTC 时间轴：

```text
GET /api/datasets/earth_merra2_3hourly_v1/overview/field
    ?timestamp=2020-01-01T01:30:00Z&variable=TO3&expected_fingerprint=<sha256>
GET /api/datasets/earth_merra2_3hourly_v1/overview/point-series
    ?lat=0.375&lon=0.375&variable=TO3&expected_fingerprint=<sha256>
```

三小时场请求只接受发布轴上带 `Z` 的 UTC timestamp，不会截断成日期。默认地图响应使用服务端 4×4 球面单元面积均值，返回 `field[60][120]`、`source_grid_shape=[240,480]`、`render_grid_shape=[60,120]` 和 `render_method=spherical_cell_area_mean_4x4`；可选 stride 8 返回 30×60。缺测保留为 `null`，不填零。`point-series` 始终从 240×480 原生网格读取，返回连续 3 小时 `timestamps` 和兼容别名 `dates`，因此显示降采样不会改变点位数值。

context 和场/序列响应都声明 `frequency_hours=3`、`step_unit=hour`、`step=3`、`time_zone=UTC`。前端数据集切换时取消旧请求并清空场、时间、点位和年度缓存；Canvas 只创建一个栅格节点，不为 115,200 个原始格点创建 SVG/React 节点。缺包、指纹变化和时间戳错误沿用结构化 `dataset_unavailable`、`dataset_version_changed`、`invalid_timestamp`/`timestamp_out_of_range`。

三小时观测轨左侧读数区直接提供 UTC 日期与时间控件，不重复显示完整时间戳；右侧时间戳按日期与 UTC 时间分行。控件宽度受轨道约束，两侧读数区按文字缩放使用相同高度，保持曲线起止位置对齐；较矮视口下两侧轨道同步纵向滚动，窄屏为两轨预留足够高度。

年度研究接口仍返回日序列，三小时发布由后端逐块读取每天 8 个连续样本，统计定义为 `utc_daily_mean_of_8_three_hour_means`。不完整 UTC 日或缺测会返回 `incomplete_daily_aggregate`，不会将部分日均值伪装成完整日数据；三小时字段的日内变化分析尚未开放，能力说明会明确区分“未实现”和日频数据没有日内采样。

URL 前缀统一为 `/api/datasets/{dataset_id}/overview`，仅支持 `earth_merra2_3hourly_v1`；两个日频 ID 返回 409 `dataset_retired`。公开只读，与目录权限一致，不使用火星上传来源参数，也不接受任何磁盘路径。

**所有请求必须传 `expected_fingerprint=<64位小写SHA>`**，值取自当前数据集元信息。这样一次页面会话不会把不同版本的数据拼接展示。

| 接口 | 必传参数 | 可选参数 |
| --- | --- | --- |
| `GET .../field` | UTC `timestamp`、`variable`、`expected_fingerprint` | `render_stride=8` |
| `GET .../regional-series` | `variable`、`expected_fingerprint` | `start`、`end` |
| `GET .../point-series` | `lat`、`lon`、`variable`、`expected_fingerprint` | `start`、`end` |

变量严格匹配 `TO3`/`U10M`/`V10M`/`T2M`/`SWGDN`；不接受 `wind`、`O3`、`temperature` 等别名，以免与火星字段混用。时间只接受发布轴上的 UTC ISO datetime。

### 公共响应字段

```text
dataset_id, dataset_version, dataset_fingerprint, planet: "earth", variable, units
```

`units` 是变量原始单位：`TO3` → `DU`，`T2M` → `K`，`U10M`/`V10M` → `m s-1`，`SWGDN` → `W m-2`。

### `/field` 额外字段

```text
date, calendar
lat[H]，lon[W]：有限、升序
dimension_order: ["lat", "lon"]
field[H][W]：有限物理值，永不归一化
coverage: {latitude_range, longitude_range, wrap_longitude: true for v2, false for v1}
color_range: {min, max, scope: "dataset", centered_on_zero}
statistics: {min, max, regional_mean, valid_count}
```

`field[0][0]` 对应**最南、最西**点，不是图像左上角。前端按真实坐标绘图，不翻转后又重复翻转。

### `/regional-series` 额外字段

```text
start, end, dates[]/timestamps[]（升序连续、含端点；三小时响应为 UTC 时间戳）, values[]（每个时间步一个有限均值）
aggregation: "spherical_cell_area_mean" (v2) or "cos_lat_sample_mean" (v1)
coverage
```

### `/point-series` 额外字段

```text
start, end
requested: {lat, lon}          请求坐标
grid_point: {lat, lon, lat_index, lon_index}   实际采样点
selection: "nearest_grid_point"
dates[]/timestamps[], values[]  原始网格点三小时值（归档日频为逐日值）
```

曲线接口**不接收当前时间参数**：当前值由前端按已展示的 `field.date` 在 `dates` 中查找。这样播放时只请求下一个时间步的场，不重复取点或区域全序列。

### 错误协议

统一为 `{"detail": {"code": "...", "message": "..."}}`；只有缺包/坏包时额外带安全的 `availability_reason`。不返回服务器路径或 traceback。

| 场景 | HTTP / code |
| --- | --- |
| 未注册 ID | 404 `unknown_dataset` |
| 已注册 Mars ID 请求总览接口 | 409 `dataset_overview_not_supported` |
| `expected_fingerprint` 与当前发布不一致 | 409 `dataset_version_changed` |
| 已注册 Earth 缺包/坏包/不可读 | 503 `dataset_unavailable`（含 `availability_reason`） |
| 校验过程中包变化 | 503 `dataset_unavailable`，reason 为 `package_changed_during_verification` |
| 无效日期 / 日期不存在 / 日期越界 | 422 `invalid_date` / `date_out_of_range` |
| `start > end` | 422 `invalid_date_range` |
| 未支持 variable | 422 `unsupported_variable` |
| 经纬度超覆盖区（即使仍在全球范围内） | 422 `point_outside_coverage` |
| 缺必需参数、SHA 格式错误、NaN/Inf 坐标 | Pydantic 422，前端以 `invalid_request` 显示 |

**严格禁止**把越界点夹到边缘，或把 240° 经度 wrap 成 −120°。同距离最近邻取**数组较小下标**，前后端一致。

## 区域均值与色带

v2 区域序列是全球 5° 单元的球面面积加权均值：

```python
edges = np.linspace(-90, 90, len(latitude) + 1)
weights = np.diff(np.sin(np.deg2rad(edges)))
means = np.einsum("thw,h->t", field.astype("float64"), weights) / (
    len(longitude) * weights.sum()
)
```

这与预处理器的球面单元重叠面积规则一致；它是发布网格上的面积加权平均，不是大气总质量。v1 的旧余弦抽样语义保持不变。`field.statistics.regional_mean` 与同一天 `/regional-series` 的值完全一致（浮点容差内）。遇到异常 NaN 不填零、不改分母，按 `invalid_dataset` 处理。

色带在同一变量的整个数据期固定，播放不会每帧重算：

- `TO3`、`T2M`、`SWGDN`：全数据期 min/max，使用用户在全局设置里选的 colormap。
- `U10M`、`V10M`：`[-max(|min|,|max|), +max(|min|,|max|)]`，`centered_on_zero=true`，固定使用 `rdbu`（低值蓝、高值红），图例标注正负方向。
- 常数场允许 min == max；前端使用色带中点，不除以零，也不伪造统计值。
- 图例与格子共用同一转换函数与范围；渲染颜色不改变数值。

## 地图、日期与页面行为

### 地图与无数据

地图使用 `viewBox="0 0 360 180"` 的 2:1 等经纬度投影；场数据使用单一 Canvas 栅格节点，SVG 只保留覆盖边界、海岸线、经纬网和点位交互：

```javascript
project(lon, lat)   -> { x: lon + 180, y: 90 - lat }
unproject(x, y)     -> { lon: x - 180, lat: 90 - y }
```

- v2 数据格覆盖经度 [−180, 180]、纬度 [−90, 90]，包括两极和经度接缝；地图使用 manifest 的真实 cell bounds。v1 仍只覆盖 [−120,120]×[−60,60]。
- v2 网格坐标是单元中心，按真实单元覆盖范围绘制完整 5° 单元；内部边界由已验证的规则网格中心中点重建，外边界取 field 的 coverage（来源于 NetCDF cell bounds）。v1 兼容路径按中心中点划格并裁到声明覆盖端点，不把旧抽样值解释成面积平均。
- 不补首尾重复列，不做经度环绕插值。
- 绘制顺序：无数据纹理 → 服务端返回的三小时显示栅格 → 真实海岸线与经纬网 → 区域边界与选中点。显示栅格默认 60×120；Canvas 只创建一个栅格节点，海岸线设 `pointer-events: none`，不遮挡点击。
- 点击用 `getScreenCTM().inverse()` 与 `DOMPoint` 换算回 viewBox 坐标，不按含 letterbox 的 DOM 尺寸线性映射。**点击处理挂在 svg 上**，因此区域外点击也能给出提示，而不是被无数据图层吞掉。
- 区域外点击只显示“区域外无数据”，**不发点位请求、不移动上一个有效点**；覆盖边界上的点允许选择。
- 点位选择先前端快速吸附到最近采样点，再以接口返回的 `grid_point` 为最终位置；不做插值。
- 补充经纬度数值输入与“查看点位”按钮，供键盘用户使用；不为每个数据格创建独立 tab stop。

### 底图

使用 Natural Earth 1:110m coastline，已下载入仓库（`frontend/public/earth/ne_110m_coastline.geojson`），运行时不请求在线瓦片或 CDN。来源、许可与 SHA 见 [底图资源说明](../frontend/public/earth/README.md)。相邻经度差大于 180° 时断开为独立子路径，不出现横跨全世界的长线，末点不自动闭合。底图加载失败时保留经纬网与有效数据，并给出可重试提示。

### 场景切换

- 路由仍为 `#/overview`，不新增路由库，不把 planet 参数写入全局 Settings。
- 默认 `earth`，按钮按地球、火星排列；`planet=mars` 挂载原有火星内容，`planet=earth` **只**挂载地球场景（互斥挂载，不做 hidden 双挂载）。地球不挂载火星三维背景、摄像头、Mars 查询组件或 Copilot。
- 切换前统一暂停火星播放；火星内容卸载时 abort 在途请求并清除 timer，旧回包不能写入持久的火星 provider。
- 离开地球时暂停并清理所有请求与 timer；返回时保留日期/变量/点位选择，数据按当前 descriptor fingerprint 重新校验加载。
- 地球缺包时仍可进入场景看到说明与重试，不自动切回火星掩盖错误。

### 日期与播放

- 当前时间状态使用带 `Z` 的 UTC ISO datetime，全部按 UTC 运算，不使用 `toLocaleDateString`，支持 3 小时中心标签；日频日期分支仅保留兼容代码。
- 日期输入 `min`/`max` 来自元数据；非法或越界值显示错误，不静默夹取。
- 滑块以 `0..count-1` 为索引，标签始终显示真实 UTC 时间（不是 Ls 0..360）。
- 前/后一个时间步与时间选择都会暂停播放；到首尾时对应按钮禁用。
- 每个成功展示帧停留 600ms 再请求下一个时间步，使用可清理的单次 `setTimeout`，不用固定 interval 堆积未完成请求。
- 场仍在加载时不前进；失败立即暂停并显示重试。
- 展示完最后一个时间步即停止，不循环；“从首日重播”显式回到首个发布时间再开始。
- 切换变量、手动选点、离开页面、标签页隐藏时暂停。暂停不取消正在加载的用户所选日期，但不再安排下一帧。

### 异步一致性

请求分四个通道：元信息、`field`、`regional-series`、`point-series`。每通道最多一个 active controller，新请求取消旧请求，并用 **token + context 双重检查**，即使 fetch mock 不尊重 abort 也不会写入旧结果：

```text
field identity         = [dataset_id, fingerprint, variable, requested_date]
regional identity      = [dataset_id, fingerprint, variable, start, end]
point identity         = [dataset_id, fingerprint, variable, requested_lat, requested_lon, start, end]
```

成功还必须校验 payload 的身份字段、维度、坐标升序、数值有限、日期数组与 values 等长；不能只看 HTTP 200。

区分 `requestedDate` 与 `field.date`：加载新日期时可以保留旧场，但标题仍显示旧 `field.date`，并另显示“正在加载所选日期”；不允许把旧值标成新日期。切换变量时清空旧场与旧曲线，不让温度数值配上 DU 单位。

- 409 `dataset_version_changed`：清空本场景缓存，重取一次元信息后等待用户继续，不无限自动重试。
- 503 `dataset_unavailable`：保留可重试错误状态与安全原因码，不显示旧数据作为当前成功结果。
- `AbortError` 不弹错误。

只允许有界内存缓存：区域序列每 fingerprint 每变量最多一份，点序列最多一个当前点，场只保留当前成功帧。**不写入 `predictCache`、认证会话或 SQLite。**

## 页面与文案

页面层现在默认挂载行星观测台（顶部观测条件工具栏、全幅行星画布、独立时间轨道、底部分析区），布局与控件位置见[共用分析工作台](earth-analysis-workbench.md)。地球的球体变量、分析年份与数据源在顶部工具栏直接选择，设置内容拆到工具栏的**图层 / 点位 / 显示**三个面板；二维地图仍可从“显示”面板的“球体视图 → 二维地图”进入，也是 WebGL 不可用时的自动降级入口。以下约定对观测台与兼容保留的二维实现同时有效。

- 桌面在 70px 全局导航下方使用正常文档流，最大内容宽 1440px；宽度 ≥1024px 且工作区高度 ≥576px 时使用观测台两档布局，否则（宽度小于 1024px，或工作区短于 576px）退回文档流单列；390px 屏宽无横向溢出。
- 数据说明卡显示来源 MERRA-2、数据频率、日期或 UTC 时间范围、原始网格与显示降采样、全球覆盖、原始单位和聚合口径，全部来自 descriptor；ID、版本与日历放在折叠详情区，SHA 与内部 schema 不占主操作流。
- 全局设置中的火星臭氧/温度单位**不作用于地球**；地球界面只显示本场景实际使用的原始单位。
- 风控件是 `U10M`/`V10M` 两个明确选项，不用其中一个表示风速；辐射标签明确为地表入射短波（SWGDN），不用火星太阳相位解释。
- 中英文文案在 `frontend/src/i18n/zh.js`、`en.js` 的 `overviewPlanet.*` 与 `earthOverview.*` 下；所有按钮与错误状态都有翻译。

## 能力声明

以下为归档日频响应记录；当前 `GET /api/datasets/earth_merra2_daily_v2` 和 `earth_merra2_daily_v1` 均返回 409 `dataset_retired`：

```json
"capabilities": {"metadata": true, "web_overview": true, "training": false, "trained_prediction": false}
```

`web_overview=true` 表示**入口已适配**，与文件是否齐备分开说明：缺包时仍为 `true`，但数据接口返回 503，前端同时要求 `availability=available` 才请求数据。两个火星训练 ID 的 `web_overview` 保持 `false`（火星总览仍走原来源选择协议）。manifest 中“读取器尚未接注册表”一类构建期说明已被过滤，`limitations` 只保留物理数据限制与当前应用能力说明。

## 手工验证

从后端目录（`AstraAtmos_backend/backend/`）执行，只读、不启动数据库或训练：

```powershell
& "$env:CONDA_PREFIX\python.exe" -B -c "from config import EARTH_MERRA2_DIR, EARTH_MERRA2_V1_DIR; from services.dataset_registry import DatasetRegistry; from services.earth_overview_service import EarthOverviewService; r = DatasetRegistry(EARTH_MERRA2_DIR, earth_dataset_id='earth_merra2_daily_v2', legacy_earth_package_dir=EARTH_MERRA2_V1_DIR); d = r.get_dataset('earth_merra2_daily_v2'); s = EarthOverviewService(r); f = s.get_field(dataset_id=d['dataset_id'], expected_fingerprint=d['dataset_fingerprint'], date='2020-02-29', variable='TO3'); print(f['date'], len(f['field']), len(f['field'][0]), f['units'], f['coverage'], f['statistics']['regional_mean'])"
```

测试：

```powershell
$earthOverviewTemp = Join-Path $env:TEMP ('.earth-overview-test-' + [guid]::NewGuid().ToString('N'))
& "$env:CONDA_PREFIX\python.exe" -m pytest tests/test_earth_overview_service.py tests/test_earth_overview_routes.py -q --basetemp "$earthOverviewTemp"
```

前端（从 `frontend/` 执行）：

```powershell
node --test src/services/datasets.test.js src/pages/DataOverviewPage/EarthOverview/earthOverviewModel.test.js src/pages/DataOverviewPage/EarthOverview/earthRequestCoordinator.test.js src/pages/DataOverviewPage/EarthOverview/earthMapGeometry.test.js src/pages/DataOverviewPage/EarthOverview/earthSeriesModel.test.js
node --test
npm run build
```

共用工作台的测试命令见[共用分析工作台](earth-analysis-workbench.md#测试)。

## 当前边界

- 地球已开放三小时数据分析工作台，观测档展示球体与时间/点位轨，分析档展示左侧条件与主题组合看板，保留单项分析入口；协议与控件位置见[共用分析工作台](earth-analysis-workbench.md)。
- 地球总览仍**没有**：昼夜分析（已有三小时采样，分析未接入）、风场粒子、派生风速、自选多边形区域、重网格、平滑/插值、臭氧单位换算、总览导出、PFI、Earth/Mars 数值叠加或跨星球比较。
- 三小时 `earth_merra2_3hourly_v1` 的官方 DLinear / 独立契约上传模型 56→24 训练、总览及有参考真值历史预测使用 UTC 时间轴。日频运行入口已停用；三小时包缺失时不回退。无参考真值的未来外推、Earth/Mars 混合比较和不兼容窗口的上传模型未开放；完整 240×480 预测场由预测 API 返回，浏览器下载、JSON 解析和内存成本高于总览降采样地图。
- 全球和区域均值按球面单元面积加权。
- 数据源为三小时区间均值，年度图按每天八个源样本聚合，不使用独立日频发布。
- 二维实现（`EarthOverviewScene` 及其组件）作为兼容层保留，页面**不再挂载**它：`EarthMap2D` 被观测台直接复用，`EarthTimeline` 与 `EarthSeriesPanel` 仅供兼容二维场景使用，二维地图仍可从“显示”面板进入并在 WebGL 不可用时自动降级；`DetailPanel`、`SidebarMenu` 同样不再由页面渲染，分析主图由 `AnalysisDock` 承担，Earth 点位结果在观测档右侧曲线展示。

## v1 历史验证记录

2026-09-23 在本地实际执行，使用 conda 环境 `AstraAtmos` 的解释器：

- **v1 真实数据只读比对**：对 `2020-01-01`、`2020-02-29`、`2021-12-31` × 五个变量，逐值比对 `field` 与本地 NetCDF（数组完全相等），**45 个点位值**与 NetCDF 完全一致，**15 个区域均值**与独立手算的纬度余弦加权结果一致（`rtol=1e-10`），另核对 15 组单位与坐标轴。
- **后端测试**：`test_earth_overview_service.py`（77）与 `test_earth_overview_routes.py`（44）通过；第一步与火星总览回归（`test_dataset_registry.py`、`test_dataset_routes.py`、`test_earth_dataset.py`、`test_training_dataset_identity.py`、`test_analysis_overview_uploaded_sources.py`、`test_mcd_overview_service.py`）共 94 项通过，Earth 训练拒绝测试保持通过。
- **前端测试与构建**：`node --test` 全量 345 项通过；`npm run build` 成功。
- **真实浏览器验收**（系统 Edge + `playwright-core`，headless，1440/900/390px）：**43/43 项通过**，该次记录覆盖当时默认火星、切换地球、初始日期与 DU、1519 个数据格、闰日 `2020-02-29`、五变量原始单位、火星单位设置不改变地球数值、地图点选与标记、覆盖区外提示且不发点位请求、逐日播放不跳日、暂停与从首日重播、慢网下“正在加载所选日期”且不把旧值标成新日期、快速切换日期与变量后只保留最后选择、Earth→Mars→Earth 保留选择、中英文与深浅主题、无横向溢出、底图仅来自本地资源、控制台无新增错误。当前入口已改为默认地球。
- 截图保存在工作区 `<工作区>\earth-overview-verification\screenshots\`（桌面地图、点位选中、区域外无数据、浅色英文、900px、390px），运行截图不属于仓库内容。

交接复核说明：上述浏览器 JSON 中 43 项确实均标记通过，但“火星单位设置不改变 Earth 数值”一项记录为 `-- DU → -- DU`，测试发生在选中有效点位之前。该断言仅证明占位文本相同，真实数值的单位隔离仍需固定日期、指纹及网格点、等待有限数值后补测；已纳入[第三步训练接入方案](plans/2026-09-23-earth-dlinear-training.md)的联调验收。第二步报告摘要的“120 passed”也与 service 77 + routes 44（合计 121）不一致，汇总数字应以实际测试输出为准。本次复核检查了源码与已有验收记录，没有重新运行浏览器或全量回归。
