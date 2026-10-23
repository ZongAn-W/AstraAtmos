# 服务器数据集注册表与训练任务身份

当前公开目录列出 `openmars_mcd`、`mcd_overview`、`earth_merra2_3hourly_v1`、`earth_merra2_3hourly_5deg_v1`。5° 发布用于训练和历史回测，数据总览默认仍为原 0.75° 发布。日频 v1/v2 的身份保留用于识别历史记录，但目录详情、数据快照、总览/分析、训练与历史预测均返回 409 `dataset_retired`。三小时是开发与生产唯一默认 Earth ID，缺包不回退；详见[日频停用约定](earth-dataset-retirement.md)。下文日频注册、读取、兼容接口和验证数量均记录历史实现。

实施方案见 [数据集注册与训练任务身份实施方案](plans/2026-09-23-dataset-registry-and-task-identity.md)。文档最近核对日期：**2026-10-16**（新增独立 5° 训练/回测身份与空间契约）；三小时注册、分块校验、官方 DLinear / 独立契约上传模型 56→24 训练与 UTC datetime 历史回测已实现，日频运行入口已停用。完整包可用不等同于真实模型精度验收，历史验证见文末及[三小时专题](earth-merra2-3hourly.md)。

[二维地球数据总览实施方案](plans/2026-09-23-earth-overview-2d.md) 对应阶段已实现；当前总览行为见 [二维地球数据总览](earth-overview.md)，活动全球发布见[三小时数据契约](earth-merra2-3hourly.md)，归档日频契约见 [地球小数据包](earth-compact-dataset.md)。

[DLinear 地球训练接入实施方案](plans/2026-09-23-earth-dlinear-training.md)（2026-09-27 修订版）规定 Earth 训练与历史预测能力、任务身份与 checkpoint 绑定、通道/归一化协议及跨场景隔离；**该方案已实施**，实现契约见 [地球训练与历史预测](earth-training.md)。

## 已实现范围

| 能力 | 状态 |
| --- | --- |
| `GET /api/datasets` 与 `GET /api/datasets/{dataset_id}` | 已实现，公开只读 |
| 活动目录 `openmars_mcd`、`mcd_overview`、`earth_merra2_3hourly_v1`、`earth_merra2_3hourly_5deg_v1` | 已实现；日频 ID 仅保留历史身份 |
| Earth 小包（manifest + NetCDF）发布指纹与内容一致性校验 | 已实现 |
| `GET /api/datasets/{dataset_id}/overview/*` 三个地球总览接口 | 已实现，见 [二维地球数据总览](earth-overview.md) |
| 训练任务五个身份列、旧任务幂等回填 | 已实现 |
| 新训练请求严格校验数据集与 profile；未知 ID 和不支持的模型配置在创建任务前拒绝 | 已实现；三小时官方/独立契约上传模型 56→24，日频已停用 |
| 地球总览页面（三维全球球体、三小时播放、点位与区域曲线） | 已实现，见 [共用分析工作台](earth-analysis-workbench.md) |
| `GET /api/analysis/earth/overview/*` 四个地球分析接口（`context`、`research-suite`、`spatial-diagnostics`、`polar-dynamics`）与 `POST .../insight` | 已实现，见 [共用分析工作台](earth-analysis-workbench.md) |
| 地球年度分析、极区统计（`\|latitude\| >= 60°`）与图表 AI 解读 | 已实现 |
| 地球昼夜变化 | 三小时已有日内采样，分析尚未接入 |
| 风场粒子、派生风速 | 未实现 |
| Earth 训练（官方 DLinear、可选 Earth 通道、单文件 checkpoint 与完成前严格重载） | 三小时网页与后端已实现，见 [地球训练与历史预测](earth-training.md) |
| 地球历史回测 API（三小时 UTC datetime、可选起点、预测/参考/残差与指标） | 已实现，56→24；日频任务已停用，见 [地球训练与历史预测](earth-training.md) |
| 持久性基线、Earth/Mars 混合比较、无真值外推 | 未实现 |
| 用户上传数据注册、在线下载、数据集管理后台 | 未实现 |

`capabilities.web_overview=true` 表示总览与三维分析工作台入口已适配，`capabilities.training=true` 表示训练入口已接通，`trained_prediction=true` 表示历史回测入口已接通；数据是否可用仍由 `availability` 单独表达。原 0.75° 三小时发布为 `metadata/training/trained_prediction/web_overview=true`，新 5° 发布除 `web_overview=false` 外相同，official DLinear 与符合独立契约的 uploaded 模型使用 56→24 profile；旧日频/Mars 上传结论不能证明三小时兼容。总览和历史回测使用 UTC timestamp；日频请求在读取归档包或 checkpoint 前拒绝。缺包时目录仍返回 200，创建训练的数据请求返回 503；已绑定三小时任务回测时的缺失或损坏发布返回 409 `dataset_version_changed`，权限无法验证仍返回 503。

## 注册身份：ID、版本与指纹分开

| 数据集 ID | planet | dataset_version | schema | 说明 |
| --- | --- | --- | --- | --- |
| `openmars_mcd` | `mars` | `null` | `null` | 现有服务器目录尚无不可变发布版本 |
| `mcd_overview` | `mars` | `null` | `null` | 同上，与前一来源区分 |
| `earth_merra2_daily_v1` | `earth` | `v1` | `astraatmos_earth_daily_v1` | 保留原 31×49 区域抽样包，原始文件和哈希不变 |
| `earth_merra2_daily_v2` | `earth` | `v2` | `astraatmos_earth_daily_v1` | 归档全球 36×72、5° 单元，运行接口已停用 |
| `earth_merra2_3hourly_v1` | `earth` | `v1` | `astraatmos_earth_3hourly_v1` | 独立全球 240×480、0.75° 单元，三小时 UTC 中心标签，分块校验 |
| `earth_merra2_3hourly_5deg_v1` | `earth` | `v1` | `astraatmos_earth_3hourly_v1` | 独立全球 36×72、5° 三小时训练发布，完整全球窗口 |

- `schema` 是文件格式协议名，不是数据集 ID，也不是数据版本。
- `manifest_sha256` 指 `manifest.json` **原始字节**的 SHA-256；日频发布继续按其固定原始字节哈希验证。
- `data_sha256` 指 NetCDF 文件原始字节的 SHA-256。
- 日频 `dataset_fingerprint` 组合 `dataset_id`、`dataset_version`、`manifest_sha256`、`data_sha256`，算法固定为对四字段做 `json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=True)` 后再取 SHA-256 十六进制；旧哈希和组合算法不变。
- 三小时 fingerprint 与离线构建器保持一致：使用 manifest 身份内容摘要（`manifest_content_sha256`）作为组合函数的 manifest 参数，另返回原文件字节 SHA。ID 和版本始终参与组合，不能与日频共用身份。
- 三小时 manifest 的 `fingerprint_rule` 决定内容摘要的取法，构建器、服务端与独立验证器各有一份实现并交叉验证：
  - 缺失或 `1`（legacy）：对除 `dataset_fingerprint` 外的整份 manifest 取 canonical 摘要；既有已注册发布包属于这一类，其 `manifest_content_sha256` 与 `dataset_fingerprint` 冻结，不因修复身份策略而漂移。摘要覆盖整份 manifest，因此给既有包补写 `fingerprint_rule: 1` 本身就会改变其内容摘要——既有发布包不得改写。
  - `2`（内容绑定）：再排除顶层 `source_mtimes` 与 `fingerprint_rule`，并把 `source_files` 投影为 `date/product/path/bytes/sha256`。`mtime_ns`（文件系统元数据）与 `name`（`path` 的冗余后缀）显式排除：纯 mtime 漂移、`name` 改写、键顺序变化都不改变身份；`path`、`bytes`、`sha256` 或清单成员增减必然改变身份并被拒绝或作为新发布上报。
  - 其他任何值（0、负数、字符串、浮点、布尔）报错，fail-closed，不静默回退 rule 1。
- `source_sha256` 校验按 `source_inventory_version` 分流：缺失或 `2` 校验含 `name`/`mtime_ns` 的完整清单，`3` 只校验身份字段投影。两侧必须一致，否则既有注册包会被误判损坏；新增字段（`fingerprint_rule`、`source_mtimes`）在消费方全部可选。
- 消费既有发布包时使用**短路**：identity 三元组取 manifest 的 `dataset_id`/`dataset_version`/`data_sha256` 与已冻结的 `manifest_content_sha256`（registry descriptor 或共享验证证明取得），不重新投影 1462 条源清单；冷路径退回 rule 1 全文摘要，两者必须得到同一 fingerprint。
- 格式化 JSON 只改变 `manifest_sha256` 字节摘要，不改变三小时产品 fingerprint。
- Mars 无版本记录时如实返回 `null`，不给旧来源编造 `v1`，也不用文件修改时间冒充版本。

保留的 Earth v1 发布指纹：

```text
manifest.json           1d346a059aff57964cb726dde7113fc495adb4b8245f533911dc6f3d63e52d7e
earth_merra2_daily.nc   c21c4ddcb03a69d2eb504f746c86b3a15cf33a06929044f50da768bdfc45cf74
dataset_fingerprint     74ce14752cb83932b29b458d9c19a61804861c270e6fec33f55316778ad64d31
```

归档 v2 固定发布 SHA：manifest 为 `935ca37bd3064772a370db6873b605e12b85c031281c2b34d8fbc49ab11f0702`，NetCDF 为 `d280a17cb291e1568ccedd1638cb2fadcc5cb8d89bb8ed0d4d51987e8e181396`。原 v2 校验核对两轴实际 cell bounds、中心、连续性、全球范围及训练日归一化摘要；该协议仅用于历史追溯。

对两个旧日频 ID，把数据和 manifest 一起替换也不构成原发布：原固定 expected SHA 始终保持。三小时发布由服务器配置目录中的 manifest 定义，校验自报 NetCDF SHA、canonical fingerprint 与实际科学契约，不固定为开发 smoke 的哈希；服务器替换为另一个自洽包会产生另一个 fingerprint。客户端无法指定发布目录或覆盖身份。

## 对外查询协议

两个接口都是公开只读目录，不返回文件绝对路径、内部异常堆栈或用户信息。

```text
GET /api/datasets
200 {"items": [DatasetDescriptor, ...], "default_earth_dataset_id": "earth_merra2_3hourly_v1"}

GET /api/datasets/{dataset_id}
200 DatasetDescriptor
409 {"detail": {"code": "dataset_retired", "message": "Daily Earth datasets are retired; use earth_merra2_3hourly_v1 and train a new model"}}
404 {"detail": {"code": "unknown_dataset", "message": "Unknown dataset id"}}
```

列表稳定按 `openmars_mcd`、`mcd_overview`、`earth_merra2_3hourly_v1` 顺序返回。活动发布文件缺失仍返回 200 和明确状态，不从列表消失，也不回退旧包。日频详情返回 409 `dataset_retired`；无效查询 ID 返回 404。

`default_earth_dataset_id` 由 `ASTRAATMOS_DEFAULT_EARTH_DATASET_ID` 配置，只接受 `earth_merra2_3hourly_v1`，未设置时也使用该值。开发与生产模板一致，旧日频值会在启动时拒绝。Earth 页面通过该 catalog 字段选择数据集；不会改写旧任务、checkpoint、时间轴或窗口，也不会自动迁移日频模型。三小时包缺失时列表照常返回不可用原因，应用启动本身不依赖该包。

`DatasetDescriptor` 字段：

| 字段 | 类型及约定 |
| --- | --- |
| `dataset_id`、`display_name` | string，固定 ID 与展示名 |
| `planet` | `mars` 或 `earth` |
| `dataset_version` | string 或 null |
| `schema` | string 或 null；Earth 使用文件协议名 |
| `availability` | `available` / `missing` / `invalid` / `unverified` |
| `availability_reason` | 稳定错误码或 null，不返回底层异常文本 |
| `manifest_sha256`、`data_sha256`、`dataset_fingerprint` | 验证通过的 64 位小写十六进制 SHA 或 null |
| `capabilities` | `{metadata, training, web_overview, trained_prediction}`，表示入口是否适配，与文件是否齐备分别说明 |
| `time` | 三小时 `kind=datetime`，ISO UTC Z 标签、间隔、时间边界和样本数；Mars 为 `mars_ls` |
| `frequency_hours`、`step_unit`、`step`、`grid_shape` | Earth 顶层便捷字段：3 / hour / 3 / `[240,480]` |
| `training_profile` | 三小时官方/独立契约上传模型 56→24、TO3 DU、三小时间隔和独立 implementation_id；可用性同时检查 availability |
| `grid` | Earth 含 shape、维度顺序、范围、`cell_bounds`、步长、顺序、覆盖类型、是否循环及完整坐标数组；未验证的 Mars 为 null |
| `variables` | Earth 变量列表（`id`/`label`/`units`/`role`）；未核实 Mars 文件变量时为空列表 |
| `channel_order` | Earth 固定五通道；Mars 为空列表 |
| `splits` | Earth manifest 划分；Mars 为 null |
| `limitations` | string[]，数据边界说明 |

归档 v1 验证成功响应（历史示例，当前请求返回 409；原 v2 使用独立 ID/哈希、36×72、±90°/±180°、`coverage=global`、`wrap_longitude=true`）：

```json
{
  "dataset_id": "earth_merra2_daily_v1",
  "display_name": "MERRA-2 daily ozone compact v1",
  "planet": "earth",
  "dataset_version": "v1",
  "schema": "astraatmos_earth_daily_v1",
  "availability": "available",
  "availability_reason": null,
  "dataset_fingerprint": "74ce14752cb83932b29b458d9c19a61804861c270e6fec33f55316778ad64d31",
  "capabilities": {
    "metadata": true, "web_overview": true, "training": true, "trained_prediction": true
  },
  "time": {
    "kind": "date", "calendar": "proleptic_gregorian",
    "start": "2020-01-01", "end": "2021-12-31", "count": 731, "step": 1, "step_unit": "day"
  },
  "grid": {
    "shape": [31, 49], "dimension_order": ["lat", "lon"],
    "latitude_range": [-60.0, 60.0], "longitude_range": [-120.0, 120.0],
    "latitude_step": 4.0, "longitude_step": 5.0,
    "latitude_order": "ascending", "longitude_order": "ascending",
    "coverage": "regional", "wrap_longitude": false,
    "latitude_values": ["…31 个数值…"], "longitude_values": ["…49 个数值…"]
  },
  "channel_order": ["TO3", "U10M", "V10M", "T2M", "SWGDN"],
  "variables": [
    {"id": "TO3", "label": "Total column ozone", "units": "DU", "role": "target_and_input"},
    {"id": "U10M", "label": "10 m eastward wind", "units": "m s-1", "role": "optional_input"},
    {"id": "V10M", "label": "10 m northward wind", "units": "m s-1", "role": "optional_input"},
    {"id": "T2M", "label": "2 m air temperature", "units": "K", "role": "optional_input"},
    {"id": "SWGDN", "label": "Surface incoming shortwave flux", "units": "W m-2", "role": "optional_input"}
  ],
  "splits": {
    "train": {"start": "2020-01-01", "end": "2020-12-31", "days": 366},
    "validation": {"start": "2021-01-01", "end": "2021-06-30", "days": 181},
    "test": {"start": "2021-07-01", "end": "2021-12-31", "days": 184}
  },
  "limitations": ["Regional point-subsampled daily means; not global coverage", "…"]
}
```

详情在 `grid.latitude_values`、`grid.longitude_values` 返回全部坐标（v2 为 36 + 72 个数值，不另设坐标或分页接口）；列表使用同一结构。

Mars 两个注册项的 `availability=unverified`、`availability_reason=legacy_dataset_not_probed`。本阶段不扫描大型 Mars 目录，`training=true` 与 `trained_prediction=true` 表示已有服务入口，实际执行仍使用原有数据检查；`web_overview=false` 表示这两个训练身份尚未通过新目录协议驱动总览，不能与现有总览的 MCD/OpenMARS 图层入口混为一谈。

### 状态与错误码

`availability` 与 `availability_reason` 的固定组合：

| availability | availability_reason | 含义 |
| --- | --- | --- |
| `available` | `null` | 对应包的 manifest、SHA、fingerprint 与科学内容全部通过验证 |
| `missing` | `package_missing` | manifest 或该 ID 固定名称的 NetCDF 不存在（目录不存在同此） |
| `invalid` | `manifest_fingerprint_mismatch` | 日频 manifest 字节 SHA 与固定发布不符；或三小时 fingerprint 与 canonical manifest 内容不符 |
| `invalid` | `data_fingerprint_mismatch` | NetCDF 字节 SHA 与相应发布声明不符 |
| `invalid` | `invalid_manifest` | manifest 非 JSON 对象，或 `data_file` 不是固定文件名 |
| `invalid` | `manifest_metadata_mismatch` | manifest 自报的数据 SHA/字节数或逐项元数据（日期、维度、通道、单位、值域、划分、来源 SHA）与文件不一致 |
| `invalid` | `invalid_dataset` | 底层数据协议不通过：日期不连续、网格非均匀、单位不符、时间坐标非 Gregorian 日历等 |
| `invalid` | `package_changed_during_verification` | 校验过程中文件签名发生变化；该结果不进入缓存，下次请求重新校验 |
| `unverified` | `legacy_dataset_not_probed` | Mars 旧目录，本阶段不探测 |

详情接口对不存在或未注册的 ID 返回 404：

```json
{"detail": {"code": "unknown_dataset", "message": "Unknown dataset id"}}
```

三小时实际读取权限失败使用 `unverified/package_unreadable`；负数 NetCDF 内容格式错误按 `invalid/invalid_dataset` 处理。底层详细原因只写日志，对外不回传路径或异常堆栈。2026-10-07 本轮只读 descriptor 查询完整两年包为 `available`；早期缺包或 7 天 smoke 的状态记录不能作为当前训练/预测验收结论。

### 校验与缓存

日频校验顺序保持：两个文件存在 → 固定 manifest SHA → manifest 对象与固定文件名 → 固定 NetCDF SHA → 自报 SHA/字节数 → 数据协议 → 逐项元数据 → 前后签名。三小时先严格解析 JSON 和固定身份/文件名，再比对 canonical fingerprint、NetCDF SHA/大小，最后分块校验科学内容和 manifest 元数据，并核对前后签名。

- 日频文件名固定为 `earth_merra2_daily.nc`，三小时固定为 `earth_merra2_3hourly.nc`；不接受客户端路径或任意 manifest 路径。
- 元数据由文件真实计算得出，不写死 731 天或 31 × 49 来伪造成功。
- 时间 `calendar` 从解码时间编码读取，只接受 `standard`、`gregorian`、`proleptic_gregorian`；其他日历按 `invalid_dataset` 拒绝，不会标成火星时间。
- 校验由锁保护，按 ID 缓存描述符及快照，key 绑定两个文件的绝对路径、大小、mtime 和设备/文件 ID/变更时间；Windows 使用 NTFS ChangeTime，不能把创建时间当作变更时间。三小时服务端和训练子进程通过 `ASTRAATMOS_EARTH_TRAINING_CACHE_DIR/verification/` 共享完整校验生成的证明，证明包含验证器版本、原始/canonical manifest SHA、NetCDF SHA、fingerprint 和完整元信息摘要。命中前后复核源身份并重新读取 manifest；证明损坏、版本变化、文件替换或修改均重新完整校验。`get_earth_snapshot(..., force_full=True)` 保留强制完整校验，直接 `read_earth_3hourly_release()` 未指定缓存目录时仍完整扫描，详见[训练准备缓存](earth-preparation-cache.md)。日频保留不可写五场数组，三小时每变量最多扫描 8 步并只缓存元信息/只读坐标；完整体积留在磁盘。三小时未变化的缺失或无效结果也复用，校验中变化的结果不复用。API 返回深拷贝，不允许客户端污染缓存。
- 注册表构造不读取文件，首次查询时才校验。

三小时目录配置、训练与 descriptor 示例见 [三小时数据构建与训练](earth-merra2-3hourly.md)；当前完整包共 5848 步，`availability=available`。`ASTRAATMOS_EARTH_MERRA2_3HOURLY_DIR` 不改变日频两个配置项；旧数据库身份迁移、任务 JSON 和 checkpoint 不重写。三小时 binding 保留完整 UTC datetime、网格、发布 split、canonical manifest SHA 与 dataset-selected profile，实际 NetCDF 路径只经过服务端内部 spec，不返回客户端。

三小时按 split 独立生成 56→24 窗口，完整两年无缺失发布的 train/validation/test 时间窗口数为 2849/1369/1393；归一化仅扫描 train，每通道最多 8 步。runner 惰性读窗并以 24×48 空间块组批，一个时间窗口有 100 块，`batch_size` 是空间块数。所选通道缺失显式拒绝；checkpoint 使用独立 schema，绑定完整 240×480 网格、UTC 3 小时区间中心规则、通道、normalization 和 fingerprint，不能加载日频 checkpoint；DU 指标含总体、24 个 lead 与累计 24/48/72 小时。

三小时回测沿用 Earth context/run 接口，使用精确 UTC datetime 起点及其后 24 个真实目标时间戳。当前发布与任务、checkpoint 的 ID/版本/fingerprint/snapshot、时间轴、窗口和网格必须一致，reference 从相同发布读取；发布变化在独立缓存命中前即返回 409 `dataset_version_changed`。进程内 LRU 键含 planet、dataset_id、dataset_version、fingerprint、task_id、origin、全部 target timestamps 和 checkpoint SHA，不进入 Mars 的持久化缓存；详见[三小时历史回测 API](earth-merra2-3hourly.md#三小时历史回测-api)。

## 训练任务身份

`model_training_tasks` 增加五个可空列，不建立数据集数据库表：

```text
dataset_id               VARCHAR(80)
dataset_version          VARCHAR(40)
dataset_fingerprint      VARCHAR(64)
dataset_identity_status  VARCHAR(32)
dataset_snapshot         TEXT   -- JSON 对象
```

身份状态：

| 状态 | 含义 |
| --- | --- |
| `legacy_inferred` | 从旧超参数或已知默认规则推断出的 Mars 来源，版本与指纹未知 |
| `legacy_unknown` | 旧 JSON 非对象、损坏或来源不认识；快照保留 `planet=mars` 与原因，`dataset_id=null` |
| `unversioned` | 新创建的旧 Mars 任务，来源明确但尚未保存文件快照 |
| `verified` | 新 Mars checkpoint 保存完整目录绑定、文件清单/指纹与训练契约；父进程验证后保存兼容任务快照，两种 Mars 数据集均在预测前复核 |
| `legacy` | 旧裸权重或历史任务没有文件身份元数据；保持兼容但不声称已验证来源；部分身份元数据不等同于无身份 |

`mcd_overview` 是逻辑数据集身份；其训练和预测实际使用 `MCD_RAW_3H_DIR` 指向的原始全量 MCD MY24–MY35（默认 `data/MCD_Output_global_10m_ls_lst/`）。`data/mcd_overview/*.nc` 是生成的 overview 产物，不是当前训练预测的默认数据源。`openmars_mcd` 仍绑定 OpenMARS + `MCD_DIR`，两个身份不会交叉读取。

### Mars 年份分段与窗口

两种 Mars 数据集使用不同的年份分段来源，由 `services/mars_data_service.py` 统一形成全局时间轴与 block：

| 数据集 | 年份分段来源 | 文件内分段 |
| --- | --- | --- |
| `openmars_mcd` | OpenMARS 的实际 Ls 时间序列，以文件名起始 MY 锚定绝对年份；额外的 MY 标记用于校验 | Ls 年末回绕（相邻值下降超过 180°）产生新 MY，例如 `openmars_ozo_my27_ls358_my28_ls13.nc` 拆为 MY27 与 MY28 |
| `mcd_overview` | 原始 MCD MY24–MY35 年度文件名 | 每个文件仍对应一个 MY；三小时维度按既有方式展开，数据仍来自 `MCD_RAW_3H_DIR` |

一个 OpenMARS 文件可以产生多个 `MarsYearSegment`。每段记录 `mars_year`、全局 `start/end`、原始 `source_file` 与文件内 `file_start/file_end`，索引均为左闭右开。每段形成独立 block；同一年不同文件也保留原有的文件分块边界，不自动合并。检查点与任务快照的 `year_blocks[].segments` 保存来源与索引，快照的 `window_count` 按实际 block 统计。

窗口在完整合并时间轴上满足 `0 <= sample_start` 且 `sample_start + window + horizon <= time_count`，输入或目标可以跨 MY/file segment。全局数据与 Ls 不删除、不改写；数据不足以容纳窗口时不产生样本。`sample_starts` 和 `sample_mars_years` 描述实际合法起点，以及起点所属年份。预测用周期最近邻选择这些起点，并用起点对应的实际 block 返回 `block_index`、`mars_year`、输入 Ls 与目标 Ls；同一年多个文件的索引不会被去重年份压缩。请求仍只使用 `ls_start`，不增加火星年选择接口。

OpenMARS 的声明结束 MY 与实际回绕数量矛盾、缺失 MY 标记、非有限/越界 Ls、异常的小幅倒退或单文件 Ls 与臭氧时间长度不同，均明确报错，不自动排序或裁掉边界数据。仅一个 MY 标记的文件可以依据实际回绕推导后续年份；但 Ls 与文件名不足以恢复缺失整年的数据或证明文件之间没有时间缺口。

OpenMARS 窗口策略 `openmars_ls_year_segments_v2`、MCD 窗口策略 `mcd_merged_timeline_v2` 都纳入连续体积与预测分析缓存身份，使旧分段规则生成的缓存失效。新任务记录当前窗口策略。历史模型的权重和归一化参数不会被改写；重新训练后才会按当前严格 split 元数据和完整合并时间轴重新生成训练窗口。

### Mars 数据绑定与预测校验

Mars 模型测试入口 `action=test` / `get_test_results()` 也复用相同身份校验，防止通过测试入口绕过预测的数据源约束。

`services/mars_dataset_identity.py` 的 `canonical_dataset_identity()` 是 Mars 文件身份的唯一计算入口，供数据加载、训练快照、检查点和 `assert_identity_current()` 共用。它返回 `dataset_id`、`planet=mars`、`source_type`、`data_directories`、`file_manifest` 和 `dataset_fingerprint`；可读取当前文件元数据，也可用已捕获的目录/清单重现训练时身份，保存检查点时不会重绑后来变化的文件。

| 数据集 | `data_directories` 顺序 | 清单与既有指纹规则 |
| --- | --- | --- |
| `mcd_overview` | `[原始 MCD 目录]`，来自 `MCD_RAW_3H_DIR` | 每项为 `name/size/mtime_ns`；对既有 `{path, files}` 规范 JSON 取 SHA-256 |
| `openmars_mcd` | `[OpenMARS 目录, MCD 目录]` | 两个目录的每项都保存 `name/size/mtime_ns`，并带 `root=openmars/mcd`；对既有 `{openmars_dir, mcd_dir, files}` 规范 JSON 取 SHA-256 |

文件按既有自然顺序列出，JSON 使用 `sort_keys=True` 和紧凑分隔符，路径为解析后的绝对目录；不引入文件内容哈希。兼容字段 `data_dir` 只表示主目录（原始 MCD 或辅助 MCD），不能替代 `data_directories`。任务快照的 `manifest` 是 `file_manifest` 的兼容别名；检查点 `data_binding.file_fingerprint` 等于规范字段 `dataset_fingerprint`。父进程通过 `mars_checkpoint_identity_snapshot()` 将 `data_binding` 与训练契约还原为 `identity_snapshot()` 兼容字段，不再只复制部分别名。

官方与上传 Mars runner 都保存首次加载时的身份和统计量。父进程验证完整新 schema 检查点后，任务才保存 `dataset_snapshot`、相同 `dataset_fingerprint` 并标记 `verified`。预测无论使用 `mcd_overview` 还是 `openmars_mcd`，都会校验任务快照及可用的检查点身份；没有数据库快照时可从完整检查点恢复身份。完整 `data_directories` 绑定不要求存在兼容 `data_dir` 字段。数据目录或任一绑定目录中的文件修改、增加、删除后，返回 409 `dataset_version_changed`，在访问预测缓存、构造窗口或执行模型前终止；需要重新训练。

旧裸 `state_dict` 或只有注册表逻辑身份、没有文件目录/清单/指纹的历史任务，继续兼容并明确标记 `legacy`；旧裸权重若已有完整文件快照，也会复核该快照。身份部分缺失、任务指纹冲突或新 schema 缺少完整 `data_binding` 时，直接拒绝，不因字段名称不同而降级。以前生成的、遗漏 OpenMARS 目录或使用错误指纹的版本化检查点需要重训或经核实后单独迁移，不会自动补齐并标为 verified。

本契约保留文件名、大小与 `mtime_ns` 语义：如果内容被替换但大小与时间戳都被保持，当前指纹无法发现；同目录移动/恢复改变路径或时间戳也会影响身份。校验是文件元数据的时点检查，数据加载还会前后比较清单，但不提供文件系统级原子快照或写锁。生产数据应在训练与预测期间保持稳定。

### Mars 预测输出网格

`/api/predict/run` 的 `prediction`、`ground_truth` 和 `residual` 每个时间步返回同一套 `lat` / `lon`：`field[row][column]` 对应 `lat[row]`、`lon[column]`，坐标顺序和空间场同步保留。官方与上传 Mars 模型都从窗口加载器返回 latitude/longitude，经 `_predict_task_with_context()` 的共同校验后交给 `_fields_to_dicts()`，不按输出尺寸重新生成等间隔坐标。

`mcd_overview` 的原始 MCD 纬度由既有 loader 转成 36 行目标中心 `87.5, 82.5, …, -87.5`；37 个边界纬度形成相邻行均值，南到北源字段会同步翻转空间行；已经是 36 行中心的字段同样按实际坐标对齐。输出使用 loader 的目标纬度，并保留经度文件坐标及既有最多 72 列截取规则。`openmars_mcd` 直接采用 OpenMARS 文件的纬度/经度，不交换行列或反转标签；当前真实文件纬度北到南，存在浮点近似（如 `87.499992`），原值保留。不同文件的坐标不一致会明确拒绝拼接。

新 checkpoint 使用 `ScaledVolume.latitude/longitude`；官方 legacy 权重由 `_prepare_data()` 的 loader 元数据取得相同坐标。两条路径均不允许坐标缺失、空数组、NaN/Infinity、多维轴或坐标长度与空间场不匹配；也不以零替换坏坐标。合法输出不会只反转 latitude 而保留字段不动。

Mars 前端二维热力图按真实轴计算单元位置和刻度，保证北纬对应北侧；全屏热图、纬向平均和三维图均使用响应坐标。通用观测台的默认球面标注仍属于显示网格，不覆盖 Mars 预测数据轴。预测分析缓存包含 `loader_coordinates_v1` 输出策略，使以前保存的错误标签结果失效。Earth 的坐标、数据和显示流程独立，未改变。

### 旧任务回填规则

迁移按行判断：只有五个身份列**全部为 NULL** 才回填，因此已有绑定（含部分列已存在）的行原样保留。

| 旧 `hyperparameters` | 回填 `dataset_id` | 状态与 `binding_basis` |
| --- | --- | --- |
| `{"training_dataset":"openmars_mcd"}` | `openmars_mcd` | `legacy_inferred` / `explicit_training_dataset` |
| `{"training_dataset":"mcd_overview"}` | `mcd_overview` | `legacy_inferred` / `explicit_training_dataset` |
| JSON 对象缺少字段、值为 null、空字符串或空白 | `openmars_mcd` | `legacy_inferred` / `historical_default`（旧代码已有默认行为） |
| JSON 损坏、数组/标量、未知非空 ID、非字符串值、Earth ID | `null` | `legacy_unknown`，不猜来源 |

所有旧行的 `dataset_version`、`dataset_fingerprint` 保持 `null`。`dataset_snapshot` 至少保存 `planet`、`binding_basis`、`version_status`。迁移**不修改**旧 `hyperparameters`、状态、权重路径、指标、日志、标签及模型来源，也不移动或删除任何权重文件。

迁移在 `init_database()` 的建表事务中执行，位于既有“尽力补列”异常捕获之外：身份迁移失败会中止启动，避免返回看似健康但任务查询全部失败的应用；SQLite 已成功增加的列允许留下，补列与回填都幂等，修复后可重试。

### 新训练请求解析与错误

`TrainingStartRequest` 新增顶层可选 `dataset_id`。旧请求继续使用 `hyperparameters.training_dataset`；两处都未给出时仍默认 `openmars_mcd`。

| 请求组合 | 结果 |
| --- | --- |
| 顶层有效 ID + 旧字段缺省 | 使用顶层 |
| 仅旧字段有效 ID | 使用旧字段 |
| 两处有效且相同 | 接受 |
| 两处有效但不同 | 400 `dataset_id_conflict` |
| 显式空字符串、非字符串、未知 ID | 400 `invalid_dataset_id` / `unknown_dataset` |
| 日频 Earth ID | 409 `dataset_retired`；创建任务前拒绝，不改为三小时或火星 |
| 三小时 Earth ID | 独立 official/uploaded 56→24 路径；上传模型须通过具体 dataset_id 的独立 spec 和实际参数 dry-run，未知或失败返回稳定错误码 |
| 两处均为 null/未提供 | 默认 `openmars_mcd` |

Pydantic 对字段类型的基本校验错误仍使用 422；上表中 400/409 指服务层语义错误。未知 ID 不会被当作默认值。

客户端**不能**提交 `dataset_version`、`dataset_fingerprint`、`dataset_identity_status`、`dataset_snapshot` 来决定绑定：

- 顶层出现这四个字段 → 请求校验失败（422）。
- `hyperparameters` 中出现这四个字段或 `dataset_id` → 400 `client_identity_not_allowed`。

只支持“顶层新字段”和“旧 `training_dataset`”两个入口，避免出现第三套协议。

### 身份解析时机与持久化

身份解析、能力校验与绑定构造发生在**数据库操作、上传模型加载和子进程调度之前**；被拒绝的请求不会创建任务、不会加载上传模型包，也不会启动子进程。

新 Mars 任务把解析后的 ID 同时写入独立列与原有 `hyperparameters.training_dataset`（保证训练 CLI 与旧读取逻辑一致）；其他身份只写独立列，不会并入 `hyperparameters`，因此不会进入 CLI 参数。官方 `demo3.py` 与上传模型 `user_model_runner.py` 的 `main()` 在 argparse 解析后、数据加载前执行同一套严格校验；两个 runner 原有的宽容 normalizer 保留给旧任务读取与推理兼容，新训练不再经过它。

任务列表/详情响应新增 `dataset_id`、`dataset_version`、`dataset_fingerprint`、`dataset_identity_status`、`dataset_snapshot` 五个字段；`dataset_snapshot` 以 JSON 对象或 null 返回，数据库仍存 TEXT。无快照或快照损坏时返回 null，不会导致整个历史列表崩溃。旧客户端可忽略新字段。

`model_source`、`data_source=default`、上传模型与标签授权语义不变。新增身份列会改变新 Mars 任务的预测缓存产物指纹；本次不重写旧任务 JSON，因此旧缓存载荷保持有效。

## 配置

| 变量 | 用途 |
| --- | --- |
| `ASTRAATMOS_EARTH_MERRA2_DIR` | 覆盖已注册 Earth 小包目录，默认 `data/earth/merra2_daily_v2` |
| `ASTRAATMOS_EARTH_MERRA2_V1_DIR` | 独立指定旧兼容包目录，默认 `data/earth/merra2_daily_v1` |
| `ASTRAATMOS_EARTH_MERRA2_3HOURLY_DIR` | 独立三小时发布目录，默认 `data/earth/merra2_3hourly_v1`，不替换日频配置 |
| `ASTRAATMOS_DEFAULT_EARTH_DATASET_ID` | 开发/生产均为 `earth_merra2_3hourly_v1`；旧日频值须修改或移除，否则启动明确报错 |

日频 v1/v2 与三小时身份均不按请求互相映射。

相对路径相对后端启动目录解析；路径本身不通过 API 返回。注册服务在 `main.py` 的 lifespan 中装配为 `app.state.dataset_registry`，构造阶段不读取文件。

## 手工验证

从后端目录（`AstraAtmos_backend/backend/`）执行；真实小包只读校验不需要启动完整应用：

```powershell
& "$env:CONDA_PREFIX\python.exe" -B -c "from config import EARTH_MERRA2_DIR, EARTH_MERRA2_V1_DIR, EARTH_MERRA2_3HOURLY_DIR; from services.dataset_registry import DatasetRegistry; r = DatasetRegistry(EARTH_MERRA2_DIR, earth_dataset_id='earth_merra2_daily_v2', legacy_earth_package_dir=EARTH_MERRA2_V1_DIR, earth_3hourly_package_dir=EARTH_MERRA2_3HOURLY_DIR).get_dataset('earth_merra2_3hourly_v1'); print(r['availability'], r['frequency_hours'], r['grid_shape'], r['dataset_fingerprint'])"
```

回归测试（使用新的纯英文临时目录）：

```powershell
$datasetTests = @('tests/test_dataset_identity.py', 'tests/test_dataset_registry.py', 'tests/test_dataset_routes.py',
  'tests/test_training_dataset_identity_migration.py', 'tests/test_training_dataset_identity.py',
  'tests/test_earth_3hourly_registry.py', 'tests/test_earth_3hourly_data_contract.py')
foreach ($datasetTest in $datasetTests) {
  $datasetTestTemp = Join-Path $env:TEMP ('.dataset-registry-test-' + [guid]::NewGuid().ToString('N'))
  & "$env:CONDA_PREFIX\python.exe" -B -m pytest $datasetTest `
    -q --basetemp "$datasetTestTemp" -p no:cacheprovider
  if ($LASTEXITCODE -ne 0) { throw "Dataset regression failed: $datasetTest" }
}
```

`tests/conftest.py` 提供显式引用的 `earth_release` fixture：它在临时目录用真实尺寸（731 天、31 × 49）重建一个临时小包并返回其实际 SHA，不读取生产文件，也不产生 autouse 全局副作用。

## 当前边界

- 训练/总览入口含两个 Mars 身份及 `earth_merra2_3hourly_v1`；Earth 默认入口由 catalog 的 `default_earth_dataset_id` 决定，开发与生产均为三小时 v1。
- 地球已开放三维工作台与二维总览（见 [地球数据总览](earth-overview.md)）：三小时播放、点位曲线、面积加权均值和日聚合年度分析；风场粒子、派生风速、自选多边形区域、重网格、平滑/插值、臭氧单位换算、总览导出、PFI 和 Earth Copilot 尚未开放。
- 日频 v1/v2 仅保留历史任务/产物识别，运行入口返回 409 `dataset_retired`。`earth_merra2_3hourly_v1` 为 3 小时 UTC、0.75°×0.75°、240×480、TO3/U10M/V10M/T2M/SWGDN 的独立发布，官方 DLinear 与独立契约上传模型固定 56→24，训练、总览和有真值历史回测入口已接通。包状态独立于 capabilities。无参考真值外推和 Earth/Mars 混合比较未开放；高分辨率预测返回完整场，浏览器传输/JSON 解析存在性能限制。Earth 在火星推理、比较、PFI、`action=test` 与迁移来源路径一律 409，不回退到火星数据。契约与错误码见 [地球训练与历史预测](earth-training.md)。
- `web_overview=true` 只表示入口已适配，与数据是否齐备分开：缺包时数据接口返回 503，前端同时要求 `availability=available`。
- manifest 中“读取器尚未接注册表”一类构建期说明会被过滤；`limitations` 只保留物理数据限制与当前应用能力说明。
- Mars 身份不伪造版本；`openmars_mcd` 与 `mcd_overview` 何时成为不可变发布版本属于另一个数据发布任务。
- 注册身份快照不包含模型权重，不能独立复现训练；训练 checkpoint 已保存通道顺序、输入/输出窗口、归一化、网格、split 和绑定，复现仍需相同数据发布。三小时官方与上传链路使用合成 30 天完整网格训练/回测 smoke 验证；完整两年包已完成独立数据验证，本轮只读检查该包，未执行真实全两年模型训练或任务回测验收。

## v1 历史验证记录（不代表 v2 验收）

2026-09-23 使用 conda 环境 `AstraAtmos` 的解释器在本地执行：

- 本地固定小包只读校验（当时训练入口尚未开放，故记录 `capabilities.training=false`）：`availability=available`、731 天、31 × 49、步长 4°/5°、`wrap_longitude=false`、`TO3` 单位为 `DU`；两个实际 SHA 与上表发布定义一致。
- 新增测试全部通过：`tests/test_dataset_identity.py`（44）、`tests/test_dataset_registry.py`、`tests/test_dataset_routes.py`、`tests/test_training_dataset_identity_migration.py`（9）、`tests/test_training_dataset_identity.py`（41，含 HTTP 请求到数据库的集成用例）。
- 方案第 7 节列出的既有回归中，`test_training_channel_contract.py`、`test_training_channels.py`、`test_training_dataset_loader.py`、`test_official_training_runner.py`、`test_user_model_schema.py`、`test_training_tags.py`、`test_earth_dataset.py`、`test_prediction_analysis_cache_identity.py`、`test_uploaded_model_runner.py`、`test_uploaded_model_ls_inference.py` 全部通过（合计 265 项通过）。
- 应用装配与真实小包联调（同一阶段记录）：在真实 SQLite 数据库的临时副本上用 `TestClient` 启动完整 `main.app`，`GET /api/datasets` 返回三个条目、Earth 为 `available` 且指纹为 `74ce…d31`；`GET /api/datasets/missing` 返回 404；响应不含小包绝对路径；启动迁移把 14 条历史任务全部回填为 `legacy_inferred`，旧超参数、状态、权重路径与指标逐字不变。
- 第二阶段（二维地球总览）完成后，`capabilities.web_overview` 改为 `true`，并新增 `test_available_earth_does_not_claim_the_overview_is_unconnected` 等测试，确认目录不再宣称地球总览未接入；当时 Earth 训练拒绝、发布 SHA 与版本断言保持不变。总览接口与真实数据比对结果见 [二维地球数据总览](earth-overview.md) 的“验证记录”。
- 第三阶段（2026-09-28，地球训练与历史预测）完成后，Earth 的 `capabilities.training` 与 `trained_prediction` 改为 `true`，目录新增 `training_profile`，Earth 训练绑定改为 `verified` 并携带发布指纹与快照；上表中“Earth 训练拒绝”的旧断言已按新契约更新为「Earth 走独立路径、按 `availability` 决定可用性」。本阶段的实测结果、任务 ID 与错误码见 [地球训练与历史预测](earth-training.md)。

## 三小时 5° 训练发布

`earth_merra2_3hourly_5deg_v1` 为新增活动身份，schema 仍为 `astraatmos_earth_3hourly_v1`，网格固定 36×72、全球 5°，原三小时时间轴不变。注册表严格拒绝两个空间发布路径互换；新条目提供 training/trained_prediction，web_overview=false。任务、缓存和 checkpoint 使用独立 ID/发布指纹，默认总览仍为 0.75°。构建、上传契约和配置见 [5° 专题](earth-merra2-5deg.md)。
