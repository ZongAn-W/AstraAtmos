# 数据集注册与训练任务身份兼容 Implementation Plan

> **For agentic workers:** 使用 `executing-plans` 技能按任务执行，步骤使用复选框跟踪。本方案用于交给另一个对话实施；本次只编写方案。实施时无需重新讨论已经明确的范围与接口；只有发现与实际源码冲突的事实时才调整，并在交付中说明。

**Goal:** 建立可查询的服务器数据集注册表，识别 MERRA-2 地球小包，为训练任务增加可追溯的数据集身份，并阻止未知或尚未支持的数据集进入火星训练流程。

**Architecture:** 新增轻量注册服务，复用已有 Earth 读取器和 manifest，向前端提供只读元数据。数据库增量补充任务身份，兼容旧请求和旧火星任务；注册能力与训练能力分别声明，地球训练仍然关闭。

**Tech Stack:** Python、FastAPI、Pydantic v2、SQLAlchemy AsyncSession、SQLite/aiosqlite、xarray、netCDF4、pytest。沿用现有依赖，不引入新数据库、迁移框架或地图库。

---

## 1. 交接说明与范围

项目根目录为 `AstraAtmos`，父工作区为 `<工作区>`。下文文件路径均相对项目根目录，命令另行标注执行目录。

执行者先阅读 [README](../../README.md) 中的“快速接手”“模块与代码入口”“关键业务链路”“当前功能边界”“README 维护约定”，以及父目录 `AGENTS.md`。遵守禁止批量删除文件/目录的规则；不要通过 Python、shell 或测试清理脚本绕过该限制。

本方案是项目功能设计文档，放在项目 `docs/plans/`；仓库忽略 `docs/superpowers/`，因此不使用该目录，保证文档可随仓库交接。智能体技能及与项目功能无关的规则资源仍放父工作区。

### 本阶段交付

1. 固定注册 `openmars_mcd`、`mcd_overview`、`earth_merra2_daily_v1` 三个 ID。
2. `GET /api/datasets` 和 `GET /api/datasets/{dataset_id}`。
3. Earth manifest、数据内容指纹、日期、坐标、单位和范围的一致性验证。
4. 训练任务新增身份字段；旧任务幂等迁移；旧前端请求继续工作。
5. 新训练请求严格校验数据集；未知数据集和 Earth 训练请求均在创建任务之前拒绝。
6. 自动化测试、相关 README 和专题文档同步。

### 本阶段不实现

- 地球总览页面、二维地图、三维球体、日期播放、点位曲线。
- Earth DLinear 训练、可选 Earth 通道、checkpoint 内容改造。
- 日期预测 API、历史回测、持久性基线、模型比较、预测缓存新协议。
- 用户上传数据注册、在线下载、数据集管理后台、ERA5 或海冰。
- 全面重构 Mars 数据加载、训练循环、推理服务或前端状态。

Earth 在注册表可见、独立读取验证成功，并不表示网页展示或训练已经支持。服务端必须保持这种区别。

## 2. 已核对事实与执行前检查

方案编制于 2026-09-23，检查了源码、Git 状态和本地 Earth 数据；没有运行应用联调或完整测试。

当时已有未提交内容：`.gitignore`、`README.md` 的修改，以及 Earth 构建脚本、检查脚本、读取器、测试和专题文档。这是本方案的依赖，不能覆盖、删除或丢失。不要认为 Git HEAD 已包含这些文件。若执行者选择新 worktree，先把这些依赖完整带入；不要只检出 HEAD 后重新制作一套。

执行前从项目根目录运行：

```powershell
git status --short --branch
git log -5 --oneline
git diff --stat
rg --files -g AGENTS.md -g '!node_modules' -g '!.venv' .
```

重点源码与当前事实：

| 文件 | 当前事实和实施影响 |
| --- | --- |
| `services/earth_dataset.py`（后端内） | 有 `SCHEMA=astraatmos_earth_daily_v1`、`load_earth_dataset()`、`validate_dataset()` 和 `EarthOzoneWindows`；这是文件格式协议，不是注册表 ID |
| `scripts/build_earth_ozone_dataset.py` | 已生成 manifest，包含源 SHA、数据 SHA、变量、范围和划分，但没有 `dataset_id`；可以由注册表补充身份，无需重建小包 |
| `services/training_channels.py` | `_training_dataset()` 会将未知值回退到 `openmars_mcd`；新请求必须改为严格拒绝 |
| `models/training_scripts/demo3.py`、`training_backbones/user_model_runner.py` | 各有 dataset normalizer；训练命令入口也必须做严格校验，不能仅修 HTTP 层 |
| `services/training_service.py` | `start_training()` 在创建任务后启动子进程；身份解析必须放在数据库操作、上传模型加载和调度之前 |
| `routers/training.py` | 持有模块级 `training_service = TrainingService()`，不使用 `app.state.training_service`；只给 app.state 赋值不会替换它 |
| `database/init_db.py` | SQLite 使用 `PRAGMA table_info` 和 `ALTER TABLE ADD COLUMN`；当前部分异常会被吞掉；新增身份迁移失败不能继续运行不完整 schema |
| `services/training_channels.py` | `build_hyperparameter_args()` 会将所有非下划线开头字段转成 CLI 参数；身份字段应独立保存，不能直接塞进超参数 |
| `services/prediction_analysis_cache.py` | 当前指纹包含整个 `hyperparameters`；不要重写旧任务 JSON 造成所有旧缓存意外失效 |

表中简称均位于 `AstraAtmos_backend/backend/`。其余项目入口详见 README。

本地小包实际核对基准：

| 项目 | 值 |
| --- | --- |
| 目录 | `AstraAtmos_backend/backend/data/earth/merra2_daily_v1/` |
| 文件 | `earth_merra2_daily.nc`、`manifest.json` |
| 时间 | `2020-01-01` 至 `2021-12-31`，731 天 |
| 网格 | 31 个纬度 × 49 个经度，纬度间隔 4°、经度间隔 5° |
| 覆盖 | 纬度 [-60, 60]，经度 [-120, 120]，经度不循环连接 |
| 通道 | `TO3, U10M, V10M, T2M, SWGDN` |
| 单位 | `DU, m s-1, m s-1, K, W m-2` |
| 划分 | train 366 天，validation 181 天，test 184 天 |
| NetCDF SHA-256 | `c21c4ddcb03a69d2eb504f746c86b3a15cf33a06929044f50da768bdfc45cf74` |
| manifest 原始字节 SHA-256 | `1d346a059aff57964cb726dde7113fc495adb4b8245f533911dc6f3d63e52d7e` |

这些 SHA 是对当前文件实际计算所得。执行时重新读取验证；若文件不同，报告差异，不自动接受新文件为相同发布版本。

## 3. 固定设计决策

### 3.1 注册 ID、版本、内容指纹分开

| 数据集 ID | planet | dataset_version | 版本情况 |
| --- | --- | --- | --- |
| `openmars_mcd` | `mars` | `null` | 现有服务器目录尚无不可变发布版本 |
| `mcd_overview` | `mars` | `null` | 同上，保留与前一来源的区别 |
| `earth_merra2_daily_v1` | `earth` | `v1` | 以当前已核验的小包为固定发布内容 |

- `schema` 使用文件本来的 `astraatmos_earth_daily_v1`，不将它混用为 `dataset_id` 或数据版本。
- `manifest_sha256` 指 manifest **原始字节**的 SHA-256，不对 JSON 再序列化后计算；因此发布后连格式变动也应视为 manifest 变更。
- `data_sha256` 指 NetCDF 原始字节 SHA-256。
- `dataset_fingerprint` 组合数据 ID、版本、manifest SHA、data SHA，算法见任务 1。
- 注册表固定两个 expected SHA；将文件和 manifest 一起替换也不能冒充 v1。测试使用注入的发布定义和临时数据，不修改生产 SHA。
- 修改 v1 数据或 manifest 必须创建新 ID/版本及新发布定义；本阶段不提供更新或覆盖接口。
- Mars 无版本记录时返回 `null`，以明确状态解释。不要给旧来源编造 `v1` 或用文件修改时间冒充版本。

新任务的身份字段结构统一，但当前仍只有 Mars 可训练，因此新 Mars 任务记录 `unversioned` 状态。未来开放 Earth 训练时必须保存 `v1` 和已验证指纹。要把现有 Mars 目录变成不可变版本，需要另一个数据发布任务；不能为了字段非空伪造版本。

### 3.2 对外查询协议

两个接口均为公开、只读的服务器数据集目录；不包含个人数据、文件绝对路径、内部异常堆栈或用户信息。

```text
GET /api/datasets
200 {"items": [DatasetDescriptor, DatasetDescriptor, DatasetDescriptor]}

GET /api/datasets/{dataset_id}
200 DatasetDescriptor
404 {"detail": {"code": "unknown_dataset", "message": "Unknown dataset id"}}
```

列表稳定按上述三个 ID 的顺序返回。已注册但文件缺失仍返回 200 和状态，不能从列表消失，也不能让整个列表 500。无效 query ID 才是 404。

`DatasetDescriptor` 字段：

| 字段 | 类型及约定 |
| --- | --- |
| `dataset_id`, `display_name` | string，固定 ID 和展示名 |
| `planet` | `mars` 或 `earth` |
| `dataset_version` | string 或 null |
| `schema` | string 或 null；Earth 使用文件协议 |
| `availability` | `available / missing / invalid / unverified` |
| `availability_reason` | 稳定错误码或 null，不直接返回底层异常文本 |
| `dataset_fingerprint`, `manifest_sha256`, `data_sha256` | 经验证的 64 位小写十六进制 SHA 或 null |
| `capabilities` | `{metadata: bool, training: bool, web_overview: bool, trained_prediction: bool}`，表示入口已适配，与文件是否齐备分别说明 |
| `time` | Earth 见下例；Mars `{kind: "mars_ls", calendar: null, start: null, end: null, count: null, step: null, step_unit: null}` |
| `grid` | Earth 见下例；未验证的 Mars 返回 null，不从常量推断所有实际文件网格 |
| `variables` | Earth 变量列表；未核实 Mars 文件变量时返回空列表 |
| `channel_order` | Earth 固定五通道；Mars 返回空列表，不把可选气象通道误报为所有模型固定输入 |
| `splits` | Earth manifest 划分对象；Mars null |
| `limitations` | string[]，数据边界说明 |

Earth 验证成功后应返回如下结构。下列 SHA 和组合指纹已由本地文件计算核对，坐标数组通过下一段约定返回：

```json
{
  "dataset_id": "earth_merra2_daily_v1",
  "display_name": "MERRA-2 daily ozone compact v1",
  "planet": "earth",
  "dataset_version": "v1",
  "schema": "astraatmos_earth_daily_v1",
  "manifest_sha256": "1d346a059aff57964cb726dde7113fc495adb4b8245f533911dc6f3d63e52d7e",
  "data_sha256": "c21c4ddcb03a69d2eb504f746c86b3a15cf33a06929044f50da768bdfc45cf74",
  "dataset_fingerprint": "74ce14752cb83932b29b458d9c19a61804861c270e6fec33f55316778ad64d31",
  "availability": "available",
  "availability_reason": null,
  "capabilities": {
    "metadata": true,
    "training": false,
    "web_overview": false,
    "trained_prediction": false
  },
  "time": {
    "kind": "date", "calendar": "proleptic_gregorian",
    "start": "2020-01-01", "end": "2021-12-31", "count": 731,
    "step": 1, "step_unit": "day"
  },
  "grid": {
    "shape": [31, 49], "dimension_order": ["lat", "lon"],
    "latitude_range": [-60.0, 60.0], "longitude_range": [-120.0, 120.0],
    "latitude_step": 4.0, "longitude_step": 5.0,
    "latitude_order": "ascending", "longitude_order": "ascending",
    "coverage": "regional", "wrap_longitude": false
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
  "limitations": ["Regional point-subsampled daily means; not global coverage", "Earth web overview, training and prediction are not connected yet"]
}
```

详情额外在 `grid.latitude_values`、`grid.longitude_values` 中返回全部坐标，列表可以使用同一完整结构：31+49 个数值很小，无需另设分页或坐标 API。非 available 的 Earth 返回身份和固定能力，动态数据（时间/网格/变量/划分/指纹）返回 null 或空列表，不把声明当成已验证结果。time 此时保留 `kind=date`，其余未知值置 null。

Mars 两个注册项的 `availability=unverified`、`availability_reason=legacy_dataset_not_probed`。本阶段不扫描大型 Mars 目录；`training=true` 和 `trained_prediction=true` 表示已有服务入口，实际执行仍使用原来的数据检查。`web_overview=false` 表示这两个训练身份尚未通过新目录协议驱动总览，不能和现有总览的 MCD/OpenMARS 图层入口混为一谈。

### 3.3 身份字段与历史迁移

给 `ModelTrainingTask` 增加五个可空字段，不创建数据集数据库表：

```python
dataset_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
dataset_version: Mapped[str | None] = mapped_column(String(40), nullable=True)
dataset_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
dataset_identity_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
dataset_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True)
```

状态定义：

- `legacy_inferred`：从旧超参数或已知默认规则推断 Mars 来源，版本/指纹未知。
- `legacy_unknown`：旧 JSON 非对象、损坏或显式来源不认识；快照保留 `planet=mars` 和原因，`dataset_id=null`。
- `unversioned`：新创建的现有 Mars 任务，来源明确，数据源尚无发布版本。
- `verified`：为将来使用固定发布版本预留，本阶段不创建 Earth 训练任务。

历史回填规则：

| 旧 hyperparameters | 回填 dataset_id | 状态和依据 |
| --- | --- | --- |
| `{"training_dataset":"openmars_mcd"}` | `openmars_mcd` | `legacy_inferred`, `explicit_training_dataset` |
| `{"training_dataset":"mcd_overview"}` | `mcd_overview` | `legacy_inferred`, `explicit_training_dataset` |
| JSON 对象缺少字段、值 null 或空字符串 | `openmars_mcd` | `legacy_inferred`, `historical_default`；这是旧代码已存在的默认行为 |
| JSON 损坏、JSON 数组/标量、未知非空 ID、非字符串值 | null | `legacy_unknown`，不猜来源 |
| 已有任一身份字段非 null | 原样保留 | 不覆盖已有绑定，不重新认定版本 |

所有旧行 `dataset_version`、`dataset_fingerprint` 均保留 null。`dataset_snapshot` 至少保存 `planet`、`binding_basis`、`version_status`，可保存不超过 80 字符的旧来源标记用于排查。不修改旧 `hyperparameters`、状态、权重路径、指标、标签及模型来源。

任务列表响应新增五个同名字段；`dataset_snapshot` 以 JSON 对象/null 返回，数据库仍存 TEXT。旧客户端可忽略新字段。

### 3.4 新训练请求解析与错误行为

`TrainingStartRequest` 新增顶层 `dataset_id: str | None = None`。旧请求继续使用 `hyperparameters.training_dataset`；两处都未给出时仍默认 `openmars_mcd`。

```text
顶层有效 ID + 旧字段缺省      -> 使用顶层
仅旧字段有效 ID              -> 使用旧字段
两处有效且相同              -> 接受
两处有效但不同              -> 400 dataset_id_conflict
显式空字符串、数字、未知 ID   -> 400 invalid_dataset_id / unknown_dataset
已知 Earth ID               -> 409 dataset_training_not_supported
两处均为 null/未提供        -> 默认 openmars_mcd
```

Pydantic 对字段类型的基本校验错误可使用现有 422；上表 400 指服务层语义错误。不要把未知 ID 当作默认值。旧历史数据的宽容解析与新请求严格解析分开。

客户端不能提交 `dataset_version`、`dataset_fingerprint`、`dataset_snapshot` 或 `dataset_identity_status` 来决定绑定；顶层或 hyperparameters 中显式传入这些字段均拒绝，并说明由服务器生成。`hyperparameters.dataset_id` 也拒绝：只支持顶层新字段和旧 `training_dataset` 两个入口，避免第三套协议。

新 Mars 任务将解析后的 ID 同时写入独立字段与原有 `hyperparameters.training_dataset`；其他身份只写独立列。`model_source`、`data_source=default`、上传模型和标签授权语义不变。

## 4. 文件职责与改动清单

后端路径前缀均为 `AstraAtmos_backend/backend/`。

| 动作 | 路径 | 职责 |
| --- | --- | --- |
| 新增 | `services/dataset_identity.py` | ID 常量、严格请求解析、组合指纹、历史来源推断；纯函数，不导入 torch/数据库/文件读取 |
| 新增 | `services/dataset_registry.py` | 固定注册项、Earth 发布定义、检查缓存、描述符及训练绑定快照 |
| 新增 | `services/earth_dataset_metadata.py` | 校验 manifest/NetCDF 一致性并提取地球元数据；不重复实现训练窗口 |
| 新增 | `schemas/datasets.py` | 描述符、时间/网格/变量/能力模型及列表响应 |
| 新增 | `routers/datasets.py` | 只读列表和详情 |
| 新增 | `database/dataset_identity_migration.py` | 可独立在临时 SQLite 上测试的补列和历史回填 |
| 修改 | `config.py`、`.env.example` | `ASTRAATMOS_EARTH_MERRA2_DIR` 覆盖小包目录 |
| 修改 | `main.py` | 注册服务及路由 |
| 修改 | `database/models.py`、`database/init_db.py` | 五个字段、迁移调用和失败处理 |
| 修改 | `schemas/training.py`、`routers/training.py` | 请求字段、响应字段、准确的错误映射 |
| 修改 | `services/training_service.py` | 解析、持久化任务身份，不改变子进程协议 |
| 修改 | `services/training_channels.py` | 新训练配置拒绝未知/不支持的数据集 |
| 修改 | `models/training_scripts/demo3.py`、`training_backbones/user_model_runner.py` | CLI 训练入口增加同一严格校验 |
| 新增 | `tests/test_dataset_identity.py` | 解析、历史推断及 fingerprint |
| 新增或扩展 | `tests/conftest.py` | 仅提供显式引用的 Earth 发布 fixture，不增加 autouse 全局副作用 |
| 新增 | `tests/test_dataset_registry.py`、`tests/test_dataset_routes.py` | 数据一致性、不可用状态、接口契约 |
| 新增 | `tests/test_training_dataset_identity_migration.py` | 真旧表结构、幂等、回滚、不修改旧数据 |
| 新增 | `tests/test_training_dataset_identity.py` | 训练写入、错误请求无副作用、身份不进入 CLI |
| 修改 | `tests/test_training_channel_contract.py` | 将“未知来源回退”改为“明确拒绝” |
| 按实际需要修改 | `tests/test_uploaded_training_contract.py` 等现有 fixture | 为路由传参补可选字段，不改变原有断言含义 |
| 新增（项目根目录） | `docs/dataset-registry.md` | 已实现接口、状态、迁移和兼容约定 |
| 修改（项目根目录） | `README.md`、`docs/earth-compact-dataset.md` | 区分已注册与尚未接入网页/训练 |

不修改前端数据集选项。旧前端仍只发送原有两种 Mars 来源。

## 5. 按任务实施

### 任务 1：定义身份纯函数及严格解析

**文件：**新增 `services/dataset_identity.py`、`tests/test_dataset_identity.py`。

- [ ] 先写下面的输入矩阵测试，运行确认失败来自函数缺失或行为未实现。

```python
import pytest
from services.dataset_identity import DatasetRequestError, resolve_dataset_id

@pytest.mark.parametrize("top,hypers,expected", [
    (None, {}, "openmars_mcd"),
    (None, {"training_dataset": "mcd_overview"}, "mcd_overview"),
    ("mcd_overview", {}, "mcd_overview"),
    (" MCD_OVERVIEW ", {"training_dataset": "mcd_overview"}, "mcd_overview"),
    ("earth_merra2_daily_v1", {}, "earth_merra2_daily_v1"),
])
def test_resolve_dataset_id(top, hypers, expected):
    assert resolve_dataset_id(top, hypers) == expected

@pytest.mark.parametrize("top,hypers,code", [
    ("", {}, "invalid_dataset_id"),
    (None, {"training_dataset": 7}, "invalid_dataset_id"),
    ("missing", {}, "unknown_dataset"),
    ("openmars_mcd", {"training_dataset": "mcd_overview"}, "dataset_id_conflict"),
    (None, {"dataset_fingerprint": "forged"}, "client_identity_not_allowed"),
])
def test_rejects_bad_dataset_requests(top, hypers, code):
    with pytest.raises(DatasetRequestError) as exc:
        resolve_dataset_id(top, hypers)
    assert exc.value.code == code
```

- [ ] 实现错误类型和规范化。错误类型保持 `ValueError` 子类，便于旧服务测试兼容。

```python
DATASET_IDS = ("openmars_mcd", "mcd_overview", "earth_merra2_daily_v1")
MARS_DATASET_IDS = DATASET_IDS[:2]
SERVER_IDENTITY_FIELDS = frozenset({
    "dataset_version", "dataset_fingerprint", "dataset_identity_status", "dataset_snapshot",
})

class DatasetRequestError(ValueError):
    def __init__(self, code, message, status_code=400):
        super().__init__(message)
        self.code = code
        self.status_code = status_code

def _normalize_explicit_id(value):
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise DatasetRequestError("invalid_dataset_id", "Dataset id must be a non-empty string")
    result = value.strip().lower()
    if result not in DATASET_IDS:
        raise DatasetRequestError("unknown_dataset", "Unknown dataset id")
    return result

def resolve_dataset_id(dataset_id, hyperparameters):
    reserved = SERVER_IDENTITY_FIELDS | {"dataset_id"}
    if reserved.intersection(hyperparameters):
        raise DatasetRequestError("client_identity_not_allowed", "Dataset identity is generated by the server")
    top = _normalize_explicit_id(dataset_id)
    legacy = _normalize_explicit_id(hyperparameters.get("training_dataset"))
    if top is not None and legacy is not None and top != legacy:
        raise DatasetRequestError("dataset_id_conflict", "dataset_id conflicts with training_dataset")
    return top or legacy or "openmars_mcd"

def require_training_dataset(dataset_id):
    normalized = _normalize_explicit_id(dataset_id)
    if normalized is None:
        normalized = "openmars_mcd"
    if normalized not in MARS_DATASET_IDS:
        raise DatasetRequestError(
            "dataset_training_not_supported",
            "Earth training is not connected yet",
            status_code=409,
        )
    return normalized
```

- [ ] 指纹使用以下固定算法，并用不同 ID、manifest SHA、data SHA 的测试证明各项参与身份。

```python
import hashlib
import json

def build_dataset_fingerprint(dataset_id, dataset_version, manifest_sha256, data_sha256):
    payload = {
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "manifest_sha256": manifest_sha256,
        "data_sha256": data_sha256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
```

- [ ] 实现 `infer_legacy_identity(raw_hyperparameters) -> dict`，使用 3.3 的完整矩阵，返回五列的可直接写入值。快照的 JSON 使用 `json.dumps(..., sort_keys=True)`；不得调用严格新请求解析器来猜旧来源。

```python
def infer_legacy_identity(raw_hyperparameters):
    dataset_id = None
    basis = "invalid_hyperparameters"
    try:
        parsed = json.loads(raw_hyperparameters) if raw_hyperparameters else {}
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        raw = parsed.get("training_dataset")
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            dataset_id, basis = "openmars_mcd", "historical_default"
        elif isinstance(raw, str) and raw.strip().lower() in MARS_DATASET_IDS:
            dataset_id, basis = raw.strip().lower(), "explicit_training_dataset"
        else:
            basis = "unknown_training_dataset"
    return {
        "dataset_id": dataset_id,
        "dataset_version": None,
        "dataset_fingerprint": None,
        "dataset_identity_status": "legacy_inferred" if dataset_id else "legacy_unknown",
        "dataset_snapshot": json.dumps({
            "planet": "mars", "binding_basis": basis, "version_status": "unknown",
        }, sort_keys=True),
    }
```

- [ ] 运行 `python -m pytest tests/test_dataset_identity.py -q`，要求全部通过。

### 任务 2：实现 Earth 发布校验和元数据提取

**文件：**新增 `services/earth_dataset_metadata.py`、`tests/test_dataset_registry.py`；复用现有 `earth_dataset.py`。

- [ ] 在 `tests/conftest.py` 增加下面的非 autouse fixture。临时小包覆盖真实尺寸和完整闰年，不依赖本地实际数据；仅在使用该 fixture 的测试中导入数据依赖。

```python
import pytest

@pytest.fixture
def earth_release(tmp_path):
    import hashlib
    import json
    import numpy as np
    from scripts.build_earth_ozone_dataset import build_dataset
    dates = np.arange("2020-01-01", "2022-01-01", dtype="datetime64[D]")
    lat = np.arange(-60, 61, 4, dtype=np.float32)
    lon = np.arange(-120, 121, 5, dtype=np.float32)
    cube = np.broadcast_to(
        np.arange(len(dates), dtype=np.float32)[:, None, None],
        (len(dates), len(lat), len(lon)),
    )
    source = tmp_path / "source.npz"
    np.savez_compressed(
        source, **{key: cube + offset for offset, key in enumerate(("O3", "U", "V", "T", "S"))},
        lat=lat, lon=lon, time_iso=dates.astype(str),
        metadata=json.dumps({
            "source": "MERRA2", "temporal": "daily", "lat_stride": 8, "lon_stride": 8,
            "used_variables": {"O3": ["TO3"], "U": ["U10M"], "V": ["V10M"], "T": ["T2M"], "S": ["SWGDN"]},
        }),
    )
    path = build_dataset(source, tmp_path / "earth")
    return {
        "earth_package_dir": path.parent,
        "expected_manifest_sha256": hashlib.sha256(path.with_name("manifest.json").read_bytes()).hexdigest(),
        "expected_data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
```
- [ ] 先写成功及篡改测试：成功结果含真实日期、DU、坐标数组和 `wrap_longitude=false`；分别修改 manifest、NetCDF 后必须得到 invalid，不能接受更新后的自报 SHA。
- [ ] 提供 `read_earth_metadata(package_dir, *, expected_manifest_sha256, expected_data_sha256) -> dict`。目录来自服务器配置；`manifest.data_file` 必须等于 `earth_merra2_daily.nc`，不接受客户端路径或任意相对路径。
- [ ] 按顺序校验：两个文件存在 → manifest SHA → manifest JSON 对象 → `data_file` → NetCDF SHA 与发布定义一致 → manifest 的 `data_sha256`/`data_bytes` 与实物一致 → `load_earth_dataset()` 的数据协议 → manifest 与 NetCDF 逐项元数据一致。

关键文件校验代码应采用以下逻辑：

```python
from pathlib import Path
import hashlib
import json

def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def read_verified_manifest(package_dir, expected_manifest_sha256, expected_data_sha256):
    root = Path(package_dir)
    manifest_path = root / "manifest.json"
    data_path = root / "earth_merra2_daily.nc"
    if not manifest_path.is_file() or not data_path.is_file():
        raise FileNotFoundError("package_missing")
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256:
        raise ValueError("manifest_fingerprint_mismatch")
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict) or manifest.get("data_file") != data_path.name:
        raise ValueError("invalid_manifest")
    if file_sha256(data_path) != expected_data_sha256:
        raise ValueError("data_fingerprint_mismatch")
    if manifest.get("data_sha256") != expected_data_sha256:
        raise ValueError("manifest_data_fingerprint_mismatch")
    if manifest.get("data_bytes") != data_path.stat().st_size:
        raise ValueError("manifest_data_size_mismatch")
    return manifest, data_path
```

- [ ] 使用 `with load_earth_dataset(data_path) as ds:` 计算元数据。校验项目必须包括以下字段，而不只是文件能打开：

```python
import numpy as np
from services.earth_dataset import CHANNELS, SCHEMA, SPLITS, UNITS

def assert_earth_manifest_matches(ds, manifest):
    dates = ds.time.values.astype("datetime64[D]")
    lat, lon = ds.lat.values, ds.lon.values
    checks = [
        manifest.get("schema") == SCHEMA,
        manifest.get("planet") == "Earth",
        manifest.get("dimensions") == dict(ds.sizes),
        manifest.get("time_start") == str(dates[0]),
        manifest.get("time_end") == str(dates[-1]),
        manifest.get("channel_order") == list(CHANNELS),
        manifest.get("latitude_range") == [float(lat.min()), float(lat.max())],
        manifest.get("longitude_range") == [float(lon.min()), float(lon.max())],
        manifest.get("source_sha256") == ds.attrs.get("source_sha256"),
        manifest.get("cadence") == "daily mean",
        ds.attrs.get("temporal_resolution") == "1 day",
    ]
    for name, unit in zip(CHANNELS, UNITS):
        variable = manifest.get("variables", {}).get(name, {})
        checks.extend([
            variable.get("units") == unit == ds[name].attrs.get("units"),
            variable.get("dtype") == "float32" == str(ds[name].dtype),
            variable.get("min") == float(ds[name].min()),
            variable.get("max") == float(ds[name].max()),
        ])
    for name, code in SPLITS.items():
        selected = dates[ds["split"].values == code]
        checks.append(manifest.get("splits", {}).get(name) == {
            "start": str(selected[0]), "end": str(selected[-1]), "days": len(selected),
        })
    if not all(checks):
        raise ValueError("manifest_metadata_mismatch")
    if not (np.allclose(np.diff(lat), np.diff(lat)[0]) and np.allclose(np.diff(lon), np.diff(lon)[0])):
        raise ValueError("nonuniform_grid")
```

固定发布定义已经通过 expected SHA 固定本地 731 天小包；元数据仍由文件计算，不通过写死 731 或 31×49 来伪造验证成功。测试需另外构造“指纹匹配但内部 manifest 与文件不一致”的 fixture，才能覆盖上述语义校验。

- [ ] 提取 grid 的两条坐标数组、shape、range、step，以及 time 和 splits；Python/Numpy 值转为 JSON 标量，日期统一 `YYYY-MM-DD`。calendar 从解码时间编码读取，确认是 Gregorian 类型；本包使用 `proleptic_gregorian`。非支持 calendar 返回 invalid，不能标成 Mars 时间。
- [ ] 错误码映射固定为 `package_missing`、`manifest_fingerprint_mismatch`、`data_fingerprint_mismatch`、`invalid_manifest`、`manifest_metadata_mismatch`、`invalid_dataset`、`package_unreadable`、`package_changed_during_verification`。manifest 自报的数据 SHA/字节数错误合并为 `manifest_metadata_mismatch`；日期/网格/单位/非均匀坐标等底层数据协议错误合并为 `invalid_dataset`。下层详细原因仅写日志；对外不回传绝对路径。
- [ ] 运行 `python -m pytest tests/test_dataset_registry.py tests/test_earth_dataset.py -q`；实际执行时使用第 7 节的新临时目录参数。

### 任务 3：注册服务、Schema 和只读 API

**文件：**新增 `services/dataset_registry.py`、`schemas/datasets.py`、`routers/datasets.py`、`tests/test_dataset_routes.py`；修改 `config.py`、`.env.example`、`main.py`。

- [ ] `DatasetRegistry` 提供以下确定接口：

```text
DatasetRegistry(earth_package_dir, *, expected_manifest_sha256=EXPECTED_MANIFEST_SHA256, expected_data_sha256=EXPECTED_DATA_SHA256)
list_datasets() -> list[dict]
get_dataset(dataset_id: str) -> dict
build_training_binding(dataset_id: str) -> dict  # 返回任务五列值
```

生产默认 SHA 使用第 2 节基准，测试构造函数可覆盖。Earth 描述符使用 `read_earth_metadata()`，再补充 ID、版本、capabilities 和 fingerprint。Mars 描述符按 3.2 返回未验证状态。

```python
EXPECTED_MANIFEST_SHA256 = "1d346a059aff57964cb726dde7113fc495adb4b8245f533911dc6f3d63e52d7e"
EXPECTED_DATA_SHA256 = "c21c4ddcb03a69d2eb504f746c86b3a15cf33a06929044f50da768bdfc45cf74"
```

- [ ] 实现以下 Mars 绑定规则，Earth 调用必须先经 `require_training_dataset()` 拒绝，不能从可见目录绕过训练能力检查。

```python
def build_training_binding(self, dataset_id):
    dataset_id = require_training_dataset(dataset_id)
    return {
        "dataset_id": dataset_id,
        "dataset_version": None,
        "dataset_fingerprint": None,
        "dataset_identity_status": "unversioned",
        "dataset_snapshot": json.dumps({
            "dataset_id": dataset_id,
            "planet": "mars",
            "dataset_version": None,
            "binding_basis": "server_registry",
            "version_status": "unversioned",
        }, sort_keys=True),
    }
```

- [ ] 使用锁保护 Earth 一次校验和元数据缓存，避免并发访问 NetCDF 引起线程问题。缓存只保存描述符，不保存整个数据立方体。缓存 key 为两个文件的 `(resolved_path, size, mtime_ns, ctime_ns)`；文件消失或任一签名变化立即重新校验。每次返回深拷贝，避免调用方篡改共享字典。文件校验前后签名不同返回 `invalid/package_changed_during_verification`，下一次请求重新检查，不缓存不一致的结果。
- [ ] 缺包/坏包是描述符状态；未知 ID 抛 `DatasetRequestError(code="unknown_dataset", status_code=404)`。不要用宽泛 `except Exception` 将编程错误永久隐藏成“无数据”。
- [ ] 按 3.2 为嵌套字段创建 Pydantic 模型；禁用 `NaN/Infinity` 的对外序列化，不用无约束 `dict` 替代整个响应协议。
- [ ] `config.py` 增加：

```python
EARTH_MERRA2_DIR = Path(os.getenv(
    "ASTRAATMOS_EARTH_MERRA2_DIR",
    DATA_DIR / "earth" / "merra2_daily_v1",
)).expanduser()
```

`.env.example` 增加可选注释 `# ASTRAATMOS_EARTH_MERRA2_DIR=data/earth/merra2_daily_v1`。相对路径明确相对后端启动目录；路径本身不返回 API。

- [ ] `main.py` lifespan 装配 `app.state.dataset_registry = DatasetRegistry(EARTH_MERRA2_DIR)`，构造不读取文件；路由挂载使用既有 `API_PREFIX`。不调用 Mars DataService 或模型加载器来获取目录。
- [ ] Router 使用同步函数，让 FastAPI 在线程池执行磁盘/NetCDF 工作；从 Request 取同一注册服务：

```python
from fastapi import APIRouter, HTTPException, Request
from schemas.datasets import DatasetDescriptor, DatasetListResponse
from services.dataset_identity import DatasetRequestError

router = APIRouter(prefix="/datasets", tags=["Datasets"])

@router.get("", response_model=DatasetListResponse)
def list_datasets(request: Request):
    return {"items": request.app.state.dataset_registry.list_datasets()}

@router.get("/{dataset_id}", response_model=DatasetDescriptor)
def get_dataset(dataset_id: str, request: Request):
    try:
        return request.app.state.dataset_registry.get_dataset(dataset_id)
    except DatasetRequestError as exc:
        raise HTTPException(status_code=exc.status_code, detail={
            "code": exc.code, "message": str(exc),
        }) from exc
```

- [ ] 用单独 FastAPI 测试 app 挂载新 router，注入临时 registry，避免导入 `main` 启动真实数据库与 Mars 资源。断言列表恰好三个 ID；Earth 缺失/损坏仍有条目；详情 404；capabilities 全部准确；结果没有包目录绝对路径；OpenAPI 包含两个路由和响应模型。

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient
from routers.datasets import router
from services.dataset_registry import DatasetRegistry

def test_earth_directory_contract(earth_release):
    app = FastAPI()
    app.state.dataset_registry = DatasetRegistry(**earth_release)
    app.include_router(router, prefix="/api")
    with TestClient(app) as client:
        response = client.get("/api/datasets")
        assert response.status_code == 200
        assert [item["dataset_id"] for item in response.json()["items"]] == [
            "openmars_mcd", "mcd_overview", "earth_merra2_daily_v1",
        ]
        result = client.get("/api/datasets/earth_merra2_daily_v1").json()
        assert result["availability"] == "available"
        assert result["grid"]["shape"] == [31, 49]
        assert len(result["grid"]["latitude_values"]) == 31
        assert len(result["grid"]["longitude_values"]) == 49
        assert result["time"]["count"] == 731
        assert result["variables"][0]["units"] == "DU"
        assert result["capabilities"]["training"] is False
        assert client.get("/api/datasets/missing").status_code == 404
```
- [ ] 运行 `python -m pytest tests/test_dataset_registry.py tests/test_dataset_routes.py -q`。

### 任务 4：任务身份列和旧 SQLite 幂等迁移

**文件：**修改 `database/models.py`、`database/init_db.py`；新增 `database/dataset_identity_migration.py`、`tests/test_training_dataset_identity_migration.py`。

- [ ] 先用原生 SQL 在临时 SQLite 建立缺少新字段的旧 `model_training_tasks` 表，插入 3.3 矩阵中的旧行。不能只用新 ORM create_all 来声称已测试旧库迁移。
- [ ] 给 ORM 增加 3.3 的五列，均可空且不设置 `openmars_mcd` 数据库默认值；否则无法区分未迁移的未知来源。
- [ ] 新模块提供 `DatasetIdentityMigrationError(RuntimeError)` 和 `async migrate_dataset_identity(conn) -> int`。由调用者的 `engine.begin()` 负责事务。使用如下补列及参数化回填逻辑：

```python
from sqlalchemy import text
from services.dataset_identity import infer_legacy_identity

DATASET_COLUMNS = {
    "dataset_id": "VARCHAR(80)",
    "dataset_version": "VARCHAR(40)",
    "dataset_fingerprint": "VARCHAR(64)",
    "dataset_identity_status": "VARCHAR(32)",
    "dataset_snapshot": "TEXT",
}

class DatasetIdentityMigrationError(RuntimeError):
    pass

async def migrate_dataset_identity(conn):
    try:
        result = await conn.execute(text("PRAGMA table_info(model_training_tasks)"))
        existing = {row[1] for row in result.fetchall()}
        if not existing:
            raise RuntimeError("model_training_tasks table is missing")
        for name, sql_type in DATASET_COLUMNS.items():
            if name not in existing:
                await conn.execute(text(f"ALTER TABLE model_training_tasks ADD COLUMN {name} {sql_type}"))
        rows = (await conn.execute(text("""
            SELECT id, hyperparameters FROM model_training_tasks
            WHERE dataset_id IS NULL AND dataset_version IS NULL
              AND dataset_fingerprint IS NULL AND dataset_identity_status IS NULL
              AND dataset_snapshot IS NULL
        """))).mappings().all()
        for row in rows:
            values = infer_legacy_identity(row["hyperparameters"])
            await conn.execute(text("""
                UPDATE model_training_tasks
                SET dataset_id=:dataset_id, dataset_version=:dataset_version,
                    dataset_fingerprint=:dataset_fingerprint,
                    dataset_identity_status=:dataset_identity_status,
                    dataset_snapshot=:dataset_snapshot
                WHERE id=:task_id
                  AND dataset_id IS NULL AND dataset_version IS NULL
                  AND dataset_fingerprint IS NULL AND dataset_identity_status IS NULL
                  AND dataset_snapshot IS NULL
            """), {**values, "task_id": row["id"]})
        return len(rows)
    except Exception as exc:
        raise DatasetIdentityMigrationError("Dataset identity migration failed") from exc
```

SQL 插值仅用于模块固定的列名和类型，行值全部绑定参数。此迁移无文件搬运、删除或权重读取。

- [ ] 在 `init_database()` 的建表事务中，现有训练表补列之后调用新迁移，且放在该旧补列的异常捕获之外。外层总异常处理前增加：

```python
except DatasetIdentityMigrationError:
    logger.exception("Required dataset identity migration failed")
    raise
```

新身份迁移失败必须中止启动，避免返回看似健康但任务查询全部失败的应用；已有其他初始化异常策略不在本次重写。

- [ ] 测试第一次迁移正确、第二次更新数为 0、已绑定任务不变、已存在部分新列也能完成、损坏 JSON 不导致全部回填失败。注入中途 UPDATE 失败，验证事务回滚后旧行原字段不变，重试可完成；SQLite 已成功增加的列允许留下，补列必须幂等。
- [ ] 逐字比较迁移前后的 `hyperparameters`、模型来源、状态、路径和指标；临时权重文件字节不变；已有标签关联不变。迁移测试只调用 helper，不能调用包含权重搬运逻辑的完整 `init_database()`。
- [ ] 运行 `python -m pytest tests/test_dataset_identity.py tests/test_training_dataset_identity_migration.py -q`。

### 任务 5：接入训练请求、任务响应和严格入口校验

**文件：**修改训练 schema/router/service、`training_channels.py` 和两个 CLI 入口；新增 `tests/test_training_dataset_identity.py`。

- [ ] `TrainingStartRequest` 增加可选顶层 `dataset_id`。使用 `model_validator(mode="before")` 拒绝顶层 `SERVER_IDENTITY_FIELDS`；不全局改成 `extra="forbid"`，以免改变其他已有额外字段兼容行为。

```python
@model_validator(mode="before")
@classmethod
def reject_client_dataset_identity(cls, value):
    if isinstance(value, dict) and SERVER_IDENTITY_FIELDS.intersection(value):
        raise ValueError("Dataset identity is generated by the server")
    return value
```

- [ ] `TrainingTaskResponse` 增加可空 `dataset_id`、`dataset_version`、`dataset_fingerprint`、`dataset_identity_status` 和 `dataset_snapshot: dict[str, Any] | None = None`，不能覆盖任务原有 `status`。快照用 field validator 将数据库 JSON TEXT 转对象；无快照返回 null。非法快照返回 null，不应导致整个历史列表崩溃。

```python
@field_validator("dataset_snapshot", mode="before")
@classmethod
def decode_dataset_snapshot(cls, value):
    if value is None or isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None
```

- [ ] `TrainingService.start_training()` 追加可选参数 `dataset_id=None, dataset_registry=None`，不改变既有位置参数。通过传入 registry 或使用配置构造轻量默认 registry。在首个 `async_session_maker()` 之前执行：

```python
resolved_dataset_id = resolve_dataset_id(dataset_id, hyperparameters or {})
require_training_dataset(resolved_dataset_id)
binding = registry.build_training_binding(resolved_dataset_id)
hyperparameters = dict(hyperparameters or {})
hyperparameters["training_dataset"] = resolved_dataset_id
```

`registry` 为显式传入的 `dataset_registry`，没有时用 `DatasetRegistry(EARTH_MERRA2_DIR)`。建立 `ModelTrainingTask(...)` 时加入 `**binding`，其余字段沿用现有代码。不能把 `binding` merge 到 `payload_hypers`。

- [ ] `routers/training.py` 将 `dataset_id=req.dataset_id` 和 `dataset_registry=getattr(request.app.state, "dataset_registry", None)` 传给现有模块级 service。更新使用简易 Req 对象的测试 fixture，补上 `dataset_id=None`。
- [ ] `start_training` route 在现有 `except ValueError` 之前新增：

```python
except DatasetRequestError as exc:
    raise HTTPException(status_code=exc.status_code, detail={
        "code": exc.code, "message": str(exc),
    }) from exc
```

保持名称重复等已有 `ValueError -> 409` 行为，不能将所有错误一次性改为 400。

- [ ] `training_channels._training_dataset()` 改为调用严格解析和能力校验；空字符串的新请求不再静默回退：

```python
def _training_dataset(value):
    dataset_id = resolve_dataset_id(None, {"training_dataset": value})
    return require_training_dataset(dataset_id)
```

- [ ] `demo3.py` 和 `user_model_runner.py` 的 `main()` 中在 argparse 解析后、数据加载前使用：

```python
args.training_dataset = require_training_dataset(
    resolve_dataset_id(None, {"training_dataset": args.training_dataset})
)
```

两个 runner 原有底层 normalizer 也用于旧任务推理读取，保留其历史兼容语义并添加注释：新训练必须走严格入口，兼容函数仅服务旧读取。不要因全局替换而无意破坏旧模型读取；用 CLI 测试证明 Earth/未知 ID 到不了数据加载阶段。

- [ ] 更新 `test_normalize_training_hyperparameters_sanitizes_training_dataset`：默认仍是 `openmars_mcd`、有效 `mcd_overview` 保留、未知值和 Earth 分别抛错误；保持其他模型参数断言。
- [ ] 任务身份通过独立列保存，`build_hyperparameter_args()` 不应出现 `--dataset_id`、`--dataset_version`、`--dataset_snapshot`、`--dataset_fingerprint`；原来的 `--training_dataset` 保留。
- [ ] 测试必须覆盖 official 和 uploaded 两条路径、旧式请求、新式请求、字段冲突、Earth 拒绝、未知 ID 拒绝、伪造身份拒绝。拒绝时断言数据库会话/上传包加载/子进程调度均未调用。成功测试使用临时 SQLite，并将 `_run_training_subprocess` 替换成受控异步测试函数，不启动真实训练。

核心拒绝测试：

```python
import asyncio
import pytest
from services import training_service as training_module
from services.dataset_identity import DatasetRequestError

def test_earth_training_rejected_before_database(monkeypatch):
    def forbidden_database():
        raise AssertionError("Database must not be touched")
    monkeypatch.setattr(training_module, "async_session_maker", forbidden_database)
    async def run():
        with pytest.raises(DatasetRequestError) as exc:
            await training_module.TrainingService().start_training(
                user_id=7, model_script="demo3.py", hyperparameters={},
                custom_model_name="earth-blocked", dataset_id="earth_merra2_daily_v1",
            )
        assert exc.value.code == "dataset_training_not_supported"
    asyncio.run(run())
```

- [ ] 运行新身份测试和现有 training channel、uploaded contract、personal source、tags 测试；依赖或历史污染问题要报告并定点处理，不用删除断言换取通过。

### 任务 6：文档同步

**文件：**新增 `docs/dataset-registry.md`；修改 `README.md`、`docs/earth-compact-dataset.md`。

- [ ] 专题文档写明两个接口、三个 ID、全部状态、错误码、版本/指纹区别、配置、迁移规则和请求兼容示例。
- [ ] README 增加后端注册入口和配置项，功能边界改成：“已注册地球数据元信息，独立读取可用；网页总览、训练任务和预测入口尚未支持 Earth，训练请求会明确拒绝”。保留上传不等于可训练的原有边界。
- [ ] Earth 小包文档将“尚未注册”更新为实际完成状态，保留二维/三维、日期预测和训练未接通说明。原始运行数据里的 manifest 保持原样，其中历史 limitations 字符串不作为最新能力开关；最新能力由注册服务给出。
- [ ] 命令示例标明后端执行目录、使用相对路径，不写本机个人解释器绝对路径或分支状态。文档最近核对日期和验证范围按实际执行更新，不把本方案里的预期检查写成通过。

### 任务 7：整体验收与交付

- [ ] 执行第 7 节的针对性回归。
- [ ] 用真实本地小包做只读 registry 验证；缺数据机器允许标记该项未执行，但临时 fixture/API/迁移测试必须通过。
- [ ] 阅读 Git diff，检查无实际数据包、数据库、权重、日志、凭证或临时文件被纳入变更；检查旧 Earth 未提交文件没有被覆盖。
- [ ] 交付逐项说明新增接口、旧任务处理、测试结果、尚未开放功能及相关文档路径。用户要求的是实现本阶段，不自动提交、推送或开 PR；若另有明确提交授权，使用显式文件列表，不混入无关修改。

## 6. 必测矩阵

| 编号 | 场景 | 必须得到的结果 |
| --- | --- | --- |
| R1 | 正确本地 Earth 小包 | 731 天、31×49、4°/5°、DU、区域范围和真实坐标 |
| R2 | 小包目录不存在 | 列表/详情 200，Earth missing，其余条目正常 |
| R3 | 文件内容改动 | invalid，旧缓存结果不继续返回 available |
| R4 | 同时修改数据及 manifest 自报 hash | 仍不匹配固定发布 SHA，不接受为 v1 |
| R5 | 测试发布定义信任某个语义错误 manifest | 内部日期/单位/网格/划分不一致仍拒绝 |
| R6 | 多次/并发查询 | 结果一致；未变文件不重复完整加载；返回对象不能修改缓存 |
| R7 | 验证过程中发生文件变化 | 不缓存不一致结果，返回明确 invalid |
| A1 | 未知查询 ID | 404 unknown_dataset |
| A2 | 缺失/损坏 Earth | 不泄露目录，不影响整个目录服务 |
| A3 | Earth available | metadata=true，training/web_overview/trained_prediction=false |
| M1 | 真旧表、两个 Mars 来源、字段缺省 | 对应归属，版本/指纹 null |
| M2 | JSON 损坏/数组/未知来源 | legacy_unknown，原 JSON 原样保存 |
| M3 | 重复迁移和部分补列 | 不重复写入，不覆盖已有绑定 |
| M4 | 中途数据库异常 | 不继续启动；可重试；不移动/删除权重 |
| T1 | 旧前端只传 training_dataset | 正确创建 Mars 任务及 unversioned 快照 |
| T2 | 新顶层 ID | 与原有 runner 的 training_dataset 一致 |
| T3 | 顶层/旧字段冲突 | 400，零任务、零调度 |
| T4 | 未知/空 ID | 拒绝，不回退 Mars |
| T5 | Earth 的 official/uploaded 请求 | 409，数据库、模型包和子进程均未触及 |
| T6 | 用户伪造身份 | 拒绝，不能污染 task 或 runner 参数 |
| T7 | 身份入库和列表序列化 | 五个字段一致，snapshot 返回对象，旧客户端兼容 |
| T8 | CLI 传未知 ID 或 Earth | 加载数据前失败；不能开始 Mars 训练 |
| C1 | 标签、重命名、权限、模型来源 | 既有行为和测试保持 |
| C2 | 旧任务 JSON/预测缓存身份 | 本次迁移不重写超参数或缓存载荷 |

## 7. 验证命令与成功标准

所有 Python 命令在 `AstraAtmos_backend/backend/` 执行。选择能导入当前后端依赖的解释器；可参考父工作区启动脚本中的现有 Conda 环境，不安装或切换全局环境。下面用 `$datasetPython` 保存已核实的解释器路径，不改系统环境变量。

```powershell
Set-Location AstraAtmos_backend\backend
$datasetPython = (Get-Command python).Source
& $datasetPython -B -c 'import sys, pytest, fastapi, pydantic, sqlalchemy, aiosqlite, numpy, xarray, netCDF4; print(sys.executable)'
```

若当前 `python` 缺依赖，读取工作区启动脚本找到已经存在的项目解释器后，将它赋给 `$datasetPython` 再运行。不要把缺依赖当成业务测试失败。

每次 pytest 使用新的纯英文临时路径，不复用旧 basetemp，不添加清理操作：

```powershell
$datasetTestTemp = Join-Path $env:TEMP ('.dataset-registry-test-' + [guid]::NewGuid().ToString('N'))
$datasetTests = @(
  'tests/test_dataset_identity.py',
  'tests/test_dataset_registry.py',
  'tests/test_dataset_routes.py',
  'tests/test_training_dataset_identity_migration.py',
  'tests/test_training_dataset_identity.py',
  'tests/test_earth_dataset.py',
  'tests/test_training_channel_contract.py',
  'tests/test_training_channels.py',
  'tests/test_training_dataset_loader.py',
  'tests/test_official_training_runner.py',
  'tests/test_uploaded_training_contract.py',
  'tests/test_user_model_schema.py',
  'tests/test_training_tags.py',
  'tests/test_personal_source_disabled_contract.py',
  'tests/test_prediction_analysis_cache_identity.py'
)
& $datasetPython -m pytest @datasetTests -q --basetemp "$datasetTestTemp"
```

预期：全部选定测试通过。若老测试使用异步插件但环境未配置，按既有项目测试方式执行对应脚本或补充正确测试运行配置；不能把 coroutine 未执行的跳过当作通过。测试仅使用临时数据库，不连接用户正在使用的 SQLite。先检查被调用测试的清理逻辑，避免运行会批量清理文件的测试路径；相关测试改用专属临时目录且不主动批量清理。

真实数据只读验证（使用已经实现的 registry，独立于完整 app lifespan）：

```powershell
@'
from config import EARTH_MERRA2_DIR
from services.dataset_registry import DatasetRegistry
registry = DatasetRegistry(EARTH_MERRA2_DIR)
result = registry.get_dataset("earth_merra2_daily_v1")
assert result["availability"] == "available", result["availability_reason"]
assert result["time"]["count"] == 731
assert result["time"]["start"] == "2020-01-01"
assert result["time"]["end"] == "2021-12-31"
assert result["grid"]["shape"] == [31, 49]
assert result["grid"]["latitude_step"] == 4.0
assert result["grid"]["longitude_step"] == 5.0
assert result["grid"]["wrap_longitude"] is False
assert result["variables"][0]["units"] == "DU"
assert len(result["dataset_fingerprint"]) == 64
assert result["capabilities"]["training"] is False
print("Earth registry metadata verified")
'@ | & $datasetPython -B -
```

验收需要 API 请求/响应测试和数据库迁移测试，不能只靠这个元数据脚本。无需为了第一步运行真实训练或启用网页 Earth 选项。

从项目根目录检查：

```powershell
Set-Location AstraAtmos
git diff --check
git diff --stat
git status --short
```

前端没有改动时不要求构建前端。若执行中确实改了前端，应说明原因并运行对应 Node 测试及构建，不能悄悄扩展到第二阶段 UI 开发。

## 8. 完成定义和下一阶段边界

- [ ] 新接口可独立测试，三个数据集身份稳定。
- [ ] 本地数据存在时 Earth 描述准确；缺失/错误时清楚表示不可用。
- [ ] Earth 包被改动时不再被接受为固定 v1。
- [ ] 旧任务有明确来源/未知状态，旧 JSON、模型、标签和缓存载荷不被重写。
- [ ] 新 Mars 任务保存独立身份；新请求不会静默落入错误数据集。
- [ ] Earth 训练仍明确拒绝，官方/自定义模型两条路径均有效。
- [ ] 新增字段不会进入 CLI 参数，不改变模型输入输出和训练循环。
- [ ] 文档与实际实现一致，并报告实际运行的验证。

下一阶段是使用这些元数据实现二维 Earth 总览。Earth 训练阶段再增加完整训练快照（选定通道顺序、输入/输出窗口、归一化数组、坐标和日期划分）、checkpoint 绑定与预测复现校验；不要把本阶段的注册信息快照称为已经足够独立复现训练的模型产物。

## 9. 可复制给执行对话的指令

```text
请实施 docs\plans\2026-09-23-dataset-registry-and-task-identity.md。

先阅读父工作区 AGENTS.md 和项目 README 指定接手章节，检查 Git 状态及近期提交，保留已有未提交 Earth 小包代码与文档。按方案完成“数据集注册与训练任务身份兼容”第一阶段，做到只读目录 API、真实 Earth 元数据验证、旧任务幂等迁移、新请求严格校验，以及测试和 README/专题文档同步。

Earth 只注册元数据，本阶段不开放地球训练、预测或网页选项。旧 Mars 版本未知时如实保留未知，不伪造版本或指纹。不要重写旧超参数、删除文件、改动真实权重、自动提交或推送。使用临时数据库和新的测试目录完成验证，并交付修改范围、实际测试结果和未开放边界。按顺序执行；若执行环境没有名为 executing-plans 的技能，仍以本方案及当前用户指令完成，不把技能名称缺失当成业务阻塞。
```
