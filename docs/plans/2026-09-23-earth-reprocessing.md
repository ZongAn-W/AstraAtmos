# 2026-09-23 Earth MERRA-2 reprocessing record

## Scope

用户最终选择只处理 E 盘 MERRA-2；ERA5 不进入本任务，也不创建只有北极温度的独立包。原始目录和旧 v1 发布包保留，工作树其他未提交修改不覆盖。

## Implemented contract

- 输入：`<数据根目录>\data\MERRA2\raw` 下 SLV 的 `TO3/U10M/V10M/T2M` 与 RAD 的 `SWGDN`。
- 时间：2020-01-01 至 2021-12-31，每日 24 个 00:30…23:30 UTC 小时中心。
- 空间：全球 5°×5°、36×72；中心纬度 −87.5…87.5、经度 −177.5…177.5，单元边界为 ±90°/±180°。
- 算法：逐日完整小时的 float64 均值，再按源/目标球面单元重叠面积加权；经度按周期接缝处理；发布字段 float32。
- 质量：日期解析和配对、连续性、坐标一致、维度顺序、单位、填充值/缺测、有限值、源文件变更和面积守恒均失败即停，不填零、不静默跳日。
- 身份：新包 `earth_merra2_daily_v2` / `v2`；旧 `earth_merra2_daily_v1` / `v1` 保留独立注册和哈希。
- 划分：train 2020（366）、validation 2021 上半年（181）、test 2021 下半年（184）。归一化统计只拟合训练日期。

## 实现入口

- `AstraAtmos_backend/backend/scripts/preprocess_earth_reanalysis.py`：生产预处理器和合成数据测试入口。
- `services/earth_dataset_metadata.py`：manifest、cell bounds、全球覆盖和发布指纹校验。
- `services/dataset_registry.py`：v1/v2 独立注册、默认 v2、只读快照。
- `services/earth_overview_service.py`：v2 球面单元面积均值、极区/接缝点位边界。
- `frontend/src/pages/DataOverviewPage/EarthOverview/earthMapGeometry.js`：按真实 cell bounds 画图和吸附点位。

## 产物与验证

v2 产物位于 `AstraAtmos_backend/backend/data/earth/merra2_daily_v2/`，包含 NetCDF、manifest 和 `verification/preprocess_report.json`。manifest 固定数据 SHA-256，报告记录 731 天×5 变量的源值计数和面积守恒误差。

已执行：

1. 16 项预处理测试：常数保持、球面面积、周期接缝、24 小时中心、单位/缺测/填充值拒绝、日期配对、训练统计隔离、v1/v2 身份隔离、边界和归一化摘要篡改拒绝。
2. 真实数据全量处理 731 天；每个源变量均检查 24×361×576，输出无非有限值。
3. 真实 v2 registry 元数据校验：731 天、36×72、global、`wrap_longitude=true`、±90°/±180° cell bounds。
4. 独立日期/变量/极区/接缝抽查，以及 DLinear CPU 前向、反向和一次优化器更新；输入 `[2,7,5,36,72]`，目标 `[2,3,1,36,72]`。

数据就绪不等于 Earth 训练页或预测页已开放；当前训练入口仍明确返回 409。

## 2026-09-24 最终复核

发布文件的实际 SHA-256 与注册表固定值一致：

| 发布 | manifest.json | earth_merra2_daily.nc |
| --- | --- | --- |
| v2 | `935ca37bd3064772a370db6873b605e12b85c031281c2b34d8fbc49ab11f0702` | `d280a17cb291e1568ccedd1638cb2fadcc5cb8d89bb8ed0d4d51987e8e181396` |
| v1（保留） | `1d346a059aff57964cb726dde7113fc495adb4b8245f533911dc6f3d63e52d7e` | `c21c4ddcb03a69d2eb504f746c86b3a15cf33a06929044f50da768bdfc45cf74` |

v2 NetCDF 为 29,207,992 字节（约 27.86 MiB）；v1 仍为 17,214,767 字节。v2 指纹为 `8871f7ef865c54e166060feb96efa47ef287b4b2c8764206335a15ebc1c8ea01`。

- 全量质量报告覆盖 3,655 个日期×变量场、18,240,145,920 个源值，缺测数为 0。float64 面积守恒最大绝对误差为 `1.7053e-13`（各变量原单位）；最终字段存储为 float32，存储误差另记于报告。
- 独立原始数据验证报告包含 225 个单元；DLinear CPU 前向、反向和一次优化器更新已重新执行通过。
- 运行中的 API：v1/v2 都为 available 且身份各自独立；五变量的 15 个日期场和 20 条极区/接缝点位序列与 v2 NetCDF 逐值一致，5 条全年区域序列与独立球面面积加权计算一致。两个版本的 Earth 训练能力仍为 false，身份校验均拒绝训练并返回 409。
- 本轮测试：预处理 16 项、后端相关回归 254 项、前端全量 347 项通过；后端有 4 个依赖弃用警告。前端构建成功，仍提示既有大 bundle。更早的后端扩展回归记录为 278 项；不与本轮不同选择范围混加。
- 浏览器已核对默认 v2、36×72、5°、全球边界、731 天及五变量物理单位；全套历史 43 项浏览器交互结果属于 v1，不将其作为 v2 全量验收。
- 收尾浏览器检查发现并修正了加载提示残留：当前场请求结算后立即清除 loading，过期响应不更改状态。该局部修复不改变 API、数据或功能边界。

本轮命令从后端目录执行，Python 使用 工作区约定的 Conda 环境；每组测试创建新的英文临时目录：

```powershell
$preprocessTemp = Join-Path (Resolve-Path ..\..\..) ('.earth-preprocess-' + [guid]::NewGuid().ToString('N'))
& "$env:CONDA_PREFIX\python.exe" -m pytest tests/test_preprocess_earth_reanalysis.py -q --basetemp "$preprocessTemp"
$regressionTemp = Join-Path (Resolve-Path ..\..\..) ('.earth-regression-' + [guid]::NewGuid().ToString('N'))
& "$env:CONDA_PREFIX\python.exe" -m pytest tests/test_dataset_identity.py tests/test_dataset_registry.py tests/test_dataset_routes.py tests/test_earth_dataset.py tests/test_earth_overview_service.py tests/test_earth_overview_routes.py tests/test_training_dataset_identity.py tests/test_training_dataset_identity_migration.py -q --basetemp "$regressionTemp"
```

前端目录执行 `node --test --test-reporter=dot` 和 `npm run build -- --emptyOutDir false`；保留既有构建文件。根目录执行 `git diff --check` 并检查文档链接与表格。没有提交、reset、stash 或删除原始数据/旧发布包。
