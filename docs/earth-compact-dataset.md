# 地球 MERRA-2 数据包

> 自 2026-10-08 起，日频 v1/v2 数据集已停止提供项目总览、训练及预测。本文的构建、字段及历史验证仅用于追溯旧发布；当前运行请使用[三小时数据集](earth-merra2-3hourly.md)，停用与历史产物处理见[停用约定](earth-dataset-retirement.md)。

## 当前发布

独立三小时产品的构建、验证、注册、官方 DLinear 后端 56→24 训练与历史回测见 [Earth MERRA-2 三小时数据构建、训练与历史回测](earth-merra2-3hourly.md)。它不替换本节日频发布；三小时后端回测已支持 UTC datetime、24 组全球预测/参考/残差与 DU 指标，三小时总览已支持 UTC timestamp、降采样显示、原生点位与后端日聚合；训练/预测前端已支持 56→24 配置、UTC 起点和 24 个 lead 展示；日频日期轴、7→3、上传模型与旧 checkpoint 保留。

项目默认 Earth 数据集是 `earth_merra2_daily_v2`（版本 `v2`），目录为 `data/earth/merra2_daily_v2/`。旧的 `earth_merra2_daily_v1` 保留在原目录并继续按 v1 指纹和 31×49 区域契约提供只读兼容访问；v1 请求不会映射到 v2。

v2 来源是 E 盘 `MERRA2/raw` 的 SLV 与 RAD 产品，从原始文件重新计算，生成后运行时不再读取原始盘。当前仅使用五个变量：`TO3`、`U10M`、`V10M`、`T2M`、`SWGDN`。日频 Earth 总览、官方 DLinear 与兼容上传模型网页训练、历史日期预测均已开放，见 [地球训练与历史预测](earth-training.md)；仍不开放 SPHERE、迁移学习与地球其他官方架构，也不提供无参考真值的未来外推。

## v2 数据契约

| 项目 | v2 约定 |
| --- | --- |
| 时间 | 2020-01-01 至 2021-12-31，连续 731 个 UTC 日 |
| 源时间 | 每日 24 个 00:30…23:30 UTC 小时中心；缺少任何小时直接失败 |
| 空间 | 全球 5°×5°，36×72；中心纬度 −87.5…87.5、经度 −177.5…177.5 |
| 单元边界 | 纬度 −90…90、经度 −180…180；包含极区和经度周期接缝 |
| 空间方法 | 源单元与目标单元的球面重叠面积加权；不是隔点抽样 |
| 数值 | 逐日均值和聚合使用 float64，发布字段存 float32；不填零、不插值 |
| 变量单位 | `TO3` DU；风 `m s-1`；`T2M` K；`SWGDN` W m-2 |
| 划分 | train 2020（366 天）、validation 2021 上半年（181 天）、test 2021 下半年（184 天） |
| 归一化 | `EarthOzoneWindows` 和 manifest 统计量只用 train 日期拟合，验证/测试复用 |

NetCDF 还保存 `lat_bounds`、`lon_bounds`、来源产品、处理规则和 `split`；训练期统计量保存在 manifest。`verification/preprocess_report.json` 记录每个日期/变量的源值数量、源文件清单摘要、面积守恒误差和 float32 存储误差。

## 从原始 MERRA-2 重建

命令从 `AstraAtmos_backend/backend/` 执行。先设置 `$merraRaw` 为实际 MERRA-2 raw 目录；此环境使用已安装依赖的 工作区约定的 Conda 环境：

```powershell
& "$env:CONDA_PREFIX\python.exe" -m scripts.preprocess_earth_reanalysis `
  --source-root "$merraRaw" `
  --output-dir data\earth\merra2_daily_v2
```

脚本会按日期解析 `MERRA2_400`/`MERRA2_401` 文件，严格配对 `slv` 与 `rad`，检查坐标、维度、单位、填充值（包括约 `1e15` 的 `_FillValue`/`missing_value`）、24 小时时间中心和连续日期。输出目录已有 NetCDF 或 manifest 时拒绝覆盖。

生产 v2 发布文件：

```text
data/earth/merra2_daily_v2/
  earth_merra2_daily.nc
  manifest.json
  verification/preprocess_report.json
  verification/validation.json
  verification/preview.png
```

不要用旧的 `build_earth_ozone_dataset.py` 生成 v2；它只负责保留 v1 的 NPZ 打包语义。

## 读取与模型冒烟

```python
from services.earth_dataset import EarthOzoneWindows, load_earth_dataset

path = 'data/earth/merra2_daily_v2/earth_merra2_daily.nc'
with load_earth_dataset(path) as ds:
    print(ds.sizes)                 # time=731, lat=36, lon=72, bounds=2
    print(ds.TO3.sel(time='2020-02-29').attrs['units'])  # DU

dataset = EarthOzoneWindows(path, split='train', window=7, horizon=3)
x, y = dataset[0]
# x: [7, 5, 36, 72]; y: [3, 1, 36, 72]
```

现有检查器可运行 DLinear CPU 前向、反向和一次优化器更新：

```powershell
& "$env:CONDA_PREFIX\python.exe" -m scripts.check_earth_ozone_dataset `
  --dataset data\earth\merra2_daily_v2\earth_merra2_daily.nc `
  --output-dir data\earth\merra2_daily_v2\verification `
  --raw-dir "$merraRaw" `
  --training-smoke
```

`--raw-dir` 独立抽查 5 个日期 × 5 个变量 × 9 个单元，包含两极与经度接缝，并比较源网格和发布网格的面积均值；不调用预处理器的权重函数。本次真实 v2 冒烟使用输入 `[2, 7, 5, 36, 72]`、目标 `[2, 3, 1, 36, 72]`，CPU loss、梯度和权重更新均为有限值。该检查只证明数据与模型接口兼容，不代表训练页已开放或预测精度。

## API 和总览边界

注册表同时保留 v1 与 v2，默认配置 `ASTRAATMOS_EARTH_MERRA2_DIR` 指向 v2，旧包路径可由 `ASTRAATMOS_EARTH_MERRA2_V1_DIR` 指定。v2 描述符、`/field` 和 `/regional-series` 返回 `wrap_longitude=true` 和真实全球单元范围；`/point-series` 允许 ±90°/±180°边界并返回吸附后的最近网格中心。

区域序列的 v2 聚合标识为 `spherical_cell_area_mean`，按纬度单元的 `sin(north)-sin(south)` 面积和等经度单元计算。v1 仍返回 `cos_lat_sample_mean`。所有响应保留物理单位；五变量顺序固定为 `TO3,U10M,V10M,T2M,SWGDN`。

地球页面与训练/预测的当前边界：数据总览提供三维日数据分析观测台、日期播放、五变量切换、全球地图、点位曲线和全球单元面积加权均值；训练开放官方 DLinear（7→3、TO3 + 四个可选辅助输入）；预测开放按历史起点的三天回测。自选区域、重网格、导出、Earth 预测持久化缓存与 Earth/Mars 混合比较仍未开放。

## 验证入口

后端测试从 `AstraAtmos_backend/backend/` 执行，建议为 Windows NetCDF 测试指定新的英文临时目录：

```powershell
$earthTestTemp = Join-Path $env:TEMP ('.earth-test-' + [guid]::NewGuid().ToString('N'))
& "$env:CONDA_PREFIX\python.exe" -m pytest `
  tests/test_preprocess_earth_reanalysis.py `
  tests/test_earth_dataset.py tests/test_dataset_registry.py `
  tests/test_dataset_routes.py -q --basetemp "$earthTestTemp"
```

生产发布的固定 SHA-256 在 `services/dataset_registry.py` 中锁定；修改 v2 数据必须创建新目录、ID/版本和新的固定哈希，不能覆盖 v1 或复用旧身份。

2026-09-24 的实际验证范围、发布 SHA、质量统计和回归命令见 [预处理与替换记录](plans/2026-09-23-earth-reprocessing.md)。
训练窗口严格限制在 manifest 发布 split 内：2020 年 train、2021-01-01 至 2021-06-30 validation、2021-07-01 至 2021-12-31 test。每个 split 独立生成完整窗口；历史预测可以使用 origin 前的完整 7 天历史输入，但不会改变训练 split 或归一化拟合范围。
