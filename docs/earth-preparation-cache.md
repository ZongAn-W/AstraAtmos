# Earth 三小时训练准备缓存

训练执行阶段的 CUDA 混合精度、批次预取和派生输入复用见 [Earth 训练提速](earth-training-performance.md)。本页缓存的存储 dtype 仍为 float32，身份、完整性校验和归一化定义保持不变。

三小时训练的准备阶段由后端和训练子进程共同使用内容寻址的证明文件。验证证明绑定发布 manifest 的 SHA、NetCDF 数据 SHA、数据 fingerprint、源文件身份和验证器版本；命中前仍读取并核对 manifest、文件大小和身份，`force_full` 可要求重新执行完整 SHA-256 与科学契约扫描。源包改变、证明损坏或版本过期会回退到完整校验。源文件清单中的 `bytes`/`mtime_ns` 元数据可供快速变化检测，但不证明内容完整；内容完整性由分块计算的逐文件 SHA-256 证明。构建或预处理发布前会再次核对全部 digest，因此同大小且恢复 mtime 的原地修改也不能复用旧输出。

归一化只扫描任务 train 的连续 UTC 时间步，使用 `per_channel_standard`、ddof=0 和 float64 合并矩；验证集、测试集和预测不能影响统计量。统计量 JSON 的身份包括数据 fingerprint、源文件身份、train 起止时间/步数、输入通道顺序和方法版本。任务返回值重新带回当前冻结划分，统计量内容本身不把 validation/test 边界混入复用键。

训练缓存的身份包括数据 fingerprint、通道顺序、归一化内容、逻辑/物理 shape、`earth_full_grid_v1` 或 `earth_spatial_tiles_v1` 布局以及缓存版本。模型架构、学习率、batch、epoch、窗口读取方式不影响已经归一化的全时间轴字节；如果实际 train 边界或通道改变，身份改变并重新准备。

写入由 `filelock` 跨进程协调：先在隐藏临时目录生成 `.npy`，刷新并计算 SHA-256，写入带完整性摘要的 metadata，最后通过 `os.replace` 发布确定性目录。读者只接受 `ready`、身份一致、shape/dtype/文件大小正确且 payload SHA 匹配的缓存；发布前后的源包和映射文件身份都要再次核对。中断目录、损坏数组、旧版本和身份不匹配缓存继续保留供诊断，不批量删除。已有 v1 全图/分块缓存首次被使用时会逐块对照源包认证，认证完成后才写入新证明。

Windows 使用设备/文件 ID、大小、mtime 和 NTFS ChangeTime 识别源包；不能用 Windows 的文件创建时间代替变更时间。缓存校验的进程内记忆同时持有共享只读句柄，禁止数组写入及替换，直到该校验结果释放或进程退出；仅释放 mmap 而继续记忆 SHA 会漏检 Windows 延迟更新时间戳的可写 mmap。其他训练进程仍可并发读取。每个新进程首次读取完整 SHA，同一进程的 train/validation/test 三次挂载共享一次结果，读批次前后继续核对身份。checkpoint 的任务完整划分、float32、全图/分块模型调用、评价指标与 strict reload 没有调整。

磁盘条目位于 `verification/*.json`、`normalization/*.json`、`earth_cache_v2/<identity>/normalized.npy` 和同目录 `metadata.json`。物理格式仍为 `astraatmos_earth_training_cache_v1/cache_version=1`，共享身份与完整性证明使用 `cache_identity_version=2`；未带证明的旧当前布局必须认证，未知版本直接拒绝。保留的失败目录不自动清理，运维按项目文件删除规则处理。

2026-10-10 对完整五通道 5848 步发布包只运行准备，不构建模型、不创建训练任务。NetCDF 为 8,623,005,351 字节，float32 缓存为 13,473,792,128 字节（含 NPY header，约 12.55 GiB）。为匹配已有缓存，基准使用 20→20、70/15/15 划分，train 为索引 `[0,4094)`、UTC `2020-01-01T01:30:00Z` 至 `2021-05-26T16:30:00Z`；这不是应用默认划分。

| 测量 | 验证 (s) | train 统计量 (s) | 缓存/认证 (s) | 挂载及一次有界读取 (s) | 总计 (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原实现训练子进程准备 | 143.858 | 72.280 | 77.676 | 0.040 | 293.855 |
| 新目录首次全图准备 | 158.745 | 77.136 | 88.015 | 9.993 | 333.890 |
| 新进程复用全图 | 0.047 | 0.007 | 9.288 | 0.073 | 9.414 |
| 首次分块布局（验证/统计量已有） | 0.029 | 0.006 | 173.100 | 11.018 | 184.153 |
| 新进程复用分块 | 0.048 | 0.005 | 9.234 | 0.051 | 9.339 |
| 已有全图首次认证（验证/统计量已有） | 0.034 | 0.013 | 156.674 | 11.876 | 168.597 |
| 新进程复用已认证的旧全图 | 0.024 | 0.005 | 9.204 | 0.069 | 9.302 |

原实现父后端的首次完整校验另测为 157.988 s，父后端加子进程准备共 451.844 s；若后端已经预热，只计算子进程的 293.855 s。新实现后端与子进程共享完整验证证明。新目录首次准备增加了缓存 payload SHA 和安全挂载检查，没有声称首次生成更快；复用耗时主要是读取完整缓存 SHA。分块首次生成与旧缓存认证并行测量，存在磁盘竞争，不能与串行全图生成直接排名。上述为一次本机实测，系统文件页缓存未清空，不是严格的冷磁盘基准，也不能推断模型吞吐、收敛、GPU 利用率或精度的加速。

已有全图认证用隔离目录内的原数组硬链接及 metadata 副本进行，没有重新写入 12.55 GiB 数组或覆盖原缓存 metadata。失败、中断和测试缓存保留，不执行批量删除。原始 JSON 计时报告保留在工作区的 `earth-preparation-final-20261010/`、`earth-preparation-legacy-final-20261010/` 和 `earth-preparation-baseline-20261010/`，不写入 README 的个人路径示例。

实现入口：[准备文件与 Windows 读保护](../AstraAtmos_backend/backend/services/earth_preparation_cache.py)、[校验证明](../AstraAtmos_backend/backend/services/earth_dataset_metadata.py)、[统计量与全图/分块缓存](../AstraAtmos_backend/backend/services/earth_dataset.py)、[训练调用](../AstraAtmos_backend/backend/models/training_scripts/earth_daily.py)。验证入口为 `tests/test_earth_verification_reuse.py`、`tests/test_earth_preparation_cache.py` 和三小时数据/registry/全图/分块训练回归。本轮隔离回归 318 项通过，补充训练 runner/指标/dtype 回归 26 项通过，使用合成完整网格；真实发布只用于准备计时，不进行真实训练或预测精度验收。

命令从 `AstraAtmos_backend/backend/` 执行，用 Conda `AstraAtmos` 的解释器。`--mode cold` 要配一个新的缓存目录；它强制全量源包校验，但不会清空系统页缓存或删除已有条目。随后在新的 Python 进程执行 `--mode reuse`，并把 `--layout` 改为 `tiles` 可测分块布局：

```powershell
# 激活工作区约定的 Conda 环境（见父目录 AGENTS.md）
$earthPython = Join-Path $env:CONDA_PREFIX 'python.exe'
& $earthPython scripts/benchmark_earth_preparation.py `
  --package-dir '<server-release-directory>' `
  --cache-root '<isolated-cache-directory>' --mode cold --layout full `
  --window 20 --horizon 20 --report '<first-report.json>'
& $earthPython scripts/benchmark_earth_preparation.py `
  --package-dir '<server-release-directory>' `
  --cache-root '<same-isolated-cache-directory>' --mode reuse --layout full `
  --window 20 --horizon 20 --report '<reuse-report.json>'
```

所有报告都包含阶段耗时、源/缓存字节数、fingerprint、缓存路径和实际 task split。`--mode existing` 会尝试认证已有当前布局缓存，写入该隔离缓存的证明文件，因此验收时只对隔离目录使用。
