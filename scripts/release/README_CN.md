# AstraAtmos Windows 一键分发说明

平台统一使用 AstraAtmos 命名；本次为硬切换，分发脚本、包目录、运行配置与环境变量前缀均已更新，现有安装需按新命名重新打包或迁移，旧数据与 checkpoint 不兼容。

目标：把项目打成压缩包，接收方只需双击 `start-astraatmos.bat` 即可启动。

## 推荐打包命令

```powershell
cd scripts/release
powershell -ExecutionPolicy Bypass -File .\build_portable_windows.ps1 -SkipNpmCi -SkipVenv
```

## 关键变化（已瘦身）

- 会打包 `python_runtime`，但现在是**最小启动集**：
  - 保留 Python 核心可执行、`DLLs`、标准库 `Lib`（不含 `site-packages`）
  - 额外补齐 SSL / ctypes 运行所需 DLL
- 不再拷整套 Anaconda 第三方库，显著降低体积。

## 启动机制

- `start-astraatmos.bat` 会优先使用包内 `python_runtime\python.exe`。
- 启动前自动执行 `repair-runtime.ps1 -Repair`：
  - `.venv` 可用则直接启动；
  - `.venv` 不可用则自动重建并安装依赖。

## 常用可选参数

- `-SkipDotEnv`：不打包真实 `.env`（改为模板）
- `-SkipPortablePython`：不打包内置 Python（不推荐）
- `-SkipVenv`：不在打包机预建 `.venv`，由目标机首启自动创建（推荐跨机分发）
