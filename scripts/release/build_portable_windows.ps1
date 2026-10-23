param(
    [string]$OutputDir = "release/AstraAtmos-Portable-Windows",
    [switch]$SkipNpmCi,
    [switch]$SkipVenv,
    [switch]$SkipDotEnv,
    [switch]$SkipPortablePython
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..\..")).Path

$frontendDir = Join-Path $repoRoot "frontend"
$backendDir = Join-Path $repoRoot "AstraAtmos_backend\backend"
$outputPath = Join-Path $repoRoot $OutputDir
$outputBackend = Join-Path $outputPath "AstraAtmos_backend\backend"

function Require-Command([string]$CommandName, [string]$Hint) {
    if (-not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        throw "$CommandName not found. $Hint"
    }
}

function Assert-LastExitCode([string]$StepName) {
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE"
    }
}

function Stop-FrontendLockingProcesses {
    $procNames = @("node", "esbuild")
    foreach ($name in $procNames) {
        Get-Process -Name $name -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    }
}

function Ensure-FrontendBuildDependencies {
    $viteCmd = Join-Path $frontendDir "node_modules\.bin\vite.cmd"
    if (Test-Path $viteCmd) {
        return
    }

    Write-Host "Frontend dependencies are missing. Running npm install --no-audit --no-fund..." -ForegroundColor Yellow
    Stop-FrontendLockingProcesses
    Start-Sleep -Seconds 2
    & npm.cmd install --no-audit --no-fund
    Assert-LastExitCode "npm install (bootstrap frontend dependencies)"
}

function Copy-PortablePythonRuntime([string]$RuntimeDir) {
    $infoRaw = & python -c "import json,sys; print(json.dumps({'executable': sys.executable, 'base_prefix': sys.base_prefix, 'version': f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}'}))"
    Assert-LastExitCode "python runtime info probe"

    $infoLine = ($infoRaw | Select-Object -Last 1)
    if ([string]::IsNullOrWhiteSpace($infoLine)) {
        throw "python runtime info probe returned empty output."
    }
    $info = $infoLine | ConvertFrom-Json
    $basePrefix = [string]$info.base_prefix
    if (-not (Test-Path $basePrefix)) {
        throw "python base prefix not found: $basePrefix"
    }

    if (Test-Path $RuntimeDir) {
        Remove-Item -Recurse -Force $RuntimeDir
    }
    New-Item -ItemType Directory -Path $RuntimeDir | Out-Null

    $rootFilePatterns = @(
        "python*.exe",
        "python*.dll",
        "python3.dll",
        "vcruntime*.dll",
        "api-ms-win-*.dll",
        "ucrtbase.dll",
        "sqlite3.dll",
        "libffi-*.dll"
    )
    foreach ($pattern in $rootFilePatterns) {
        Get-ChildItem -Path $basePrefix -Filter $pattern -File -ErrorAction SilentlyContinue | ForEach-Object {
            Copy-Item $_.FullName (Join-Path $RuntimeDir $_.Name) -Force
        }
    }

    $dllDest = Join-Path $RuntimeDir "DLLs"
    New-Item -ItemType Directory -Path $dllDest -Force | Out-Null

    $dllSource = Join-Path $basePrefix "DLLs"
    if (Test-Path $dllSource) {
        Copy-Item -Path (Join-Path $dllSource "*") -Destination $dllDest -Recurse -Force
    }

    # Keep stdlib for venv / ensurepip; exclude third-party site-packages to avoid huge bundle.
    $libSource = Join-Path $basePrefix "Lib"
    if (-not (Test-Path $libSource)) {
        throw "python Lib directory not found: $libSource"
    }
    $libDest = Join-Path $RuntimeDir "Lib"
    New-Item -ItemType Directory -Path $libDest -Force | Out-Null
    $libRoboArgs = @(
        $libSource,
        $libDest,
        "/MIR",
        "/XD", "site-packages", "__pycache__", "test", "tests",
        "/XF", "*.pyc", "*.pyo"
    )
    & robocopy @libRoboArgs | Out-Host
    if ($LASTEXITCODE -gt 7) {
        throw "robocopy python Lib failed with exit code $LASTEXITCODE"
    }

    # Conda Python keeps some runtime DLLs in Library\bin; copy only the minimal set we need.
    $libraryBin = Join-Path $basePrefix "Library\bin"
    if (Test-Path $libraryBin) {
        $runtimeDllPatterns = @(
            "libcrypto-*.dll",
            "libssl-*.dll",
            "ffi*.dll",
            "libffi-*.dll",
            "zlib*.dll",
            "libz*.dll",
            "sqlite3*.dll",
            "libsqlite3*.dll"
        )
        foreach ($pattern in $runtimeDllPatterns) {
            Get-ChildItem -Path $libraryBin -Filter $pattern -File -ErrorAction SilentlyContinue | ForEach-Object {
                Copy-Item $_.FullName (Join-Path $dllDest $_.Name) -Force
            }
        }
    }

    $runtimePython = Join-Path $RuntimeDir "python.exe"
    if (-not (Test-Path $runtimePython)) {
        throw "portable python runtime generation failed: missing $runtimePython"
    }

    & $runtimePython -c "import sys,venv,ensurepip,ssl,ctypes,sqlite3; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')"
    Assert-LastExitCode "portable python runtime sanity check"

    @(
        "source_executable=$($info.executable)",
        "source_base_prefix=$basePrefix",
        "source_version=$($info.version)"
    ) | Set-Content -Path (Join-Path $RuntimeDir "ARES_RUNTIME_INFO.txt") -Encoding UTF8

    $runtimeSize = (Get-ChildItem -Path $RuntimeDir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    Write-Host "Portable Python runtime prepared at $runtimePython (source: $($info.version), size: $([math]::Round($runtimeSize / 1MB, 1)) MB)" -ForegroundColor Green
}

Write-Host "[1/7] Checking prerequisites..." -ForegroundColor Cyan
Require-Command "npm.cmd" "Install Node.js 18+ and ensure npm.cmd is in PATH."
Require-Command "python" "Install Python 3.10+ and ensure python is in PATH."
Require-Command "robocopy" "robocopy is required on Windows."

Write-Host "[2/7] Building frontend dist..." -ForegroundColor Cyan
Push-Location $frontendDir
if (-not $SkipNpmCi) {
    & npm.cmd ci
    if ($LASTEXITCODE -ne 0) {
        Write-Host "npm ci failed (likely Windows file lock). Trying fallback: npm install --no-audit --no-fund" -ForegroundColor Yellow
        Stop-FrontendLockingProcesses
        Start-Sleep -Seconds 2
        & npm.cmd install --no-audit --no-fund
        Assert-LastExitCode "npm install (fallback after npm ci)"
    }
} else {
    Ensure-FrontendBuildDependencies
}
& npm.cmd run build
if ($LASTEXITCODE -ne 0) {
    Write-Host "npm run build failed. Retrying after npm install --no-audit --no-fund..." -ForegroundColor Yellow
    Stop-FrontendLockingProcesses
    Start-Sleep -Seconds 2
    & npm.cmd install --no-audit --no-fund
    Assert-LastExitCode "npm install (fallback after npm run build)"

    & npm.cmd run build
    Assert-LastExitCode "npm run build (retry)"
}
Pop-Location

Write-Host "[3/7] Preparing release directory..." -ForegroundColor Cyan
if (Test-Path $outputPath) {
    Remove-Item -Recurse -Force $outputPath
}
New-Item -ItemType Directory -Path $outputPath | Out-Null

Write-Host "[4/7] Copying backend files..." -ForegroundColor Cyan
$null = New-Item -ItemType Directory -Path (Join-Path $outputPath "AstraAtmos_backend") -Force
$excludeFiles = @("*.pyc", "*.pyo", "astraatmos.db")
if ($SkipDotEnv) {
    $excludeFiles += ".env"
}
$robocopyArgs = @(
    $backendDir,
    $outputBackend,
    "/MIR",
    "/XD", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "logs", "data\perf_cache", "data\user_uploads", "data\pending_review",
    "/XF"
)
$robocopyArgs += $excludeFiles
& robocopy @robocopyArgs | Out-Host
if ($LASTEXITCODE -gt 7) {
    throw "robocopy failed with exit code $LASTEXITCODE"
}

Write-Host "[5/7] Copying frontend dist and launchers..." -ForegroundColor Cyan
$frontendDist = Join-Path $frontendDir "dist"
$portableDist = Join-Path $outputBackend "frontend_dist"
New-Item -ItemType Directory -Path $portableDist -Force | Out-Null
Copy-Item -Path (Join-Path $frontendDist "*") -Destination $portableDist -Recurse -Force

$packagedEnv = Join-Path $outputBackend ".env"
$packagedEnvExample = Join-Path $outputBackend ".env.example"
if ($SkipDotEnv) {
    if (-not (Test-Path $packagedEnv) -and (Test-Path $packagedEnvExample)) {
        Copy-Item $packagedEnvExample $packagedEnv
        Write-Host ".env was excluded by -SkipDotEnv. Generated a non-secret template .env from .env.example." -ForegroundColor Yellow
    } else {
        Write-Host ".env was intentionally excluded by -SkipDotEnv." -ForegroundColor Yellow
    }
} elseif (-not (Test-Path $packagedEnv) -and (Test-Path $packagedEnvExample)) {
    Copy-Item $packagedEnvExample $packagedEnv
    Write-Host "No real .env found in source package. Falling back to .env.example." -ForegroundColor Yellow
} else {
    Write-Host "Using real .env from source package." -ForegroundColor Green
}

Copy-Item (Join-Path $scriptDir "start-astraatmos.bat") (Join-Path $outputPath "start-astraatmos.bat") -Force
Copy-Item (Join-Path $scriptDir "stop-astraatmos.bat") (Join-Path $outputPath "stop-astraatmos.bat") -Force
Copy-Item (Join-Path $scriptDir "repair-runtime.ps1") (Join-Path $outputPath "repair-runtime.ps1") -Force

Write-Host "[6/7] Preparing bundled bootstrap Python runtime..." -ForegroundColor Cyan
if (-not $SkipPortablePython) {
    Copy-PortablePythonRuntime -RuntimeDir (Join-Path $outputPath "python_runtime")
} else {
    Write-Host "Skipped bundled bootstrap Python runtime due to -SkipPortablePython." -ForegroundColor Yellow
}

Write-Host "[7/7] Creating portable Python runtime (.venv)..." -ForegroundColor Cyan
if (-not $SkipVenv) {
    Push-Location $outputBackend
    & python -m venv .venv
    Assert-LastExitCode "python -m venv .venv"

    $pythonExe = Join-Path $outputBackend ".venv\Scripts\python.exe"
    if (-not (Test-Path $pythonExe)) {
        throw "python executable was not found in the generated virtual environment."
    }
    & $pythonExe -m pip install --upgrade pip wheel
    Assert-LastExitCode "python -m pip install --upgrade pip wheel"

    & $pythonExe -m pip install -r requirements.txt
    Assert-LastExitCode "python -m pip install -r requirements.txt"

    & $pythonExe -c "import torch; print(torch.__version__)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "torch is missing after requirements install. Trying CPU wheel fallback..." -ForegroundColor Yellow
        & $pythonExe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
        Assert-LastExitCode "python -m pip install torch==2.5.1 (cpu fallback)"
    }

    & $pythonExe -c "import psutil; print('psutil=' + psutil.__version__)"
    Assert-LastExitCode "python import check (psutil)"

    & $pythonExe -c "import matplotlib, seaborn; print('matplotlib=' + matplotlib.__version__ + ', seaborn=' + seaborn.__version__)"
    Assert-LastExitCode "python import check (matplotlib, seaborn)"
    Pop-Location
} else {
    Write-Host "Skipped .venv creation due to -SkipVenv." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Portable package is ready:" -ForegroundColor Green
Write-Host "  $outputPath"
Write-Host ""
Write-Host "Distribution steps:" -ForegroundColor Green
Write-Host "  1) Zip the folder: $outputPath"
Write-Host "  2) Receiver unzips it"
Write-Host "  3) Receiver double-clicks start-astraatmos.bat"
Write-Host ""
Write-Host "Tip: for best cross-machine compatibility, keep bundled python_runtime and optionally build with -SkipVenv to let first startup create .venv automatically." -ForegroundColor Yellow
Write-Host "Security reminder: package includes backend .env by default. Use -SkipDotEnv if you do not want to share real secrets." -ForegroundColor Yellow
