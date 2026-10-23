param(
    [string]$BackendDir = "",
    [switch]$Repair,
    [string]$BootstrapPython = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($BackendDir)) {
    $BackendDir = Join-Path $PSScriptRoot "AstraAtmos_backend\backend"
}

$BackendDir = (Resolve-Path $BackendDir).Path
$venvDir = Join-Path $BackendDir ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$requirements = Join-Path $BackendDir "requirements.txt"

function Invoke-CommandParts {
    param(
        [string[]]$CommandParts,
        [string[]]$CommandArgs
    )
    $exe = $CommandParts[0]
    $prefix = @()
    if ($CommandParts.Length -gt 1) {
        $prefix = $CommandParts[1..($CommandParts.Length - 1)]
    }
    & $exe @prefix @CommandArgs
}

function Parse-Version {
    param([string]$VersionText)
    if ([string]::IsNullOrWhiteSpace($VersionText)) {
        return $null
    }
    $firstLine = ($VersionText -split "`r?`n")[0].Trim()
    $parts = $firstLine.Split(".")
    if ($parts.Length -lt 2) {
        return $null
    }
    try {
        $patch = 0
        if ($parts.Length -ge 3 -and $parts[2] -match '^\d+$') {
            $patch = [int]$parts[2]
        }
        return [Version]::new([int]$parts[0], [int]$parts[1], $patch)
    }
    catch {
        return $null
    }
}

function Get-PythonVersionFromParts {
    param([string[]]$CommandParts)
    try {
        $out = Invoke-CommandParts -CommandParts $CommandParts -CommandArgs @(
            "-c",
            "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')"
        ) 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) {
            return $null
        }
        $match = [regex]::Match($out, '(?m)^\s*(\d+\.\d+\.\d+)\s*$')
        if (-not $match.Success) {
            return $null
        }
        return Parse-Version -VersionText $match.Groups[1].Value
    }
    catch {
        return $null
    }
}

function Assert-LastExitCode {
    param([string]$StepName)
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE"
    }
}

function Test-VenvHealth {
    if (-not (Test-Path $venvPython)) {
        return $false
    }
    try {
        $null = & $venvPython -c "import sys; import sqlite3, aiosqlite; import fastapi, uvicorn, numpy, torch, psutil, matplotlib, seaborn; import websockets; print(f'{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}')" 2>&1
        return ($LASTEXITCODE -eq 0)
    }
    catch {
        return $false
    }
}

function Find-BootstrapPython {
    $candidates = @()

    if (-not [string]::IsNullOrWhiteSpace($BootstrapPython)) {
        if (Test-Path $BootstrapPython) {
            $candidates += ,@($BootstrapPython)
        } else {
            Write-Host "[WARN] BootstrapPython path not found: $BootstrapPython" -ForegroundColor Yellow
        }
    }

    # 1) Prefer py launcher / PATH python
    $candidates += @(
        @("py", "-3.12"),
        @("py", "-3.11"),
        @("py", "-3.10"),
        @("py", "-3.9"),
        @("python")
    )

    # 2) Common absolute install paths (works even when PATH/py is missing)
    $pathCandidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python310\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python39\python.exe"),
        "C:\Python312\python.exe",
        "D:\Python312\python.exe",
        "C:\Python311\python.exe",
        "C:\Python310\python.exe",
        "C:\Python39\python.exe",
        "D:\Python311\python.exe",
        "D:\Python310\python.exe",
        "D:\Python39\python.exe",
        "C:\ProgramData\Anaconda3\python.exe",
        "D:\Anaconda3\python.exe",
        "C:\Anaconda3\python.exe",
        "D:\Anaconda\python.exe",
        "C:\Anaconda\python.exe",
        "C:\Miniconda3\python.exe",
        "D:\Miniconda3\python.exe",
        "C:\ProgramData\Miniconda3\python.exe"
    )
    foreach ($p in $pathCandidates) {
        if (-not [string]::IsNullOrWhiteSpace($p) -and (Test-Path $p)) {
            $candidates += ,@($p)
        }
    }

    # 3) Conda-specific env var
    if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PYTHON_EXE) -and (Test-Path $env:CONDA_PYTHON_EXE)) {
        $candidates += ,@($env:CONDA_PYTHON_EXE)
    }

    # 4) Discover from where.exe (PATH + app aliases)
    foreach ($cmdName in @("py", "python")) {
        try {
            $whereOut = & where.exe $cmdName 2>$null
            if ($LASTEXITCODE -eq 0 -and $whereOut) {
                foreach ($line in ($whereOut | Out-String).Split("`r", "`n")) {
                    $p = $line.Trim()
                    if (-not [string]::IsNullOrWhiteSpace($p) -and (Test-Path $p)) {
                        $candidates += ,@($p)
                    }
                }
            }
        }
        catch {}
    }

    # 5) Discover from Windows registry PythonCore
    $regRoots = @(
        "HKCU:\Software\Python\PythonCore",
        "HKLM:\Software\Python\PythonCore",
        "HKLM:\Software\WOW6432Node\Python\PythonCore"
    )
    foreach ($root in $regRoots) {
        if (-not (Test-Path $root)) {
            continue
        }
        try {
            $verKeys = Get-ChildItem $root -ErrorAction SilentlyContinue
            foreach ($verKey in $verKeys) {
                $installKey = Join-Path $verKey.PSPath "InstallPath"
                if (-not (Test-Path $installKey)) {
                    continue
                }
                $props = Get-ItemProperty -Path $installKey -ErrorAction SilentlyContinue
                if ($null -eq $props) {
                    continue
                }
                $exeFromProp = $null
                if ($props.PSObject.Properties.Name -contains "ExecutablePath") {
                    $exeFromProp = [string]$props.ExecutablePath
                }
                if (-not [string]::IsNullOrWhiteSpace($exeFromProp) -and (Test-Path $exeFromProp)) {
                    $candidates += ,@($exeFromProp)
                }
                $installDir = $null
                try {
                    $installDir = (Get-Item $installKey).GetValue("")
                }
                catch {}
                if (-not [string]::IsNullOrWhiteSpace($installDir)) {
                    $exe = Join-Path ([string]$installDir) "python.exe"
                    if (Test-Path $exe) {
                        $candidates += ,@($exe)
                    }
                }
            }
        }
        catch {}
    }

    $seen = @{}
    foreach ($parts in $candidates) {
        $key = ($parts -join " ").ToLowerInvariant()
        if ($seen.ContainsKey($key)) {
            continue
        }
        $seen[$key] = $true
        $ver = Get-PythonVersionFromParts -CommandParts $parts
        if ($null -ne $ver -and $ver -ge [Version]::new(3, 9, 0)) {
            return @{
                Parts = $parts
                Version = $ver
                Display = ($parts -join " ")
            }
        }
    }
    return $null
}

if (Test-VenvHealth) {
    Write-Host "[OK] Runtime check passed." -ForegroundColor Green
    exit 0
}

Write-Host "[WARN] Existing runtime is unavailable or incompatible." -ForegroundColor Yellow
if (-not $Repair) {
    Write-Host "[INFO] Run with -Repair to rebuild .venv automatically." -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $requirements)) {
    throw "requirements.txt not found: $requirements"
}

$bootstrap = Find-BootstrapPython
if ($null -eq $bootstrap) {
    throw "No usable Python interpreter found (need Python >= 3.9). Install Python 3.12 or 3.11 x64 (Add to PATH + Install launcher), then rerun start-astraatmos.bat. Download: https://www.python.org/downloads/windows/"
}

Write-Host "[INFO] Rebuilding .venv using $($bootstrap.Display) (Python $($bootstrap.Version))..." -ForegroundColor Cyan
if (Test-Path $venvDir) {
    Remove-Item -Recurse -Force $venvDir
}

Invoke-CommandParts -CommandParts $bootstrap.Parts -CommandArgs @("-m", "venv", $venvDir)
Assert-LastExitCode "python -m venv .venv"

if (-not (Test-Path $venvPython)) {
    throw "Rebuild failed: missing $venvPython"
}

& $venvPython -m pip install --upgrade pip wheel
Assert-LastExitCode "python -m pip install --upgrade pip wheel"

& $venvPython -m pip install -r $requirements
Assert-LastExitCode "python -m pip install -r requirements.txt"

& $venvPython -c "import torch; print(torch.__version__)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] torch missing after requirements install. Trying CPU wheel fallback..." -ForegroundColor Yellow
    & $venvPython -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
    Assert-LastExitCode "python -m pip install torch==2.5.1 (cpu fallback)"
}

& $venvPython -c "import sqlite3, aiosqlite; import fastapi, uvicorn, numpy, torch, psutil, matplotlib, seaborn; import websockets"
Assert-LastExitCode "python import check (runtime packages)"

Write-Host "[OK] Runtime repaired successfully." -ForegroundColor Green
exit 0
