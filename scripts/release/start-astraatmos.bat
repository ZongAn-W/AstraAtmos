@echo off
setlocal EnableExtensions

cd /d "%~dp0"
set "ROOT=%~dp0"
set "BACKEND_DIR=%ROOT%AstraAtmos_backend\backend"
set "PYTHON_EXE=%BACKEND_DIR%\.venv\Scripts\python.exe"
set "REPAIR_PS=%ROOT%repair-runtime.ps1"
set "PORTABLE_BOOTSTRAP=%ROOT%python_runtime\python.exe"
set "BOOTSTRAP_PYTHON=%ASTRAATMOS_BOOTSTRAP_PYTHON%"
set "HEALTH_URL=http://localhost:8000/health"
set "APP_URL=http://localhost:8000"

if not defined BOOTSTRAP_PYTHON (
  if exist "%PORTABLE_BOOTSTRAP%" (
    set "BOOTSTRAP_PYTHON=%PORTABLE_BOOTSTRAP%"
  )
)

if exist "%REPAIR_PS%" (
  echo [INFO] Ensuring Python runtime...
  if defined BOOTSTRAP_PYTHON (
    echo [INFO] Bootstrap Python: %BOOTSTRAP_PYTHON%
  )
  if defined BOOTSTRAP_PYTHON (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%REPAIR_PS%" -BackendDir "%BACKEND_DIR%" -Repair -BootstrapPython "%BOOTSTRAP_PYTHON%"
  ) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%REPAIR_PS%" -BackendDir "%BACKEND_DIR%" -Repair
  )
  if errorlevel 1 (
    echo [ERROR] Runtime repair failed.
    echo [ERROR] Open this file for details: "%REPAIR_PS%"
    echo [HINT] Install Python 3.12 or 3.11 x64 with:
    echo [HINT]   - Add python.exe to PATH
    echo [HINT]   - Install launcher: py
    echo [HINT] Or set:
    echo [HINT]   set ASTRAATMOS_BOOTSTRAP_PYTHON=C:\Path\To\python.exe
    pause
    exit /b 1
  )
)

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Runtime not found: %PYTHON_EXE%
  echo         Try running: powershell -ExecutionPolicy Bypass -File repair-runtime.ps1 -Repair
  pause
  exit /b 1
)

for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":8000 .*LISTENING"') do (
  echo [ERROR] Port 8000 is already in use by PID %%p.
  echo         Close the process first, or run stop-astraatmos.bat.
  pause
  exit /b 1
)

set "ASTRAATMOS_FRONTEND_DIST=%BACKEND_DIR%\frontend_dist"
set "ASTRAATMOS_WARMUP_ON_STARTUP=0"

echo [INFO] Starting AstraAtmos backend...
echo [INFO] A backend console window will open and stay visible for diagnostics.
start "AstraAtmos Backend" /D "%BACKEND_DIR%" cmd /k ""%PYTHON_EXE%" -m uvicorn main:app --host 127.0.0.1 --port 8000"

echo [INFO] Waiting for service startup...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; $ok=$false; for($i=0;$i -lt 180;$i++){try{$r=Invoke-WebRequest -Uri '%HEALTH_URL%' -UseBasicParsing -TimeoutSec 2; if($r.StatusCode -eq 200){$ok=$true; break}}catch{}; Start-Sleep -Milliseconds 500}; if(-not $ok){ exit 1 }"
if errorlevel 1 (
  echo [ERROR] Startup timed out (about 90 seconds).
  echo [ERROR] Please check the "AstraAtmos Backend" console window for the real error message.
  pause
  exit /b 1
)

echo [OK] AstraAtmos is running at %APP_URL%
start "" "%APP_URL%"
echo [TIP] Use stop-astraatmos.bat to stop the service.
pause
