@echo off
setlocal EnableExtensions

set "FOUND="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /R /C:":8000 .*LISTENING"') do (
  echo [INFO] Stopping PID %%p on port 8000...
  taskkill /PID %%p /F >nul 2>&1
  set "FOUND=1"
)

if not defined FOUND (
  echo [INFO] No running process found on port 8000.
) else (
  echo [OK] Service stopped.
)
pause

