@echo off
rem Double-click in Windows File Explorer to start the demo using native Windows Python or WSL.
setlocal
title Plant Disease Demo
pushd "%~dp0"
if errorlevel 1 goto failed
echo Finding a ready Python environment...
echo Keep this window open. Press Ctrl+C here to stop the demo.
echo.
rem Policy bypass applies only to this local launcher process, not system settings.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/start_demo.ps1 %*
if errorlevel 1 goto failed
popd
exit /b 0
:failed
echo.
echo The demo could not start. Read the error above.
echo Setup and troubleshooting: docs\presentation\demo-guide.md
pause
popd
exit /b 1
