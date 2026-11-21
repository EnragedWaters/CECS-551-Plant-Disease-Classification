@echo off
rem Double-click in Windows File Explorer to start the demo in default WSL.
setlocal
title Plant Disease Demo
pushd "%~dp0"
if errorlevel 1 goto failed
echo Starting the demo in your default WSL distribution...
echo Keep this window open. Press Ctrl+C here to stop the demo.
echo.
rem WSL inherits this directory, so repository paths with spaces need no escaping.
wsl.exe --exec bash scripts/start_demo.sh %*
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
