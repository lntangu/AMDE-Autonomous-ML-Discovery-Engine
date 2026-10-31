@echo off
setlocal EnableExtensions
set "ROOT=%~dp0.."
pushd "%ROOT%" >nul || exit /b 1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\scripts\watchdog_24x7.ps1" %*
set "CODE=%ERRORLEVEL%"
popd
exit /b %CODE%
