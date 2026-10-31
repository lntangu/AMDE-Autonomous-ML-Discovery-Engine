@echo off
setlocal EnableExtensions
rem Graceful stop for the latest run, or pass a run id / --global.
set "ROOT=%~dp0.."
pushd "%ROOT%" >nul || exit /b 1
if exist "%ROOT%\.venv\Scripts\python.exe" (
    set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)
"%PYTHON%" "%ROOT%\run_amde.py" --root "%ROOT%" stop %*
set "CODE=%ERRORLEVEL%"
if not "%CODE%"=="0" (
    echo.
    echo If no run has been created yet, there is nothing to stop.
)
popd
exit /b %CODE%
