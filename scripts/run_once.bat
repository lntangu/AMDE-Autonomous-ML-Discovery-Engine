@echo off
setlocal EnableExtensions
rem Bounded batch run. Defaults come from config/default.json.
set "ROOT=%~dp0.."
pushd "%ROOT%" >nul || exit /b 1
if exist "%ROOT%\.venv\Scripts\python.exe" (
    set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)
"%PYTHON%" "%ROOT%\run_amde.py" --root "%ROOT%" once %*
set "CODE=%ERRORLEVEL%"
popd
exit /b %CODE%
