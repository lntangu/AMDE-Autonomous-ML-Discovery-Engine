@echo off
setlocal EnableExtensions
rem Examples: status.bat   status.bat --all   status.bat --json --run-id RUN_ID
set "ROOT=%~dp0.."
pushd "%ROOT%" >nul || exit /b 1
if exist "%ROOT%\.venv\Scripts\python.exe" (
    set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)
"%PYTHON%" "%ROOT%\run_amde.py" --root "%ROOT%" status %*
set "CODE=%ERRORLEVEL%"
popd
exit /b %CODE%
