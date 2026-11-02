@echo off
setlocal EnableExtensions
set "ROOT=%~dp0.."
pushd "%ROOT%" >nul || exit /b 1
if exist "%ROOT%\.venv\Scripts\python.exe" (
    set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)
"%PYTHON%" "%ROOT%\run_amde.py" --root "%ROOT%" doctor %*
set "CODE=%ERRORLEVEL%"
popd
exit /b %CODE%
