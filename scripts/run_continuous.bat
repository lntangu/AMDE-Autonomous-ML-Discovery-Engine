@echo off
setlocal EnableExtensions
rem Long-running mode. Stop with scripts\stop.bat or a time/generation budget.
set "ROOT=%~dp0.."
pushd "%ROOT%" >nul || exit /b 1
if exist "%ROOT%\.venv\Scripts\python.exe" (
    set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)
"%PYTHON%" "%ROOT%\run_amde.py" --root "%ROOT%" continuous %*
set "CODE=%ERRORLEVEL%"
popd
exit /b %CODE%
