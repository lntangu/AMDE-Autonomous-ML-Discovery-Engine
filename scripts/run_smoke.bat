@echo off
setlocal EnableExtensions
set "ROOT=%~dp0.."
pushd "%ROOT%" >nul || exit /b 1
if exist "%ROOT%\.venv\Scripts\python.exe" (
    set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)
rem The tiny smoke workload is CPU-first to avoid CUDA startup overhead.
rem Set AMDE_SMOKE_DEVICE=cuda (or auto) explicitly to test that backend.
if not defined AMDE_SMOKE_DEVICE set "AMDE_SMOKE_DEVICE=cpu"
"%PYTHON%" "%ROOT%\run_amde.py" --root "%ROOT%" --device "%AMDE_SMOKE_DEVICE%" smoke %*
set "CODE=%ERRORLEVEL%"
popd
exit /b %CODE%
