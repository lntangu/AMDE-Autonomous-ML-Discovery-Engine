@echo off
setlocal
set "PROJECT_ROOT=%~dp0.."
if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    "%PROJECT_ROOT%\.venv\Scripts\python.exe" "%~dp0hybrid_control.py" %*
) else (
    py -3 "%~dp0hybrid_control.py" %*
)
exit /b %ERRORLEVEL%
