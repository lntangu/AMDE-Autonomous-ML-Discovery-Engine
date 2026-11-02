@echo off
setlocal
set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\python.exe"
if exist "%PY%" (
  "%PY%" "%~dp0amde2.py" %*
  exit /b %errorlevel%
)
py -3.14 "%~dp0amde2.py" %*
exit /b %errorlevel%
