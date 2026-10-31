@echo off
setlocal EnableExtensions
rem Create a project-local virtual environment. No global packages are changed.
set "ROOT=%~dp0.."
pushd "%ROOT%" >nul || exit /b 1

if exist "%ROOT%\.venv\Scripts\python.exe" (
    echo Reusing existing virtual environment: %ROOT%\.venv
) else (
    echo Creating virtual environment: %ROOT%\.venv
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -m venv "%ROOT%\.venv"
    ) else (
        python -m venv "%ROOT%\.venv"
    )
    if errorlevel 1 (
        echo ERROR: could not create the virtual environment.
        popd
        exit /b 1
    )
)

"%ROOT%\.venv\Scripts\python.exe" -c "import sys; print('Python', sys.version.split()[0], 'ready')"
if errorlevel 1 (
    echo ERROR: the virtual environment Python is not usable.
    popd
    exit /b 1
)

rem Register the source tree for direct unittest discovery without a network install.
"%ROOT%\.venv\Scripts\python.exe" "%ROOT%\scripts\register_source_path.py"
if errorlevel 1 (
    echo ERROR: could not register the source path in the virtual environment.
    popd
    exit /b 1
)

echo.
echo Setup complete. The core engine uses only the Python standard library.
echo Optional monitoring is not installed automatically. If you want psutil:
echo   "%ROOT%\.venv\Scripts\python.exe" -m pip install psutil
echo.
echo Launchers use .venv automatically when it exists.
popd
exit /b 0
