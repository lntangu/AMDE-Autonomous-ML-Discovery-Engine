# Create a project-local virtual environment. No global packages are changed.
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Venv = Join-Path $Root ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Host "Creating virtual environment: $Venv"
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv $Venv
    } else {
        & python -m venv $Venv
    }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Python)) {
        throw "Could not create the virtual environment."
    }
} else {
    Write-Host "Reusing existing virtual environment: $Venv"
}

& $Python -c "import sys; print('Python', sys.version.split()[0], 'ready')"
if ($LASTEXITCODE -ne 0) { throw "The virtual environment Python is not usable." }

& $Python (Join-Path $Root "scripts\register_source_path.py")
if ($LASTEXITCODE -ne 0) { throw "Could not register the source path in the virtual environment." }

Write-Host ""
Write-Host "Setup complete. The core engine uses only the Python standard library."
Write-Host "Optional monitoring is not installed automatically. To add psutil:"
Write-Host "  `"$Python`" -m pip install psutil"
Write-Host ""
Write-Host "Launchers use .venv automatically when it exists."
