# Bounded one-generation smoke run through all A-D levels.
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$Python = if (Test-Path $VenvPython) { $VenvPython } else { "python" }
$SmokeDevice = if ($env:AMDE_SMOKE_DEVICE) { $env:AMDE_SMOKE_DEVICE } else { "cpu" }
Push-Location $Root
try {
    & $Python (Join-Path $Root "run_amde.py") --root $Root --device $SmokeDevice smoke @Arguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
