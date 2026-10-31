# Examples: .\status.ps1   .\status.ps1 --all   .\status.ps1 --json --run-id RUN_ID
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$Python = if (Test-Path $VenvPython) { $VenvPython } else { "python" }
Push-Location $Root
try {
    & $Python (Join-Path $Root "run_amde.py") --root $Root status @Arguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
