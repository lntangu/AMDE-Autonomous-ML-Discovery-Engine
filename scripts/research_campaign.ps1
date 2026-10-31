param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$Python = if (Test-Path $VenvPython) { $VenvPython } else { "python" }
Push-Location $Root
try {
    & $Python (Join-Path $Root "scripts\research_campaign.py") --root $Root @Arguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
