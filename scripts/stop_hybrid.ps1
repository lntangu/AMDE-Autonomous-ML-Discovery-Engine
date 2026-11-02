$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Runs = Join-Path $Root "runs"
New-Item -ItemType Directory -Force -Path $Runs | Out-Null
Set-Content -LiteralPath (Join-Path $Runs "HYBRID_STOP") -Value "Hybrid stop requested at $(Get-Date -Format o)" -Encoding UTF8
Write-Host "Stop requested. The supervisor will stop both workers cooperatively."
