[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$')]
    [string]$CampaignId,

    [string]$PythonPath = ".\.venv\Scripts\python.exe",

    [switch]$AdoptCurrentCampaign
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path ".").Path
$Python = (Resolve-Path $PythonPath).Path
$Entry = Join-Path $Root "scripts\amde2.py"
$Plan = Join-Path $Root "reports\amde2\$CampaignId\plan.json"
if (-not (Test-Path $Entry)) { throw "Missing AMDE 2.0 entrypoint: $Entry" }
if (-not (Test-Path $Plan)) { throw "Campaign plan not found: $Plan. Start the campaign once before installing auto-start." }

$TaskName = "AMDE2-$CampaignId"
$Identity = "$env:USERDOMAIN\$env:USERNAME"
$Arguments = '"{0}" guardian --root "{1}" --campaign-id "{2}"' -f $Entry, $Root, $CampaignId
$Action = New-ScheduledTaskAction -Execute $Python -Argument $Arguments -WorkingDirectory $Root
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $Identity
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew `
    -DontStopIfGoingOnBatteries
$Principal = New-ScheduledTaskPrincipal -UserId $Identity -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings `
    -Principal $Principal -Description "AMDE 2.0 campaign guardian for $CampaignId" -Force | Out-Null

Write-Host "Registered Task Scheduler entry: $TaskName"
Write-Host "Trigger: when $Identity logs on; restart on failure; unlimited execution time."

if ($AdoptCurrentCampaign) {
    Write-Host "Switching the active campaign to Task Scheduler supervision..."
    & $Python $Entry stop --root $Root --campaign-id $CampaignId --wait-seconds 1200
    if ($LASTEXITCODE -ne 0) { throw "Cooperative stop did not complete; inspect campaign status/logs before proceeding." }
    & $Python $Entry resume --root $Root --campaign-id $CampaignId
    if ($LASTEXITCODE -ne 0) { throw "Campaign resume through Task Scheduler failed." }
    Write-Host "Campaign is now supervised by Task Scheduler."
}
else {
    Write-Host "To move an already-running campaign under Task Scheduler, rerun with -AdoptCurrentCampaign."
    Write-Host "Do not start a second guardian manually while one is already active."
}
