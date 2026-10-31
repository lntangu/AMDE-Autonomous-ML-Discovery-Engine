[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$')]
    [string]$CampaignId
)
$ErrorActionPreference = "Stop"
$TaskName = "AMDE2-$CampaignId"
$Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -eq $Task) {
    Write-Host "Task not found: $TaskName"
    exit 0
}
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
Write-Host "Removed auto-start task: $TaskName"
Write-Host "The campaign itself was not stopped or deleted."
