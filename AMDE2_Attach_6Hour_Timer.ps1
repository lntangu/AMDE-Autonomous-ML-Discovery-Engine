[CmdletBinding()]
param(
    [ValidateSet('Attach', 'Timer')]
    [string]$Mode = 'Attach',
    [string]$ProjectRoot = (Get-Location).Path,
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$')]
    [string]$CampaignId = 'amde2main001',
    [ValidateRange(1, 168)]
    [int]$DurationHours = 6
)

$ErrorActionPreference = 'Stop'

try {
    $Root = (Resolve-Path -LiteralPath $ProjectRoot).Path
} catch {
    throw "Project root does not exist: $ProjectRoot"
}

$PythonPath = Join-Path $Root '.venv\Scripts\python.exe'
$CliPath = Join-Path $Root 'scripts\amde2.py'
$CampaignBase = Join-Path $Root "reports\amde2\$CampaignId"
$LogDirectory = Join-Path $CampaignBase 'logs'
$TimerLog = Join-Path $LogDirectory 'six_hour_attach_timer.log'
$TimerRecord = Join-Path $CampaignBase 'six_hour_attach_timer.json'
$ResultPath = Join-Path $CampaignBase 'six_hour_attach_test_result.json'

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) { throw "Python environment missing: $PythonPath" }
if (-not (Test-Path -LiteralPath $CliPath -PathType Leaf)) { throw "AMDE 2.0 CLI missing: $CliPath" }
if (-not (Test-Path -LiteralPath (Join-Path $CampaignBase 'plan.json') -PathType Leaf)) {
    throw "Campaign plan not found: $CampaignBase. This script attaches a timer; it does not create a campaign."
}

New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null

function Write-TimerLog {
    param([string]$Message)
    Add-Content -LiteralPath $TimerLog -Value "[$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssK')] $Message" -Encoding UTF8
}

function Invoke-AMDECLI {
    param([string[]]$CliArguments)
    $output = & $PythonPath $CliPath @CliArguments 2>&1
    $code = $LASTEXITCODE
    return [pscustomobject]@{
        ExitCode = [int]$code
        Text = (($output | Out-String).TrimEnd())
    }
}

if ($Mode -eq 'Timer') {
    Write-TimerLog "Detached timer started; campaign=$CampaignId; duration_hours=$DurationHours; pid=$PID"
    Start-Sleep -Seconds ($DurationHours * 3600)
    Write-TimerLog 'Duration elapsed; requesting campaign-scoped cooperative stop.'

    $stop = Invoke-AMDECLI -CliArguments @('stop', '--root', $Root, '--campaign-id', $CampaignId, '--wait-seconds', '1800')
    Write-TimerLog "STOP exit=$($stop.ExitCode); output=$($stop.Text -replace '[\r\n]+', ' | ')"

    $status = Invoke-AMDECLI -CliArguments @('status', '--root', $Root, '--campaign-id', $CampaignId)
    Write-TimerLog "STATUS exit=$($status.ExitCode); output=$($status.Text -replace '[\r\n]+', ' | ')"

    $logs = Invoke-AMDECLI -CliArguments @('logs', '--root', $Root, '--campaign-id', $CampaignId, '--tail', '150')
    if ($logs.Text) { Add-Content -LiteralPath $TimerLog -Value $logs.Text -Encoding UTF8 }

    $isStopped = ($stop.ExitCode -eq 0) -and ($status.ExitCode -eq 0) -and ($status.Text -match '(?m)^Campaign state: stopped$')
    $verifyCode = $null
    $backupCode = $null
    $backupOutput = 'Skipped because clean cooperative shutdown was not confirmed.'

    if ($isStopped) {
        $verify = Invoke-AMDECLI -CliArguments @('verify', '--root', $Root, '--campaign-id', $CampaignId)
        $verifyCode = $verify.ExitCode
        Write-TimerLog "VERIFY exit=$($verify.ExitCode); output=$($verify.Text -replace '[\r\n]+', ' | ')"
        if ($verify.ExitCode -eq 0) {
            $backup = Invoke-AMDECLI -CliArguments @('backup', '--root', $Root, '--campaign-id', $CampaignId)
            $backupCode = $backup.ExitCode
            $backupOutput = $backup.Text
            Write-TimerLog "BACKUP exit=$($backup.ExitCode); output=$($backup.Text -replace '[\r\n]+', ' | ')"
        }
    }

    $summary = [ordered]@{
        campaign_id = $CampaignId
        duration_hours = $DurationHours
        timer_pid = $PID
        timer_finished_at = (Get-Date).ToString('o')
        stop_exit_code = $stop.ExitCode
        status_exit_code = $status.ExitCode
        confirmed_stopped = [bool]$isStopped
        verify_exit_code = $verifyCode
        backup_exit_code = $backupCode
        backup_output = $backupOutput
        note = if ($isStopped) { 'Cooperative stop confirmed; verification and backup attempted.' } else { 'Stop was not confirmed. No automatic backup was attempted; preserve logs and inspect campaign state before any further action.' }
    }
    $summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $ResultPath -Encoding UTF8
    Write-TimerLog "FINISHED confirmed_stopped=$isStopped verify=$verifyCode backup=$backupCode"
    Write-Host "Six-hour timer finished. Result: $ResultPath"
    if (-not $isStopped -or $verifyCode -ne 0 -or $backupCode -ne 0) { exit 1 }
    exit 0
}

# Attach mode: require an already-running campaign; never call 'start'.
Write-Host "Checking existing campaign '$CampaignId'..."
$status = Invoke-AMDECLI -CliArguments @('status', '--root', $Root, '--campaign-id', $CampaignId)
$status.Text | Write-Output
if ($status.ExitCode -ne 0 -or
    $status.Text -notmatch '(?m)^Campaign state: running$' -or
    $status.Text -notmatch '(?m)^Guardian: PID=\d+ alive=True' -or
    $status.Text -notmatch '(?m)^Manager: PID=\d+ alive=True') {
    throw 'Campaign is not confirmed running with live Guardian and Manager. No timer was launched.'
}

if (Test-Path -LiteralPath $TimerRecord) {
    $oldRecord = $null
    try { $oldRecord = Get-Content -LiteralPath $TimerRecord -Raw | ConvertFrom-Json } catch {}
    if ($null -ne $oldRecord -and $oldRecord.timer_pid) {
        $oldProcess = Get-Process -Id ([int]$oldRecord.timer_pid) -ErrorAction SilentlyContinue
        if ($null -ne $oldProcess) { throw "A timer process may already be active (PID $($oldRecord.timer_pid)). Refusing to attach a duplicate." }
    }
    $archive = "$TimerRecord.previous.$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    Move-Item -LiteralPath $TimerRecord -Destination $archive
}

$WindowsPowerShell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
if (-not (Test-Path -LiteralPath $WindowsPowerShell -PathType Leaf)) { throw 'Windows PowerShell executable not found; no timer was launched.' }
$ScriptPath = (Resolve-Path -LiteralPath $PSCommandPath).Path
$argLine = '-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "{0}" -Mode Timer -ProjectRoot "{1}" -CampaignId {2} -DurationHours {3}' -f $ScriptPath, $Root, $CampaignId, $DurationHours
$timerProcess = Start-Process -FilePath $WindowsPowerShell -ArgumentList $argLine -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 2
$timerAlive = $null -ne (Get-Process -Id $timerProcess.Id -ErrorAction SilentlyContinue)
if (-not $timerAlive) {
    throw "Timer process exited immediately (PID $($timerProcess.Id)). Inspect $TimerLog; campaign was not stopped."
}

$record = [ordered]@{
    campaign_id = $CampaignId
    duration_hours = $DurationHours
    attached_at = (Get-Date).ToString('o')
    planned_stop_at = (Get-Date).AddHours($DurationHours).ToString('o')
    timer_pid = $timerProcess.Id
    timer_mode = 'detached_hidden_powershell_attached_to_existing_campaign'
}
$record | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $TimerRecord -Encoding UTF8
Write-TimerLog "Attached detached timer PID=$($timerProcess.Id); campaign=$CampaignId; duration_hours=$DurationHours; planned_stop_at=$($record.planned_stop_at)"
Write-Host ''
Write-Host '===== SIX-HOUR TIMER ATTACHED ====='
Write-Host "Campaign ID : $CampaignId"
Write-Host "Timer PID   : $($timerProcess.Id)"
Write-Host "Stop at     : $($record.planned_stop_at)"
Write-Host "Timer log   : $TimerLog"
Write-Host 'The timer will request cooperative stop, verify evidence, and back up only after clean shutdown.'
