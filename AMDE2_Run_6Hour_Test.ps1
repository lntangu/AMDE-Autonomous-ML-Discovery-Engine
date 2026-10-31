[CmdletBinding()]
param(
    [ValidateSet('Start', 'StopAfter')]
    [string]$Mode = 'Start',

    [string]$ProjectRoot = (Get-Location).Path,

    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$')]
    [string]$CampaignId = 'amde2six001',

    [ValidateRange(1, 168)]
    [int]$DurationHours = 6
)

$ErrorActionPreference = 'Stop'

function Write-RunLog {
    param([string]$Path, [string]$Message)
    $line = "[$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssK')] $Message"
    Add-Content -LiteralPath $Path -Value $line -Encoding UTF8
}

function Invoke-AMDECLI {
    param(
        [string]$PythonPath,
        [string]$CliPath,
        [string]$LogPath,
        [string[]]$CliArguments
    )

    $output = & $PythonPath $CliPath @CliArguments 2>&1
    $exitCode = $LASTEXITCODE
    $outputText = ($output | Out-String).TrimEnd()
    if ($outputText) {
        Add-Content -LiteralPath $LogPath -Value $outputText -Encoding UTF8
    }
    Write-RunLog -Path $LogPath -Message "CLI exit code: $exitCode | args: $($CliArguments -join ' ')"
    return [int]$exitCode
}

try {
    $Root = (Resolve-Path -LiteralPath $ProjectRoot).Path
} catch {
    throw "Project root does not exist: $ProjectRoot"
}

$PythonPath = Join-Path $Root '.venv\Scripts\python.exe'
$CliPath = Join-Path $Root 'scripts\amde2.py'
$VersionScript = Join-Path $Root 'run_amde.py'
$QuantumDoctor = Join-Path $Root 'scripts\quantum_doctor.py'
$CampaignBase = Join-Path $Root "reports\amde2\$CampaignId"
$LogDirectory = Join-Path $CampaignBase 'logs'
$TimerLog = Join-Path $LogDirectory 'six_hour_test_timer.log'

if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Virtual environment not found: $PythonPath. Install AMDE 2.0 into this project first."
}
if (-not (Test-Path -LiteralPath $CliPath -PathType Leaf)) {
    throw "AMDE 2.0 runtime CLI not found: $CliPath. Extract the complete 2.0.0 ZIP first."
}

if ($Mode -eq 'StopAfter') {
    New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
    Write-RunLog -Path $TimerLog -Message "6-hour stop timer online; campaign=$CampaignId; duration_hours=$DurationHours; root=$Root"
    Write-RunLog -Path $TimerLog -Message "Timer process PID=$PID; waiting $($DurationHours * 3600) seconds."
    Start-Sleep -Seconds ($DurationHours * 3600)

    Write-RunLog -Path $TimerLog -Message 'Duration elapsed; requesting cooperative shutdown.'
    $stopCode = Invoke-AMDECLI -PythonPath $PythonPath -CliPath $CliPath -LogPath $TimerLog -CliArguments @(
        'stop', '--root', $Root, '--campaign-id', $CampaignId, '--wait-seconds', '1800'
    )

    $statusCode = Invoke-AMDECLI -PythonPath $PythonPath -CliPath $CliPath -LogPath $TimerLog -CliArguments @(
        'status', '--root', $Root, '--campaign-id', $CampaignId
    )
    $logsCode = Invoke-AMDECLI -PythonPath $PythonPath -CliPath $CliPath -LogPath $TimerLog -CliArguments @(
        'logs', '--root', $Root, '--campaign-id', $CampaignId, '--tail', '120'
    )
    $verifyCode = Invoke-AMDECLI -PythonPath $PythonPath -CliPath $CliPath -LogPath $TimerLog -CliArguments @(
        'verify', '--root', $Root, '--campaign-id', $CampaignId
    )
    $backupCode = Invoke-AMDECLI -PythonPath $PythonPath -CliPath $CliPath -LogPath $TimerLog -CliArguments @(
        'backup', '--root', $Root, '--campaign-id', $CampaignId
    )

    $summary = [ordered]@{
        campaign_id = $CampaignId
        duration_hours = $DurationHours
        timer_finished_at = (Get-Date).ToString('o')
        stop_exit_code = $stopCode
        status_exit_code = $statusCode
        logs_exit_code = $logsCode
        verify_exit_code = $verifyCode
        backup_exit_code = $backupCode
        note = 'No force-kill was attempted. Inspect logs if stop_exit_code is nonzero.'
    }
    $summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $CampaignBase 'six_hour_test_result.json') -Encoding UTF8
    Write-RunLog -Path $TimerLog -Message "6-hour timer finished. stop=$stopCode status=$statusCode logs=$logsCode verify=$verifyCode backup=$backupCode"
    Write-Host "6-hour stop sequence finished. Results: $CampaignBase"
    Write-Host "Timer log: $TimerLog"
    if ($stopCode -ne 0 -or $verifyCode -ne 0 -or $backupCode -ne 0) {
        Write-Warning 'At least one finalization step did not return PASS. Do not force-kill Python; inspect the timer log and campaign logs.'
        exit 1
    }
    exit 0
}

# START MODE: preflight before creating any campaign state.
Write-Host '===== AMDE 2.0 SIX-HOUR TEST PREFLIGHT ====='
$versionOutput = & $PythonPath $VersionScript --version 2>&1
$versionCode = $LASTEXITCODE
$versionText = ($versionOutput | Out-String).Trim()
$versionText
if ($versionCode -ne 0 -or $versionText -notmatch 'amde\s+2\.0\.[0-9]+') {
    throw "Expected AMDE 2.0.x; received '$versionText'. Do not start the test."
}

& $PythonPath $CliPath doctor --root $Root --device cuda --min-free-gb 5 --min-free-vram-mb 1536
if ($LASTEXITCODE -ne 0) {
    throw 'AMDE 2.0 doctor failed. Resolve every FAIL before starting the six-hour test.'
}

& $PythonPath $QuantumDoctor --device cuda
if ($LASTEXITCODE -ne 0) {
    throw 'Quantum CUDA doctor failed. The six-hour CUDA campaign was NOT started.'
}

$CollisionPaths = @(
    (Join-Path $Root "reports\amde2\$CampaignId\plan.json"),
    (Join-Path $Root "reports\amde2\$CampaignId\state.json"),
    (Join-Path $Root "reports\campaigns\${CampaignId}_classical\state.json"),
    (Join-Path $Root "reports\quantum_campaigns\${CampaignId}_quantum\state.json")
)
$Collisions = @($CollisionPaths | Where-Object { Test-Path -LiteralPath $_ })
if ($Collisions.Count -gt 0) {
    throw "Campaign ID '$CampaignId' already has state. Preserve it and choose a new CampaignId. Conflicts: $($Collisions -join '; ')"
}

Write-Host "Preflight PASS. Starting campaign '$CampaignId' for a six-hour measured run."
$startArgs = @(
    'start', '--root', $Root, '--campaign-id', $CampaignId,
    '--duration-hours', $DurationHours.ToString(),
    '--seed', '20261031',
    '--hours-per-round', '0.25',
    '--max-trials-per-round', '100',
    '--classical-max-gb', '8',
    '--quantum-max-gb', '4',
    '--min-free-gb', '5',
    '--quantum-device', 'cuda',
    '--classical-threads', '4',
    '--quantum-cpu-threads', '2',
    '--quantum-epochs', '6',
    '--quantum-batch-size', '16',
    '--quantum-train-size', '64',
    '--quantum-val-size', '32',
    '--quantum-test-size', '64',
    '--quantum-max-configs', '3',
    '--quantum-max-qubits', '8',
    '--quantum-statevector-mb', '256',
    '--memory-reserve-gb', '3',
    '--stale-heartbeat-seconds', '900',
    '--manager-stale-seconds', '180',
    '--stable-after-seconds', '600',
    '--max-worker-restarts-per-hour', '8',
    '--degraded-retry-seconds', '1800',
    '--shutdown-grace-seconds', '900',
    '--health-interval-seconds', '30',
    '--max-gpu-temperature-c', '80'
)
& $PythonPath $CliPath @startArgs
if ($LASTEXITCODE -ne 0) {
    throw 'AMDE 2.0 campaign did not start cleanly. Do not launch the timer.'
}

New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null

# Start a separate hidden PowerShell timer so closing this console does not cancel the 6-hour stop request.
$WindowsPowerShell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
if (-not (Test-Path -LiteralPath $WindowsPowerShell -PathType Leaf)) {
    & $PythonPath $CliPath stop --root $Root --campaign-id $CampaignId --wait-seconds 1800
    throw 'Windows PowerShell executable was not found. Campaign stop was requested because the stop timer could not be launched.'
}
$quotedScript = '"' + $PSCommandPath + '"'
$quotedRoot = '"' + $Root + '"'
$childArguments = '-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File {0} -Mode StopAfter -ProjectRoot {1} -CampaignId {2} -DurationHours {3}' -f $quotedScript, $quotedRoot, $CampaignId, $DurationHours
try {
    $timerProcess = Start-Process -FilePath $WindowsPowerShell -ArgumentList $childArguments -WindowStyle Hidden -PassThru
} catch {
    & $PythonPath $CliPath stop --root $Root --campaign-id $CampaignId --wait-seconds 1800
    throw "Could not launch six-hour stop timer; cooperative stop requested. Error: $_"
}

$timerInfo = [ordered]@{
    campaign_id = $CampaignId
    duration_hours = $DurationHours
    started_at = (Get-Date).ToString('o')
    planned_stop_at = (Get-Date).AddHours($DurationHours).ToString('o')
    timer_pid = $timerProcess.Id
    timer_mode = 'detached_hidden_powershell'
}
$timerInfo | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $CampaignBase 'six_hour_timer.json') -Encoding UTF8

& $PythonPath $CliPath status --root $Root --campaign-id $CampaignId
if ($LASTEXITCODE -ne 0) {
    Write-Warning 'Campaign started, but the immediate status query failed. The six-hour timer is still running.'
}

Write-Host ''
Write-Host '===== SIX-HOUR TEST STARTED ====='
Write-Host "Campaign ID : $CampaignId"
Write-Host "Timer PID   : $($timerProcess.Id)"
Write-Host "Stop at     : $($timerInfo.planned_stop_at)"
Write-Host "Timer log   : $TimerLog"
Write-Host 'The hidden timer will request cooperative stop after six hours, then verify and back up the campaign.'
Write-Host 'Keep Windows powered on and do not sign out or reboot during the test. Sleep is inhibited by the AMDE guardian while it remains alive.'
Write-Host 'Manual status: python scripts\amde2.py status --campaign-id ' $CampaignId
Write-Host 'Manual stop  : python scripts\amde2.py stop --campaign-id ' $CampaignId ' --wait-seconds 1800'
