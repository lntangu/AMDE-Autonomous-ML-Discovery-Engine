[CmdletBinding()]
param(
    [string]$RunId = "amde_24x7",
    [switch]$ClearStop
)
$ErrorActionPreference = "Stop"
# Run IDs are path components, never arbitrary user-supplied paths. Dots and
# separators are deliberately disallowed in the watchdog even though the CLI
# accepts a wider safe character set.
if ($RunId -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$') {
    throw "Unsafe RunId '$RunId'. Use 1-96 ASCII letters, digits, '_' or '-' and start with a letter/digit."
}
$WindowsReservedIds = @('STOP', 'CON', 'PRN', 'AUX', 'NUL') +
    @(1..9 | ForEach-Object { "COM$_"; "LPT$_" })
if ($WindowsReservedIds -contains $RunId.ToUpperInvariant()) {
    throw "RunId '$RunId' is reserved by the watchdog or Windows filesystem. Choose a different ID."
}
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Runs = Join-Path $Root "runs"
$RunDir = Join-Path $Runs $RunId
$GlobalStop = Join-Path $Runs "STOP"
$RunStop = Join-Path $RunDir "STOP"
$LogDir = Join-Path $Root "reports\watchdog"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$MasterLog = Join-Path $LogDir "watchdog.log"
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$Python = if (Test-Path $VenvPython) { $VenvPython } else { (Get-Command python -ErrorAction Stop).Source }
$Launcher = Join-Path $Root "run_amde.py"

function Write-WatchLog([string]$Message) {
    if ((Test-Path $MasterLog) -and ((Get-Item -LiteralPath $MasterLog).Length -gt 5MB)) {
        for ($i = 3; $i -ge 1; $i--) {
            $old = "$MasterLog.$i"
            $new = "$MasterLog.$($i + 1)"
            if (Test-Path $old) { Move-Item -LiteralPath $old -Destination $new -Force }
        }
        Move-Item -LiteralPath $MasterLog -Destination "$MasterLog.1" -Force
    }
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $MasterLog -Value $line -Encoding UTF8
    Write-Host $line
}

function Prune-AttemptLogs {
    # Keep the newest 40 per-attempt stdout/stderr logs; experiment ledger and
    # checkpoints are never removed by the watchdog.
    $oldLogs = @(Get-ChildItem -LiteralPath $LogDir -Filter "worker_*.log" -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -Skip 40)
    foreach ($item in $oldLogs) {
        Remove-Item -LiteralPath $item.FullName -Force -ErrorAction SilentlyContinue
    }
}

if ($ClearStop) {
    Remove-Item -LiteralPath $GlobalStop -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $RunStop -Force -ErrorAction SilentlyContinue
    Write-WatchLog "explicit --ClearStop requested; cleared stop files for run '$RunId'"
}
if (Test-Path $GlobalStop) {
    Write-WatchLog "global STOP exists at $GlobalStop; remove it or restart with -ClearStop after verifying no other run should stop"
    exit 20
}
if (Test-Path $RunStop) {
    Write-WatchLog "run STOP exists at $RunStop; resume intentionally with -ClearStop after inspecting prior run"
    exit 20
}

Push-Location $Root
$consecutiveFailures = 0
$attempt = 0
try {
    Write-WatchLog "watchdog start; root=$Root run_id=$RunId python=$Python"
    while ($true) {
        if ((Test-Path $GlobalStop) -or (Test-Path $RunStop)) {
            Write-WatchLog "stop file detected; watchdog exits without restarting the worker"
            break
        }
        $attempt++
        Prune-AttemptLogs
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
        $OutLog = Join-Path $LogDir "worker_${stamp}_out.log"
        $ErrLog = Join-Path $LogDir "worker_${stamp}_err.log"
        Write-WatchLog "starting worker attempt=$attempt"
        $argumentLine = '"{0}" --root "{1}" continuous --run-id "{2}"' -f $Launcher, $Root, $RunId
        $proc = Start-Process -FilePath $Python -ArgumentList $argumentLine -WorkingDirectory $Root -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog -PassThru -Wait
        $code = [int]$proc.ExitCode
        Write-WatchLog "worker exited code=$code pid=$($proc.Id) stdout=$OutLog stderr=$ErrLog"

        if ((Test-Path $GlobalStop) -or (Test-Path $RunStop)) {
            Write-WatchLog "stop requested; no restart"
            break
        }
        if ($code -eq 2 -or $code -eq 20) {
            Write-WatchLog "configuration/validation failure; circuit breaker stops auto-restarts. Fix the cause then relaunch."
            exit $code
        }
        if ($code -ne 0) {
            $consecutiveFailures++
            if ($consecutiveFailures -ge 5) {
                Write-WatchLog "5 consecutive non-zero exits; circuit breaker stopped the watchdog to prevent an error loop"
                exit 1
            }
            $delay = [Math]::Min(300, [int](5 * [Math]::Pow(2, $consecutiveFailures - 1)))
            Write-WatchLog "transient failure; retry in ${delay}s (failure $consecutiveFailures/5)"
            Start-Sleep -Seconds $delay
            continue
        }
        $consecutiveFailures = 0
        # A resource-pressure pause writes a resumable checkpoint; wait longer
        # before retrying so the watcher does not repeatedly hit the same wall.
        $summaryPath = Join-Path $RunDir "summary.json"
        $lastStatus = $null
        if (Test-Path $summaryPath) {
            try { $lastStatus = (Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json).status }
            catch { $lastStatus = $null }
        }
        if ($lastStatus -eq "paused") {
            $pauseReason = ""
            try { $pauseReason = [string]((Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json).error.message) } catch { $pauseReason = "" }
            if ($pauseReason -match '(?i)storage budget|disk free .*below reserve') {
                Write-WatchLog "persistent storage/disk pressure detected; stopping watchdog rather than restarting every 5 minutes. Review disk use, then relaunch intentionally. reason=$pauseReason"
                exit 20
            }
            Write-WatchLog "run paused by transient resource pressure; retry in 300s (review logs if it repeats)"
            Start-Sleep -Seconds 300
        } else {
            # Continuous mode should not exit on its own. This delay prevents a
            # bad configuration from creating a rapid restart loop.
            Start-Sleep -Seconds 10
        }
    }
    Write-WatchLog "watchdog finished"
} finally {
    Pop-Location
}
