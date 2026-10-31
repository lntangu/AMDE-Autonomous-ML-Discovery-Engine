# AMDE Migration Guide: Upgrading to 2.0.1

This document outlines the migration path and compatibility guarantees when upgrading from **AMDE 1.3.0** or **AMDE 2.0.0** to **AMDE 2.0.1**.

---

## 1. Compatibility & Data Protection Principles

AMDE 2.0.1 maintains **absolute backward compatibility** with existing data assets:
- **No Schema Destruction**: The SQLite discovery archive (`storage/discovery_archive.sqlite3`), campaign databases, and ledger schemas remain completely compatible.
- **Existing Campaigns Preserved**: Campaigns generated under 1.3.0 (`amde130hp001`) and 2.0.0 (`amde2main001`, `amde2six001`) are preserved intact.
- **No Silent Plan Mutation**: Pre-existing `plan.json` files retain their canonical SHA-256 hashes. Resuming a 2.0.0 campaign under 2.0.1 requires no manual edits to `plan.json`.

---

## 2. Key Architectural Upgrades in 2.0.1

| Component | AMDE 2.0.0 Behavior | AMDE 2.0.1 Enhanced Behavior |
| :--- | :--- | :--- |
| **File Operations on Windows** | Raw `os.replace` (threw `WinError 5` / `WinError 32` under sharing collisions) | `robust_replace` with exponential backoff and jitter (up to 12 retries over 3.5s) |
| **Worker Heartbeat Flush** | Worker threw exception and exited with code 2 on transient lock failure | Protected with try/except retry loop; does not abort running model evaluations |
| **Exit Code 2 Handling** | Manager terminated campaign immediately on any code 2 exit | Startup fatal (<10s, round 0) triggers `needs_attention`; in-flight code 2 follows restart budget and backoff |
| **Campaign Duration Governor** | Relied on external detached PowerShell `Start-Sleep` | Cooperative persistent deadline embedded in `plan.json` (`--duration-hours`, `deadline_utc`) |
| **Campaign Directory Creation** | `start` crashed with `FileExistsError` if harness created `logs/` early | `start` accepts pre-created directory if `plan.json` does not exist yet |
| **Worker Restart Clock** | Used wall clock `time.time()` for `next_start_at`, compared against `time.monotonic()` | Uses `time.monotonic()` consistently; workers restart reliably |
| **Worker Startup Deadline** | Workers hung before first heartbeat could stall indefinitely | 180s startup deadline monitored by manager; terminates and reschedules hung workers |
| **Campaign Modes** | Hardcoded hybrid execution only | Selectable `--mode {hybrid, classical, quantum}` |

---

## 3. Step-by-Step Migration Instructions

### Step 1: Verify Environment & Prerequisites
Ensure `psutil` is installed in the target Python environment:
```bash
python -m pip install --upgrade psutil
```

### Step 2: Verify Existing Historical Campaigns
Before resuming or modifying any existing campaigns, verify their data integrity:
```bash
# Verify historical campaign amde2main001:
python scripts/amde2.py verify --campaign-id amde2main001

# Run runtime preflight doctor:
python scripts/amde2.py doctor --device cpu
```

### Step 3: Resuming an Existing 2.0.0 Campaign
To resume `amde2main001` under the upgraded 2.0.1 runtime:
```bash
python scripts/amde2.py resume --campaign-id amde2main001
```
The manager will load the existing plan, adopt the SQLite database, and continue from the next incomplete trial.

### Step 4: Running New Campaigns with Persistent Deadlines
For new campaigns, use the updated CLI flags or the refactored PowerShell harnesses:
```powershell
# Bounded 6-hour test with persistent deadline:
powershell -ExecutionPolicy Bypass -File .\AMDE2_Run_6Hour_Test.ps1 -CampaignId amde2six001 -DurationHours 6
```
