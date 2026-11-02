# AMDE Data Storage Specification

AMDE separates trial accounting, candidate genome tracking, and campaign telemetry into independent, crash-resilient storage layers.

---

## 1. Storage Architecture

1. **`storage/discovery_archive.sqlite3`**:
   - `discovery_candidates`: Stores unique canonical genome JSON representations, representation level (A–D), complexity scores, first/last seen timestamps, and best validation/confirmation scores.
   - `discovery_sightings`: Records individual evaluation events across runs and generations with generating operator names (`crossover`, `mutation`, `random`, `campaign_seed`).
2. **`reports/quantum_campaigns/<id>/quantum_ledger.sqlite3`**:
   - `model_trials`: Records every quantum trial (VQC, kernel ridge), config JSON, random seed, execution duration, and numerical outcome.
   - `rounds`: Tracks round-level metadata and completion timestamps.
3. **Write-Ahead Logging (WAL)**:
   - All SQLite connections open in WAL mode (`PRAGMA journal_mode=WAL;`).
   - Ensures readers do not block writers and transactions remain ACID-compliant even under sudden power loss or process kill.

---

## 2. Backup & Restoration

Campaigns support full-fidelity backup and offline verification:
```bash
# Verify backup archive checksum and SQLite database integrity
python scripts/amde2.py verify-backup backups/campaign_20261101.zip

# Restore backup to an independent target directory
python scripts/amde2.py restore backups/campaign_20261101.zip --root C:\Restored_Campaign --overwrite
```
Restoration re-validates file counts, SHA-256 digests, and executes `PRAGMA integrity_check;` on all restored SQLite databases.
