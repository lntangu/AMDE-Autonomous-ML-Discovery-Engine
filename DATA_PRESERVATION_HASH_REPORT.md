# AMDE Historical Data Preservation & Integrity Report

**Verification Timestamp**: 2026-10-31  
**Verification Scope**: All historical research databases, evidence archives, and run checkpoints.  
**Methodology**: Non-destructive read-only SHA-256 hashing and SQLite `PRAGMA integrity_check`.  

---

## 1. Canonical Evidence Archives

| Evidence Archive | Recorded Historical Digest | Current Calculated Digest | Status |
| :--- | :--- | :--- | :--- |
| `amde130hp001_audit_evidence.zip` | `2d8ca23df164992ff09e6822c92d886405eceb144f0348cc391ad4916ae3042b` | `2d8ca23df164992ff09e6822c92d886405eceb144f0348cc391ad4916ae3042b` | **PASS (Match)** |

---

## 2. Historical SQLite Research Ledgers & Discovery Archives

| Database Path | SHA-256 Digest | Size (KiB) | Integrity Check | Row Count | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `storage/discovery_archive.sqlite3` | `efbe67bb8ac8bb4e...` | 116.0 | `ok` | 216 | **PASS** |
| `reports/quantum_campaigns/amde130hp001_quantum/quantum_ledger.sqlite3` | `d12d1709117bd979...` | 4332.0 | `ok` | 3075 | **PASS** |
| `reports/quantum_campaigns/amde2main001_quantum/quantum_ledger.sqlite3` | `501d327e20cd7afa...` | 108.0 | `ok` | 63 | **PASS** |
| `runs/amde130_cpu_smoke_001/ledger.sqlite3` | `2a7f65c07cdb6194...` | 184.0 | `ok` | 54 | **PASS** |
| `runs/amde130_cuda_smoke_001/ledger.sqlite3` | `974e2cbe031815f8...` | 184.0 | `ok` | 58 | **PASS** |
| `runs/amde130hp001_classical_r00000/ledger.sqlite3` | `6aea52942282c3fb...` | 312.0 | `ok` | 74 | **PASS** |
| `runs/amde130hp001_classical_r00001/ledger.sqlite3` | `4e0c75a28d508e69...` | 344.0 | `ok` | 82 | **PASS** |
| `runs/amde130hp001_classical_r00002/ledger.sqlite3` | `c8bd653a3a8d0ddb...` | 344.0 | `ok` | 85 | **PASS** |
| `runs/amde130hp001_classical_r00003/ledger.sqlite3` | `2106b6e9a7ebeab2...` | 344.0 | `ok` | 84 | **PASS** |
| `runs/amde130hp001_classical_r00004/ledger.sqlite3` | `c970193c0c6100b4...` | 352.0 | `ok` | 83 | **PASS** |
| `runs/amde130hp001_classical_r00005/ledger.sqlite3` | `fa1d7b9e13f53ba0...` | 356.0 | `ok` | 81 | **PASS** |
| `runs/amde130hp001_classical_r00006/ledger.sqlite3` | `7ab6a8b84bf9c799...` | 352.0 | `ok` | 84 | **PASS** |
| `runs/amde130hp001_classical_r00007/ledger.sqlite3` | `f08c2abd7ec720cb...` | 356.0 | `ok` | 80 | **PASS** |
| `runs/amde130hp001_classical_r00008/ledger.sqlite3` | `ddf74a6a98ff6aab...` | 352.0 | `ok` | 85 | **PASS** |
| `runs/amde130hp001_classical_r00009/ledger.sqlite3` | `ab4e57755e77d1de...` | 352.0 | `ok` | 83 | **PASS** |
| `runs/amde130hp001_classical_r00010/ledger.sqlite3` | `ec5ffcabbc8bfbbc...` | 352.0 | `ok` | 83 | **PASS** |
| `runs/amde130hp001_classical_r00011/ledger.sqlite3` | `72af055b1e8aba85...` | 356.0 | `ok` | 82 | **PASS** |
| `runs/amde130hp001_classical_r00012/ledger.sqlite3` | `cdf4a1c7a0acc298...` | 352.0 | `ok` | 82 | **PASS** |
| `runs/amde130hp001_classical_r00013/ledger.sqlite3` | `8b8bb0c7a50e857c...` | 248.0 | `ok` | 26 | **PASS** |
| `runs/amde2main001_classical_r00000/ledger.sqlite3` | `8ca0a59caea375e0...` | 240.0 | `ok` | 26 | **PASS** |

---

## 3. Checkpoint Envelopes & Checksum Verification

| Checkpoint File | Schema Version | Envelope Checksum Valid | Status |
| :--- | :--- | :--- | :--- |
| `runs/amde130_cpu_smoke_001/checkpoints/latest.json` | `1.0.0` | `True` | **PASS** |
| `runs/amde130_cuda_smoke_001/checkpoints/latest.json` | `1.0.0` | `True` | **PASS** |
| `runs/amde130hp001_classical_r00000/checkpoints/latest.json` | `1.0.0` | `True` | **PASS** |
| `runs/amde130hp001_classical_r00001/checkpoints/latest.json` | `1.0.0` | `True` | **PASS** |
| `runs/amde130hp001_classical_r00002/checkpoints/latest.json` | `1.0.0` | `True` | **PASS** |
| `runs/amde130hp001_classical_r00003/checkpoints/latest.json` | `1.0.0` | `True` | **PASS** |
| `runs/amde130hp001_classical_r00004/checkpoints/latest.json` | `1.0.0` | `True` | **PASS** |
| `runs/amde130hp001_classical_r00005/checkpoints/latest.json` | `1.0.0` | `True` | **PASS** |
| `runs/amde130hp001_classical_r00006/checkpoints/latest.json` | `1.0.0` | `True` | **PASS** |
| `runs/amde130hp001_classical_r00007/checkpoints/latest.json` | `1.0.0` | `True` | **PASS** |
| `runs/amde130hp001_classical_r00008/checkpoints/latest.json` | `1.0.0` | `True` | **PASS** |
| `runs/amde130hp001_classical_r00009/checkpoints/latest.json` | `1.0.0` | `True` | **PASS** |
| `runs/amde130hp001_classical_r00010/checkpoints/latest.json` | `1.0.0` | `True` | **PASS** |
| `runs/amde130hp001_classical_r00011/checkpoints/latest.json` | `1.0.0` | `True` | **PASS** |
| `runs/amde130hp001_classical_r00012/checkpoints/latest.json` | `1.0.0` | `True` | **PASS** |
| `runs/amde130hp001_classical_r00013/checkpoints/latest.json` | `1.0.0` | `True` | **PASS** |
| `runs/amde2main001_classical_r00000/checkpoints/latest.json` | `1.0.0` | `True` | **PASS** |

---

## 4. Preservation Conclusion

- Canonical Evidence ZIP: **VERIFIED INTACT**
- SQLite Databases: **20 verified** (ALL PASS)
- Checkpoints: **17 verified** (ALL PASS)
- Historical Campaigns `amde130hp001`, `amde2main001`: **100% Preserved without modification or data loss.**
