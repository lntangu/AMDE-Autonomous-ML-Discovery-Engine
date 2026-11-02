# AMDE Quickstart Guide

This guide walks through initializing a workspace, running a small experimental discovery campaign, and inspecting candidate results.

---

## 1. Minimal 5-Minute CPU Discovery Run

Run a small discovery session on the synthetic XOR benchmark:

```bash
python scripts/research_campaign.py \
    --campaign-id quickstart_demo \
    --rounds 2 \
    --generations 3 \
    --quick-budget 8 \
    --confirm-budget 4 \
    --tasks xor_classification
```

### What Happens:
1. An island population of candidate models is initialized at Level A (hyperparameters).
2. Candidates are evaluated on the quick filter stage.
3. High-performing candidates advance to the confirmation stage.
4. Top candidates and sightings are stored in `storage/discovery_archive.sqlite3`.

---

## 2. Inspecting Discovery Results

To query the discovery archive:

```python
import sqlite3

conn = sqlite3.connect("storage/discovery_archive.sqlite3")
cur = conn.cursor()

cur.execute("""
    SELECT candidate_hash, level, best_quick_score, best_confirm_score 
    FROM discovery_candidates 
    ORDER BY best_confirm_score DESC 
    LIMIT 5
""")

for row in cur.fetchall():
    print(row)
conn.close()
```

---

## 3. Running with Hard Deadline Enforcement

To launch a supervised campaign with strict process containment and a hard deadline:

```bash
python scripts/amde2.py run-campaign \
    --campaign-id timed_demo \
    --duration-hours 0.1 \
    --deadline-policy hard \
    --mode classical
```

The Guardian process will monitor execution and terminate all subprocesses cleanly once the hard deadline expires.
