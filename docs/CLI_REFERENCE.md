# AMDE CLI Reference

AMDE provides two CLI entrypoints: `amde` (high-level discovery interface) and `amde2` (operational lifecycle manager).

---

## 1. Operational Manager (`scripts/amde2.py` / `amde2`)

### Common Commands:

* `doctor`: Inspects platform capabilities, Python environment, CUDA support, and Windows Job Object availability.
  ```bash
  python scripts/amde2.py doctor
  ```

* `run-campaign`: Launches a supervised, crash-resilient endurance campaign.
  ```bash
  python scripts/amde2.py run-campaign \
      --campaign-id <id> \
      --duration-hours <float> \
      --deadline-policy <soft|hard> \
      --mode <classical|quantum|hybrid> \
      --seed <int>
  ```

* `status`: Reports active status, process tree, CPU/memory telemetry, and deadline countdown.
  ```bash
  python scripts/amde2.py status --campaign-id <id>
  ```

* `stop`: Initiates cooperative graceful shutdown of an active campaign.
  ```bash
  python scripts/amde2.py stop --campaign-id <id> --wait-seconds 120
  ```

* `verify-backup`: Verifies SHA-256 and ZIP integrity of a campaign backup.
  ```bash
  python scripts/amde2.py verify-backup <archive_path>
  ```

* `restore`: Restores an archived campaign into a specified target directory.
  ```bash
  python scripts/amde2.py restore <archive_path> --root <target_dir> --overwrite
  ```

---

## 2. Research Campaign Runners

* `scripts/research_campaign.py`: Runs classical evolutionary search across Levels A–D.
* `scripts/quantum_research_campaign.py`: Runs Variational Quantum Classifier (VQC) and quantum fidelity kernel ridge simulations.
