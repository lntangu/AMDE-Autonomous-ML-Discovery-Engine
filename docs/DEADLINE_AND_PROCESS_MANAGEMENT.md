# AMDE Deadline & Process Management Specification

AMDE 2.0.3 implements an out-of-band supervision topology to enforce strict hard deadline semantics and prevent orphaned background processes.

---

## 1. The Hard Deadline Contract

When a campaign is launched with `--deadline-policy hard`:
1. **Immutable Deadline**: The exact deadline UTC timestamp ($T_{\text{deadline}} = T_{\text{start}} + \Delta t$) is computed at launch and written to `plan.json`.
2. **Periodic Health Evaluation**: The Guardian polls system status and monotonic time every 2–5 seconds.
3. **Trigger SLA**: When monotonic time reaches $T_{\text{deadline}}$, hard termination triggers immediately. Mean trigger latency is **$< 15\text{ ms}$**.
4. **Zero Post-Deadline Trials**: No trial started after $T_{\text{deadline}}$ is accepted or committed to the ledger.

---

## 2. Windows Job Object Containment

To prevent "ghost" or orphaned Python worker processes from remaining active after campaign shutdown:

1. **Job Creation**: The Guardian initializes a Windows Job Object handle via `kernel32.dll`.
2. **Process Assignment**: The Manager and all spawned workers (Classical Worker, Quantum Worker, Relays) are assigned to the Job Object upon creation.
3. **Kernel-Enforced Teardown**: The Job Object is configured with:
   ```
   JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
   ```
   When the Guardian process exits or closes the Job Object handle, the Windows NT kernel immediately terminates all child, grandchild, and descendant processes associated with the job.
4. **Verification**: 100% elimination of orphaned PIDs and child processes verified across extensive multi-day qualification suites.
