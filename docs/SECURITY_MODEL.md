# AMDE Security & Privacy Architecture

This document describes the security boundaries, data privacy guarantees, and process isolation model of AMDE.

---

## 1. Security Architecture

1. **Local-First Execution**: AMDE operates entirely within the local user filesystem. It does not phone home, transmit telemetry to external servers, or require cloud connectivity.
2. **No Arbitrary Code Execution**: Level D symbolic DSL expressions are evaluated via a safe, restricted AST interpreter (`src/amde/discovery/dsl.py`). No `eval()` or `exec()` calls are used to interpret discovered update rules.
3. **Safe Storage**: SQLite trial databases and JSON plan files use parameterized SQL queries to prevent injection vulnerabilities.

---

## 2. Process Isolation & Resource Containment

* **Windows Job Objects**: On Windows systems, all child processes are bound to an operating system Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, ensuring forced termination when the parent guardian exits.
* **Thermal & Memory Throttling**: The Guardian continuously samples GPU temperature and host RAM. If thresholds are exceeded (e.g. GPU $> 80^\circ\text{C}$ or free RAM $< 2\text{ GB}$), worker processes are paused until hardware recovers.
