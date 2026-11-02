# AMDE Troubleshooting Guide

Solutions to common runtime issues encountered when operating AMDE campaigns.

---

## 1. Process & Lock Issues

### Problem: `LockError: another instance is active`
* **Cause**: A previous campaign crashed or was terminated improperly without releasing `.amde.lock`.
* **Fix**: Verify no running python processes exist (`Get-Process python` on Windows or `ps aux | grep python` on Linux). If no process is running, delete the lock file:
  ```bash
  rm runs/.amde.lock
  ```

---

## 2. Windows Job Object Permissions

### Problem: `WindowsError: Access is denied when assigning process to Job Object`
* **Cause**: Process was spawned under a different user session or elevated security token.
* **Fix**: Ensure the Guardian, Manager, and CLI commands are launched from the same standard user terminal (do not mix Administrator and non-elevated shells).

---

## 3. SQLite Database Locked

### Problem: `sqlite3.OperationalError: database is locked`
* **Cause**: Multiple processes attempting concurrent writes without WAL mode enabled.
* **Fix**: AMDE enables WAL mode by default. Ensure external SQLite viewing tools (e.g. DB Browser) open databases in **read-only** mode while campaigns are active.
