# AMDE CI/CD & GitHub Actions Security Audit Report

| Workflow | Security Check | Status | Detail |
| :--- | :--- | :--- | :--- |
| `ci.yml` | Explicit least-privilege permissions block | **PASS** | Workflow explicitly defines permissions block |
| `ci.yml` | Absence of unsafe pull_request_target trigger | **PASS** | pull_request_target not used |
| `ci.yml` | Action version pinning | **PASS** | All actions pinned with major tags or commit hashes |
| `publish-pypi.yml` | Explicit least-privilege permissions block | **PASS** | Workflow explicitly defines permissions block |
| `publish-pypi.yml` | Absence of unsafe pull_request_target trigger | **PASS** | pull_request_target not used |
| `publish-pypi.yml` | Action version pinning | ❌ **WARN** | Unpinned actions: [('pypa/gh-action-pypi-publish', 'release/v1')] |
| `publish-pypi.yml` | OIDC id-token scoped to named environment | **PASS** | id-token: write is correctly scoped to deployment environment |
