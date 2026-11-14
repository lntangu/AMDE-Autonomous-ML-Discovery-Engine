# AMDE DataFrame & Preprocessing Audit

- **Total Checks**: 10
- **Passed**: 10
- **Failed**: 0

| Test ID | Description | Status | Detail |
| :--- | :--- | :--- | :--- |
| `E-CAT-01` | Category dtype with missing values | **PASS** | Output shape: (4, 4) |
| `E-CAT-02` | Ordered categorical dtype with missing values | **PASS** | Output shape: (4, 5) |
| `E-CAT-03` | Numerical-looking category labels | **PASS** | Output shape: (4, 4) |
| `E-CAT-04` | Unseen categories during inference | **PASS** | Output shape: (3, 3) |
| `E-CAT-05` | Category sentinel collision ('__missing__') | **PASS** | Output shape: (3, 3) |
| `E-CAT-06` | Entirely missing categorical column | **PASS** | Output shape: (3, 2) |
| `E-NUM-01` | +/- Infinity in numerical columns | **PASS** | Output shape: (4, 5) |
| `E-NUM-02` | All-missing numerical column | **PASS** | Output shape: (3, 4) |
| `E-NUM-03` | Pandas nullable Int64 and Float64 dtypes | **PASS** | Output shape: (3, 5) |
| `E-MUT-01` | Caller DataFrame immutability guarantee | **PASS** | Original DataFrame untouched. |
