---
schema_version: 1
object_type: implementation_check
file_version: v2
check_id: check-0012
task_id: task-0001
implementation_run: run-0003
timestamp: "2026-07-26T10:46:42Z"
command:
  ruff check pyproject.toml setup.py pykokoro/__init__.py
  tests/test_packaging_metadata.py
argv:
  - ruff
  - check
  - pyproject.toml
  - setup.py
  - pykokoro/__init__.py
  - tests/test_packaging_metadata.py
exit_code: 0
status: passed
category: lint
summary:
  Ran ruff check pyproject.toml setup.py pykokoro/__init__.py
  tests/test_packaging_metadata.py (exit 0)
stdout_ref: null
stderr_ref: null
combined_ref: null
---
