---
schema_version: 1
object_type: implementation_check
file_version: v2
check_id: check-0004
task_id: task-0001
implementation_run: run-0003
timestamp: "2026-07-26T10:33:06Z"
command:
  ruff check pykokoro/pipeline.py pykokoro/stages/g2p/kokorog2p.py
  tests/test_maintainer_review4_reproductions.py
argv:
  - ruff
  - check
  - pykokoro/pipeline.py
  - pykokoro/stages/g2p/kokorog2p.py
  - tests/test_maintainer_review4_reproductions.py
exit_code: 1
status: failed
category: lint
summary:
  Ran ruff check pykokoro/pipeline.py pykokoro/stages/g2p/kokorog2p.py
  tests/test_maintainer_review4_reproductions.py (exit 1)
stdout_ref: null
stderr_ref: null
combined_ref: null
---
