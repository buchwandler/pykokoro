---
schema_version: 1
object_type: implementation_check
file_version: v2
check_id: check-0008
task_id: task-0001
implementation_run: run-0003
timestamp: "2026-07-26T10:43:48Z"
command:
  ruff check pykokoro/onnx_backend.py pykokoro/runtime/cache.py
  pykokoro/stages/g2p/kokorog2p.py tests/test_maintainer_review4_download_cache.py
argv:
  - ruff
  - check
  - pykokoro/onnx_backend.py
  - pykokoro/runtime/cache.py
  - pykokoro/stages/g2p/kokorog2p.py
  - tests/test_maintainer_review4_download_cache.py
exit_code: 1
status: failed
category: lint
summary:
  "Ran ruff check pykokoro/onnx_backend.py pykokoro/runtime/cache.py
  pykokoro/stages/g2p/kokorog2p.py tests/test_maintainer_review4_download_cache.py (exit
  1) output: @tasks/task-0001/artifacts/run-0003-command-0002.log"
stdout_ref: null
stderr_ref: null
combined_ref: null
---
