---
schema_version: 1
object_type: implementation_check
file_version: v2
check_id: check-0021
task_id: task-0001
implementation_run: run-0003
timestamp: "2026-07-26T10:54:08Z"
command:
  ruff check pykokoro/generation_config.py pykokoro/short_sentence_handler.py
  tests/test_generation_config.py tests/test_short_sentence_handler.py
argv:
  - ruff
  - check
  - pykokoro/generation_config.py
  - pykokoro/short_sentence_handler.py
  - tests/test_generation_config.py
  - tests/test_short_sentence_handler.py
exit_code: 1
status: failed
category: lint
summary:
  Ran ruff check pykokoro/generation_config.py pykokoro/short_sentence_handler.py
  tests/test_generation_config.py tests/test_short_sentence_handler.py (exit 1)
stdout_ref: null
stderr_ref: null
combined_ref: null
---
