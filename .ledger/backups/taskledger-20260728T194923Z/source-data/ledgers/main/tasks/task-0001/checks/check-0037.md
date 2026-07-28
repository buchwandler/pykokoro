---
schema_version: 1
object_type: implementation_check
file_version: v2
check_id: check-0037
task_id: task-0001
implementation_run: run-0003
timestamp: "2026-07-26T11:07:58Z"
command: python -m pytest -q -m 'not slow'
argv:
  - python
  - -m
  - pytest
  - -q
  - -m
  - not slow
exit_code: 0
status: passed
category: test
summary:
  "Ran python -m pytest -q -m 'not slow' (exit 0) output:
  @tasks/task-0001/artifacts/run-0003-command-0008.log"
stdout_ref: null
stderr_ref: null
combined_ref: null
---
