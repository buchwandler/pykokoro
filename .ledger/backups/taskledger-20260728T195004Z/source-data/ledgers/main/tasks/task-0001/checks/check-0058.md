---
schema_version: 1
object_type: implementation_check
file_version: v2
check_id: check-0058
task_id: task-0001
implementation_run: run-0005
timestamp: "2026-07-26T11:29:38Z"
command: python -m pytest -q -m 'not slow'
argv:
  - python
  - -m
  - pytest
  - -q
  - -m
  - not slow
exit_code: 1
status: failed
category: test
summary:
  "Ran python -m pytest -q -m 'not slow' (exit 1) output:
  @tasks/task-0001/artifacts/run-0005-command-0002.log"
stdout_ref: null
stderr_ref: null
combined_ref: null
---
