---
schema_version: 1
object_type: implementation_check
file_version: v2
check_id: check-0029
task_id: task-0001
implementation_run: run-0003
timestamp: "2026-07-26T11:02:44Z"
command: ruff format --check .
argv:
  - ruff
  - format
  - --check
  - .
exit_code: 1
status: failed
category: format
summary:
  "Ran ruff format --check . (exit 1) output:
  @tasks/task-0001/artifacts/run-0003-command-0006.log"
stdout_ref: null
stderr_ref: null
combined_ref: null
---
