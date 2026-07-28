---
schema_version: 1
object_type: implementation_check
file_version: v2
check_id: check-0016
task_id: task-0001
implementation_run: run-0003
timestamp: "2026-07-26T10:47:08Z"
command:
  sh -c 'tar -tzf dist/*.tar.gz | rg
  "(^|/)LICENSE$|pykokoro/_version.py|pykokoro/compat"'
argv:
  - sh
  - -c
  - tar -tzf dist/*.tar.gz | rg "(^|/)LICENSE$|pykokoro/_version.py|pykokoro/compat"
exit_code: 0
status: passed
category: other
summary:
  Ran sh -c 'tar -tzf dist/*.tar.gz | rg
  "(^|/)LICENSE$|pykokoro/_version.py|pykokoro/compat"' (exit 0)
stdout_ref: null
stderr_ref: null
combined_ref: null
---
