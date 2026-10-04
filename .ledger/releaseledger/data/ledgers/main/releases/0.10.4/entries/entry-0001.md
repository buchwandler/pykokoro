---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.10.4
kind: added
summary: Added a versioned request API compatibility contract and installed-wheel checks
status: accepted
audience: null
scopes: []
source_refs:
  - tl:task-0118
paths:
  - pykokoro/api_contract.py
  - pykokoro/synthesizer.py
  - tests/test_request_api_contract.py
  - .github/workflows/python-publish.yml
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 1
---

Integrations can inspect a stable request API declaration before synthesis runtime
initialization. Release CI verifies the public contract in built wheels and exercises
the request API on Python 3.14 without downloading models.
