---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.10.9
kind: fixed
summary:
  Fixed character say-as projection to use space-free hyphens and a terminal hyphen
status: accepted
audience: null
scopes: []
source_refs:
  - tl:task-0124
paths:
  - pykokoro/frontend_projection.py
  - tests/test_frontend_projection.py
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 1
---

Marked letters and digits now map through the full projected span while public request
text, alignment, and word-timing coordinates remain canonical.
