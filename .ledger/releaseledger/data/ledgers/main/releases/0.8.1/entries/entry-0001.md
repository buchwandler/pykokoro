---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.8.1
kind: added
summary:
  Added paragraph-unit preparation and streaming for bounded, resumable document
  rendering
status: accepted
audience: null
scopes: []
source_refs:
  - tl:task-0013
paths:
  - pykokoro/pipeline.py
  - pykokoro/types.py
  - tests/test_pipeline_unit_streaming.py
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 1
---

Documents are parsed and phonemized once, then rendered as ordered paragraph units with
deterministic pauses and markers, explicit audio release, and skip support.
