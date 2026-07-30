---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 2
entry_id: entry-0002
release_version: 0.7.3
kind: fixed
summary: Fixed SSMD emphasis none handling and deduplicated warnings by source segment
status: accepted
audience: null
scopes: []
source_refs: []
paths:
  - pykokoro/emphasis.py
  - pykokoro/pipeline.py
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 2
---

Effectful emphasis in error mode is rejected before model inference; explicit prosody
remains authoritative.
