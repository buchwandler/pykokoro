---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 2
entry_id: entry-0001
release_version: 0.7.3
kind: changed
summary:
  Changed SSMD emphasis to default to metadata-preserving plain speech with opt-in
  volume-only approximation
status: accepted
audience: null
scopes: []
source_refs:
  - tl:task-0007
paths:
  - pykokoro/ssmd_config.py
  - pykokoro/emphasis.py
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 1
---

Approximate mode maps strong to +6dB, moderate to +3dB, and reduced to -3dB without
requiring optional prosody backends.
