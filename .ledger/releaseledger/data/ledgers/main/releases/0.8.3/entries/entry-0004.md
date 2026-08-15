---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0004
release_version: 0.8.3
kind: added
summary:
  Added configurable SSMD emphasis gain scaling for gain-only automatic emphasis
  approximation
status: accepted
audience: null
scopes: []
source_refs:
  - tl:task-0020
paths:
  - pykokoro/ssmd_config.py
  - pykokoro/emphasis.py
  - tests/test_emphasis_policy.py
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 4
---

SSMDRenderConfig now exposes emphasis_gain_scale from 0.0 through 2.0; the default 1.0
preserves existing reduced, moderate, and strong gain behavior while explicit volume
metadata retains precedence.
