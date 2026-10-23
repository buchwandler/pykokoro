---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 4
entry_id: entry-0002
release_version: 0.10.6
kind: fixed
summary:
  Fixed AkinVox reference synthesis by omitting the lower-level seed=None when
  random_seed is unset
status: accepted
audience: null
scopes: []
source_refs:
  - tl:task-0121
paths:
  - pykokoro/_onnxvoice.py
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 2
---

PyKokoro now omits the lower-level seed argument when GenerationConfig.random_seed is
None instead of forwarding seed=None, which previously broke AkinVox reference
synthesis. Explicit integer seeds continue to be forwarded unchanged.
