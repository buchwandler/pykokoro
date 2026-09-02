---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0003
release_version: 0.9.0
kind: changed
summary: Changed the locked dependency set to the v0.9 companion releases
status: accepted
audience: null
scopes: []
source_refs:
  - git:9f874b210ddeb26a7ab593e36a2fc1f55be6c277
paths:
  - uv.lock
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 3
---

The lockfile now resolves kokorog2p 0.9.0, phrasplit 0.3.7, Spokenform 0.3.6, and SSMD
0.8.6 for the v0.9 integration contract.
