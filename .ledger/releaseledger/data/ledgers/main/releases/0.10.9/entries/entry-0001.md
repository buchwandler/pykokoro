---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 3
entry_id: entry-0001
release_version: 0.10.9
kind: fixed
summary:
  Fixed character say-as projection to preserve spaced spelling and attach detached
  possessive suffixes
status: accepted
audience: null
scopes: []
source_refs:
  - tl:task-0124
  - tl:task-0125
paths:
  - pykokoro/frontend_projection.py
  - tests/test_frontend_projection.py
  - tests/test_prepared_g2p_adapter.py
  - tests/test_request_renderer.py
  - tests/test_reference_frontend_parity.py
  - tests/test_synthesis_identity.py
  - docs/advanced_features.md
  - docs/api_reference.md
  - docs/changelog.md
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 1
---

Character say-as now keeps already-separated characters as ordinary spaced text,
separates compact character sequences with spaces, and joins a detached adjacent 's/’s
suffix in the private frontend projection while preserving canonical request, alignment,
and timing coordinates.
