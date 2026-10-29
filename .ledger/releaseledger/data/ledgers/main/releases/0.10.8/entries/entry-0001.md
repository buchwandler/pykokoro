---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.10.8
kind: fixed
summary:
  Fixed character-level say-as projections to preserve outer punctuation and canonical
  timing and offset mappings
status: accepted
audience: null
scopes: []
source_refs:
  - git:ff1fd8f8f16a781c8919b59b8612ed4b8701f0e5
paths:
  - pykokoro/frontend_projection.py
  - tests/test_frontend_projection.py
  - tests/test_prepared_g2p_adapter.py
  - tests/test_request_renderer.py
issues: []
prs: []
sources:
  - git:ff1fd8f8f16a781c8919b59b8612ed4b8701f0e5
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
