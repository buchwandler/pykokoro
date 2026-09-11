---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.9.4
kind: changed
summary:
  Changed release manifest resolution to select model assets by quality and validate
  package dependency versions
status: accepted
audience: null
scopes: []
source_refs:
  - git:5e50a8ac6076fb36c6a3552ad700dfc00414b7cf
paths:
  - .github/workflows/python-publish.yml
  - pykokoro/pipeline_config.py
  - tests/test_packaging_metadata.py
  - tests/test_release_profiles.py
issues: []
prs: []
sources:
  - git:5e50a8ac6076fb36c6a3552ad700dfc00414b7cf
contributors: []
breaking: false
internal: false
order: 1
---
