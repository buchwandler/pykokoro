---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.10.2
kind: changed
summary:
  Raised PyKokoro's minimum supported OnnxVoice version to 0.2.0 while keeping its
  model-local voice API unchanged
status: accepted
audience: null
scopes: []
source_refs:
  - tl:task-0114
paths:
  - pyproject.toml
  - .github/workflows/tests.yml
  - .github/workflows/python-publish.yml
  - tests/test_packaging_metadata.py
  - docs/breaking-change-0.10.0.md
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 1
---
