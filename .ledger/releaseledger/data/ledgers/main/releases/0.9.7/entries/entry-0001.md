---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.9.7
kind: added
summary:
  Added automatic high-confidence clausal-comma pause detection for dependency-aware
  English segmentation
status: accepted
audience: null
scopes: []
source_refs:
  - git:9d36442a35f40d932f8999be8fe33c5e20ae8a66
paths:
  - .github/workflows/tests.yml
  - README.md
  - docs/basic_usage.md
  - docs/installation.md
  - docs/pipeline_stages.md
  - pykokoro/stages/segmentation/phrasplit.py
  - pyproject.toml
  - requirements-test.txt
  - tests/test_clausal_comma_pauses.py
  - tests/test_packaging_metadata.py
issues: []
prs: []
sources:
  - git:9d36442a35f40d932f8999be8fe33c5e20ae8a66
contributors: []
breaking: false
internal: false
order: 1
---
