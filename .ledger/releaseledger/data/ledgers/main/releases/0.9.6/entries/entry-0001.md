---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.9.6
kind: changed
summary:
  Changed pipeline-owned stages to follow model variant changes while preserving custom
  stages after replacement failures
status: accepted
audience: null
scopes: []
source_refs:
  - git:21bcf55906ab1ca7b98dddf096d8ff009014cd30
paths:
  - pykokoro/pipeline.py
  - tests/test_lexicon_hot_swap.py
  - tests/test_pipeline_lifecycle.py
issues: []
prs: []
sources:
  - git:21bcf55906ab1ca7b98dddf096d8ff009014cd30
contributors: []
breaking: false
internal: false
order: 1
---
