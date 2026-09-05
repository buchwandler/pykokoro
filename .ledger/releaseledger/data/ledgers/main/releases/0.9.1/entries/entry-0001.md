---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.9.1
kind: added
summary:
  Added public resolve_pipeline_config for metadata-only automatic model, source,
  quality, and voice selection
status: accepted
audience: null
scopes: []
source_refs:
  - tl:task-0053
paths:
  - pykokoro/pipeline_config.py
  - pykokoro/__init__.py
  - tests/test_public_pipeline_resolution.py
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 1
---

Supports preflight, orchestration, dry-run planning, and configuration inspection
without constructing the ONNX-backed synthesis runtime or loading model and voice assets
