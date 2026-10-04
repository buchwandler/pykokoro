---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0003
release_version: 0.10.3
kind: changed
summary:
  Changed voice blending to support canonical `=` syntax, multi-voice linear
  interpolation, and row-wise SLERP fallback
status: accepted
audience: null
scopes: []
source_refs:
  - git:2a9a079cb9dea2f21d10e8ed87ca0a5adbb14b93
paths:
  - docs/advanced_features.md
  - examples/voice_blend.py
  - pykokoro/voice_manager.py
  - tests/test_voice_manager.py
issues: []
prs: []
sources:
  - git:2a9a079cb9dea2f21d10e8ed87ca0a5adbb14b93
contributors:
  - "@holgern"
breaking: false
internal: false
order: 3
---
