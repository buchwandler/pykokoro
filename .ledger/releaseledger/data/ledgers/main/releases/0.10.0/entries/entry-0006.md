---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0006
release_version: 0.10.0
kind: quality
summary: Improved regression and integration coverage for the engine boundary
status: accepted
audience: null
scopes: []
source_refs:
  - git:63346451da51c3f4a0585e9a40a1830c8760613f
  - git:6792b81f979cdda044eb27d92b4de754a2a50ffe
  - git:b94802a3c5884050f07621359f4eda82fd94931c
  - git:dd8a26e0956551d9250167b3c30db199cc0b61a5
paths:
  - tests/test_onnxvoice_boundary.py
  - tests/test_word_timings.py
  - tests/test_hard_cases_frontend.py
  - benchmarks/hard_cases/frontend.py
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 6
---

Updated benchmark, runtime-boundary, timing, and model-profile checks during the
request-centric refactor, including test portability corrections.
