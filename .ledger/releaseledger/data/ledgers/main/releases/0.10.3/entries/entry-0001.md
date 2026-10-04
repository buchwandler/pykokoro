---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.10.3
kind: added
summary:
  Added transcript-guided English voice enrollment with reusable, model-bound
  reference-voice conditioning
status: accepted
audience: null
scopes: []
source_refs:
  - git:4fa043229a3eacd73a2f185304391ad749822b61
paths:
  - docs/reference_voice.md
  - docs/api_reference.md
  - pykokoro/reference_audio.py
  - pykokoro/reference_voice.py
  - pykokoro/synthesizer.py
  - pykokoro/request_renderer.py
  - tests/test_reference_enrollment.py
  - tests/test_reference_synthesis.py
issues: []
prs: []
sources:
  - git:4fa043229a3eacd73a2f185304391ad749822b61
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
