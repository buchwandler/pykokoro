---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: 0.10.3
kind: added
summary:
  Added transcript-free Inno voice enrollment that exports reusable Kokoro voice packs
status: accepted
audience: null
scopes: []
source_refs:
  - git:0b463faf6aa7099df9679d7472964de71215e58d
paths:
  - docs/reference_voice.md
  - docs/api_reference.md
  - pykokoro/synthesizer.py
  - pykokoro/voice_enrollment.py
  - pykokoro/voice_pack.py
  - pykokoro/model_registry.py
  - tests/test_inno_enrollment.py
  - tests/test_inno_synthesis.py
  - tests/test_voice_pack.py
issues: []
prs: []
sources:
  - git:0b463faf6aa7099df9679d7472964de71215e58d
contributors:
  - "@holgern"
breaking: false
internal: false
order: 2
---
