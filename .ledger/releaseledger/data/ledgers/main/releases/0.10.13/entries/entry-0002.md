---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: 0.10.13
kind: added
summary:
  Added a silence-preserving output policy that disables generic trimming while
  retaining semantic processing
status: accepted
audience: null
scopes: []
source_refs:
  - git:79eff0b794a67847c28d652ab4987bdc2f9a2c42
paths:
  - docs/advanced_features.md
  - docs/api_reference.md
  - pykokoro/audio_generator.py
  - pykokoro/request_renderer.py
  - pykokoro/synthesis_config.py
  - tests/test_audio_generator_performance.py
  - tests/test_public_engine_api.py
  - tests/test_reference_synthesis.py
  - tests/test_request_renderer.py
  - tests/test_synthesis_config.py
  - tests/typecheck/public_api.py
issues: []
prs: []
sources:
  - git:79eff0b794a67847c28d652ab4987bdc2f9a2c42
contributors:
  - "@holgern"
breaking: false
internal: false
order: 2
---
