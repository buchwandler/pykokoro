---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.10.13
kind: added
summary:
  Added a raw audio output policy that preserves inference sample count and sample
  values
status: accepted
audience: null
scopes: []
source_refs:
  - git:c9bc94692e3323d6cc862b1786319a0ac3e6a3c0
paths:
  - docs/advanced_features.md
  - docs/api_reference.md
  - pykokoro/__init__.py
  - pykokoro/audio_generator.py
  - pykokoro/onnx_backend.py
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
  - git:c9bc94692e3323d6cc862b1786319a0ac3e6a3c0
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
