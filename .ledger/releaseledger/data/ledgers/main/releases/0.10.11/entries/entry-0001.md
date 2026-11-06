---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.10.11
kind: changed
summary:
  Changed router inference to route by request language, cache by acoustic route, and
  validate voice-language pairs
status: accepted
audience: null
scopes: []
source_refs:
  - git:b189296d40b55da9be0b3d1bd931fcc8babf1db5
paths:
  - .github/workflows/python-publish.yml
  - .github/workflows/tests.yml
  - pykokoro/_onnxvoice.py
  - pykokoro/audio_generator.py
  - pykokoro/onnx_backend.py
  - pykokoro/request_renderer.py
  - pyproject.toml
  - tests/test_audio_generator_logging.py
  - tests/test_onnxvoice_boundary.py
  - tests/test_packaging_metadata.py
  - tests/test_request_renderer.py
issues: []
prs: []
sources:
  - git:b189296d40b55da9be0b3d1bd931fcc8babf1db5
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
