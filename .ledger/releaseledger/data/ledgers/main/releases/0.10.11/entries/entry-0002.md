---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: 0.10.11
kind: added
summary:
  Added English/German AKinVox multispeaker synthesis with 16 language-specific voices
  and contextual long-text chunks
status: accepted
audience: null
scopes: []
source_refs:
  - git:8798dc5970a39653c6647cd7a5b2c2cd00339d06
paths:
  - .github/workflows/tests.yml
  - docs/installation.md
  - docs/languages.md
  - pykokoro/_onnxvoice.py
  - pykokoro/audio_generator.py
  - pykokoro/discovery.py
  - pykokoro/frontend_contracts.py
  - pykokoro/model_profiles.py
  - pykokoro/onnx_backend.py
  - pykokoro/request_renderer.py
  - pykokoro/runtime_protocol.py
  - pykokoro/synthesis_config.py
  - tests/test_akinvox_multispeaker.py
  - tests/test_audio_generator_logging.py
  - tests/test_audio_generator_performance.py
  - tests/test_long_text_splitting.py
  - tests/test_model_discovery.py
  - tests/test_onnxvoice_boundary.py
  - tests/test_packaging_metadata.py
  - tests/test_release_profiles.py
  - tests/test_request_renderer.py
issues: []
prs: []
sources:
  - git:8798dc5970a39653c6647cd7a5b2c2cd00339d06
contributors:
  - "@holgern"
breaking: false
internal: false
order: 2
---
