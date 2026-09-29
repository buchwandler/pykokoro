---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.10.1
kind: fixed
summary:
  Fixed forwarding of SynthesisConfig.cache_dir to OnnxVoice for managed model and voice
  assets
status: accepted
audience: null
scopes: []
source_refs:
  - git:9d11eb7e34247f7733f916a77b2c0ed7a0c6f254
paths:
  - pykokoro/synthesis_config.py
  - pykokoro/request_renderer.py
  - pykokoro/onnx_backend.py
  - tests/test_onnxvoice_boundary.py
issues: []
prs: []
sources:
  - git:9d11eb7e34247f7733f916a77b2c0ed7a0c6f254
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
