---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0003
release_version: 0.10.0
kind: changed
summary: Changed model and ONNX session management to use OnnxVoice
status: accepted
audience: null
scopes: []
source_refs:
  - git:825a49a7b6b90d7308eb7d9a672dbcc745f576be
  - git:0e22b45622154ad005302d8daa8df38c2df4d2c0
  - git:9a496a82e9972cdc689df604abcf46d0cca74907
  - git:36606f5bf85993924952da71b48c0536fb9388b1
paths:
  - pykokoro/_onnxvoice.py
  - pykokoro/onnx_backend.py
  - pykokoro/audio_generator.py
  - pykokoro/prepared_g2p.py
  - pykokoro/discovery.py
  - pykokoro/lexicon_discovery.py
issues: []
prs: []
sources:
  - git:7824a27c37ff13b422e7e11933b0cd62f3fca579
  - tl:task-0112
contributors: []
breaking: false
internal: false
order: 3
---

OnnxVoice owns managed model installation, resolution, local opening, provider
normalization, and session lifecycle. KokoroG2P owns prepared-text phonemization and
AudioSig supplies DSP primitives. PyKokoro retains request rendering, model and voice
profile selection, and request-local timing metadata. Metadata-only discover_models()
and discover_lexicons() inspection remains available without loading an inference
session.
