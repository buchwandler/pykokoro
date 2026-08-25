---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: 0.8.7
kind: fixed
summary:
  Fixed timing alignment for named duration outputs and final-waveform transformations
status: accepted
audience: null
scopes: []
source_refs:
  - git:02d48e8018e2f3ad520bfed2d43c951f84c0a0d4
paths:
  - README.md
  - docs/advanced_features.md
  - docs/api_reference.md
  - docs/basic_usage.md
  - examples/stream_with_word_timings.py
  - pykokoro/audio_generator.py
  - pykokoro/pipeline.py
  - pykokoro/short_sentence_handler.py
  - pykokoro/stages/audio_postprocessing/onnx.py
  - pykokoro/stages/g2p/kokorog2p.py
  - pykokoro/types.py
  - tests/test_audio_splitting.py
  - tests/test_public_api_compatibility.py
  - tests/test_word_timings.py
issues: []
prs: []
sources:
  - git:02d48e8018e2f3ad520bfed2d43c951f84c0a0d4
contributors: []
breaking: false
internal: false
order: 2
---
