---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.8.7
kind: added
summary: Added model-derived word timings to audio results with clean-text offsets
  and sample positions
status: accepted
audience: null
scopes: []
source_refs:
- git:2a9768503be7bfc47ca06f7b7918958405781971
paths:
- README.md
- docs/advanced_features.md
- docs/api_reference.md
- docs/basic_usage.md
- examples/stream_with_word_timings.py
- pykokoro/__init__.py
- pykokoro/audio_generator.py
- pykokoro/pipeline.py
- pykokoro/short_sentence_cutters/energy_valley.py
- pykokoro/short_sentence_handler.py
- pykokoro/stages/audio_postprocessing/onnx.py
- pykokoro/stages/g2p/kokorog2p.py
- pykokoro/types.py
- tests/test_maintainer_review4_download_cache.py
- tests/test_word_timings.py
issues: []
prs: []
sources:
- git:2a9768503be7bfc47ca06f7b7918958405781971
contributors: []
breaking: false
internal: false
order: 1
---
