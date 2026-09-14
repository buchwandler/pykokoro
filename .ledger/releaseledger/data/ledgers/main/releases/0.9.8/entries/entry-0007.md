---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 2
entry_id: entry-0007
release_version: 0.9.8
kind: fixed
summary:
  Fixed Chinese v1.1 vocabulary handling across tokenization and ONNX audio input
  generation
status: accepted
audience: null
scopes: []
source_refs:
  - git:50802ee9a70cd3eea7f2609858255dd39d3adf92
paths:
  - pykokoro/audio_generator.py
  - pykokoro/tokenizer.py
  - tests/test_all_voices_example.py
  - tests/test_audio_input_contract.py
  - tests/test_audio_splitting.py
  - tests/test_pipeline_phoneme_consistency.py
  - tests/test_tokenizer.py
issues: []
prs: []
sources:
  - git:50802ee9a70cd3eea7f2609858255dd39d3adf92
contributors: []
breaking: false
internal: false
order: 7
---
