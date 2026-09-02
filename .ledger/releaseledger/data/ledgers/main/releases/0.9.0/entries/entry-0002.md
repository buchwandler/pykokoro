---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: 0.9.0
kind: changed
summary:
  Changed to require explicit document languages and SSMD spans for mixed-language
  synthesis
status: accepted
audience: null
scopes: []
source_refs:
  - git:65ed9eec616468ccf37dd1a2138c6b464baf7d98
paths:
  - README.md
  - docs/basic_usage.md
  - docs/installation.md
  - docs/languages.md
  - examples/mixed_language.py
  - pykokoro/generation_config.py
  - pykokoro/pipeline.py
  - pykokoro/pipeline_config.py
  - pykokoro/runtime/language_plan.py
  - pykokoro/runtime/linguistics.py
  - pykokoro/stages/g2p/kokorog2p.py
  - pykokoro/stages/segmentation/phrasplit.py
  - pykokoro/stages/text_preparation/spokenform.py
  - pykokoro/tokenizer.py
  - pyproject.toml
  - tests/test_language_plan.py
  - tests/test_pipeline_refactor.py
issues: []
prs: []
sources: []
contributors: []
breaking: true
internal: false
order: 2
---

The pipeline now plans validated language runs before preparation, shares local
linguistic analysis across preparation, segmentation, and G2P, and preserves
prepared-text offsets. Generic written-to-spoken and say-as realization is owned by
Spokenform.
