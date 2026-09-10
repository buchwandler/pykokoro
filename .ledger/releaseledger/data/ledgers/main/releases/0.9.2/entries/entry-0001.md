---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.9.2
kind: added
summary: Added automatic language-aware routing for mixed-language documents
status: accepted
audience: null
scopes: []
source_refs:
  - git:e550648b7cf8ca83bd8881e0ff6da61434cf226c
paths:
  - .github/workflows/tests.yml
  - README.md
  - docs/basic_usage.md
  - docs/languages.md
  - examples/mixed_language_auto.py
  - pykokoro/__init__.py
  - pykokoro/language_detection.py
  - pykokoro/pipeline.py
  - pykokoro/pipeline_config.py
  - pykokoro/runtime/cache.py
  - pykokoro/runtime/language_plan.py
  - pykokoro/stages/doc_parsers/ssmd.py
  - pykokoro/stages/g2p/kokorog2p.py
  - pykokoro/stages/segmentation/phrasplit.py
  - pyproject.toml
  - requirements-test.txt
  - tests/test_g2p_cache.py
  - tests/test_language_detection_config.py
  - tests/test_language_plan.py
  - tests/test_maintainer_review4_download_cache.py
  - tests/test_packaging_metadata.py
issues: []
prs: []
sources:
  - git:e550648b7cf8ca83bd8881e0ff6da61434cf226c
contributors: []
breaking: false
internal: false
order: 1
---
