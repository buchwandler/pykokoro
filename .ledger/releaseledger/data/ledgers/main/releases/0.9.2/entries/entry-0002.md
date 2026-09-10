---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: 0.9.2
kind: changed
summary: Improved mixed-language frontend selection and caching
status: accepted
audience: null
scopes: []
source_refs:
  - git:4d0f5feba14eb650a7f6c57101d5fb67b2fa0505
paths:
  - .github/workflows/tests.yml
  - examples/mixed_language_auto.py
  - pykokoro/pipeline.py
  - pykokoro/stages/g2p/kokorog2p.py
  - pyproject.toml
  - requirements-test.txt
  - tests/test_g2p_cache.py
  - tests/test_g2p_language_routing.py
  - tests/test_g2p_spokenform.py
  - tests/test_language_detection_config.py
  - tests/test_maintainer_review4_download_cache.py
  - tests/test_packaging_metadata.py
issues: []
prs: []
sources:
  - git:4d0f5feba14eb650a7f6c57101d5fb67b2fa0505
contributors: []
breaking: false
internal: false
order: 2
---
