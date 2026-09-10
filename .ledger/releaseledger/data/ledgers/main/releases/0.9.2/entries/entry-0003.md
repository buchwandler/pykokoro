---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0003
release_version: 0.9.2
kind: added
summary: Added explicit G2P fallback and named-lexicon controls
status: accepted
audience: null
scopes: []
source_refs:
- git:93f5bac3032d8d6c06a1949694b59ee7c621069b
paths:
- .github/workflows/tests.yml
- README.md
- benchmarks/hard_cases/__main__.py
- benchmarks/polynorm_eval.py
- benchmarks/polynorm_phoneme.py
- docs/installation.md
- examples/backend_comparison.py
- examples/german_fallback_control.py
- pykokoro/stages/g2p/kokorog2p.py
- pykokoro/tokenizer.py
- pykokoro/types.py
- pyproject.toml
- requirements-test.txt
- tests/test_cache_keys.py
- tests/test_g2p_adapter_spacy_model.py
- tests/test_g2p_cache.py
- tests/test_lexphon_provisioning.py
- tests/test_maintainer_review4_download_cache.py
- tests/test_maintainer_review4_reproductions.py
- tests/test_packaging_metadata.py
- tests/test_tokenizer.py
- tests/test_word_timings.py
issues: []
prs: []
sources:
- git:93f5bac3032d8d6c06a1949694b59ee7c621069b
contributors: []
breaking: false
internal: false
order: 3
---
