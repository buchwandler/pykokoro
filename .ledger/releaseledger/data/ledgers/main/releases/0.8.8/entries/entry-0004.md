---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0004
release_version: 0.8.8
kind: changed
summary:
  Changed the pipeline to explicit Spokenform, Phrasplit, and prepared G2P stages with
  remapped offsets
status: accepted
audience: null
scopes: []
source_refs:
  - git:45277116465ace6c4833c3d1e91819a64d529f71
paths:
  - README.md
  - pykokoro/pipeline.py
  - pykokoro/stages/text_preparation/spokenform.py
  - pykokoro/stages/segmentation/phrasplit.py
  - pykokoro/stages/g2p/kokorog2p.py
  - tests/test_pipeline_refactor.py
  - tests/test_splitter_offsets.py
  - tests/test_g2p_spokenform.py
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 4
---
