---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.10.10
kind: fixed
summary:
  Fixed character say-as projection to preserve spaced spelling and attach detached
  possessive suffixes
status: accepted
audience: null
scopes: []
source_refs:
  - git:263a3f8943631e74e91e6b5dd45a087bcdc6eadc
paths:
  - .ledger/releaseledger/data/ledgers/main/events/events.jsonl
  - .ledger/releaseledger/data/ledgers/main/releases/0.10.9/entries/entry-0001.md
  - docs/advanced_features.md
  - docs/api_reference.md
  - docs/changelog.md
  - pykokoro/frontend_projection.py
  - tests/test_frontend_projection.py
  - tests/test_prepared_g2p_adapter.py
  - tests/test_reference_frontend_parity.py
  - tests/test_request_renderer.py
  - tests/test_synthesis_identity.py
issues: []
prs: []
sources:
  - git:263a3f8943631e74e91e6b5dd45a087bcdc6eadc
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---

Updated the private character say-as projection to preserve spaced spelling, separate
compact character sequences, and attach an immediately following detached possessive
suffix while keeping canonical request text, offsets, alignments, and timings unchanged.
