---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 2
entry_id: entry-0006
release_version: 0.9.8
kind: added
summary:
  Added configurable automatic pauses at detected parenthetical boundaries with G2P
  propagation and example coverage
status: accepted
audience: null
scopes: []
source_refs:
  - git:4df2043695e889d56fd4b0ddeb21f15f7c9f65f8
paths:
  - .github/workflows/tests.yml
  - .ledger/releaseledger/data/ledgers/main/events/events.jsonl
  - .ledger/releaseledger/data/ledgers/main/releases/0.9.7/entries/entry-0003.md
  - .ledger/releaseledger/data/ledgers/main/releases/0.9.7/release.md
  - README.md
  - docs/basic_usage.md
  - docs/changelog.md
  - docs/installation.md
  - docs/pipeline_stages.md
  - examples/english_parenthetical_pause.py
  - examples/run_all.py
  - pykokoro/generation_config.py
  - pykokoro/stages/segmentation/phrasplit.py
  - pyproject.toml
  - requirements-test.txt
  - tests/test_example_runner.py
  - tests/test_g2p_pause_propagation.py
  - tests/test_generation_config.py
  - tests/test_packaging_metadata.py
  - tests/test_parenthetical_pauses.py
issues: []
prs: []
sources:
  - git:4df2043695e889d56fd4b0ddeb21f15f7c9f65f8
contributors: []
breaking: false
internal: false
order: 6
---
