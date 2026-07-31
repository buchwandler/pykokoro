---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0003
release_version: 0.7.4
kind: changed
summary:
  Added explicit SSMD emphasis policies and diagnostics for plain, approximate, warning,
  and error handling
status: accepted
audience: null
scopes: []
source_refs:
  - git:a12ff7a445851791d9895b4667eef9891a14bb2d
paths:
  - .codecrate.toml
  - .github/workflows/python-publish.yml
  - .github/workflows/tests.yml
  - .ledger/releaseledger/data/ledgers/main/events/events.jsonl
  - .ledger/releaseledger/data/ledgers/main/releases/0.7.3/entries/entry-0001.md
  - .ledger/releaseledger/data/ledgers/main/releases/0.7.3/entries/entry-0002.md
  - .ledger/releaseledger/data/ledgers/main/releases/0.7.3/entries/entry-0003.md
  - .ledger/releaseledger/data/ledgers/main/releases/0.7.3/release.md
  - README.md
  - docs/api_reference.md
  - docs/basic_usage.md
  - docs/changelog.md
  - docs/pipeline_stages.md
  - docs/quickstart.md
  - examples/ssmd_demo.py
  - pykokoro/constants.py
  - pykokoro/emphasis.py
  - pykokoro/pipeline.py
  - pykokoro/prosody.py
  - pykokoro/py.typed
  - pykokoro/ssmd_config.py
  - pykokoro/stages/audio_postprocessing/onnx.py
  - pyproject.toml
  - tests/test_emphasis_plain_and_prosody_probe.py
  - tests/test_emphasis_policy.py
  - tests/test_prosody_backends.py
issues: []
prs: []
sources:
  - git:a12ff7a445851791d9895b4667eef9891a14bb2d
contributors: []
breaking: false
internal: false
order: 3
---
