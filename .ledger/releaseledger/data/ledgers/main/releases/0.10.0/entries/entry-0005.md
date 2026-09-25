---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0005
release_version: 0.10.0
kind: changed
summary: Changed and validated the supported companion-library minimum versions
status: accepted
audience: null
scopes: []
source_refs:
  - tl:task-0112
paths:
  - pyproject.toml
  - .github/workflows/tests.yml
  - .github/workflows/python-publish.yml
  - tests/test_packaging_metadata.py
issues: []
prs: []
sources:
  - git:7a9ce60fd9d44a84bc9f58dfb86ed5db7f00a231
  - git:825a49a7b6b90d7308eb7d9a672dbcc745f576be
  - git:7824a27c37ff13b422e7e11933b0cd62f3fca579
contributors: []
breaking: false
internal: false
order: 5
---

The supported floors are KokoroG2P 0.9.9, Lexphon 0.2.3, PhraseSplit 0.3.9, AudioSig
0.1.4, and OnnxVoice 0.1.7. Lower-bound CI pins OnnxVoice's CPU extra and exercises its
model, provider, voice, and timing integration contracts; package-artifact checks
validate the dependency floors in built metadata.
