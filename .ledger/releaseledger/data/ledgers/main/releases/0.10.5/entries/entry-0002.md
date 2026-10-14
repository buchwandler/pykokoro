---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: 0.10.5
kind: added
summary:
  Added request measurement for frontend token usage without rendering or reference
  enrollment
status: accepted
audience: null
scopes: []
source_refs:
  - git:163c114f253dce5bc7258c193729a65441efb75c
paths:
  - docs/api_reference.md
  - docs/reference_voice.md
  - pykokoro/__init__.py
  - pykokoro/api_contract.py
  - pykokoro/synthesis_types.py
  - pykokoro/synthesizer.py
  - tests/test_reference_frontend_parity.py
  - tests/test_request_api_contract.py
  - tests/test_synthesis_types.py
  - tests/test_synthesizer_requests.py
  - tests/typecheck/public_api.py
issues: []
prs: []
sources:
  - git:163c114f253dce5bc7258c193729a65441efb75c
contributors:
  - "@holgern"
breaking: false
internal: false
order: 2
---
