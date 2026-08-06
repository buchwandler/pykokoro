---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.9.0
kind: changed
summary:
  German synthesis defaults to the verified Martin v1.2 fp32 profile with explicit
  legacy model compatibility
status: accepted
audience: null
scopes: []
source_refs:
  - tl:task-0017
paths:
  - pykokoro/model_profiles.py
  - pykokoro/pipeline.py
  - pykokoro/onnx_backend.py
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 1
---

German language runs select the GitHub model and martin voice automatically before
backend and G2P construction. Downloads are SHA-256 verified, and German normalization
covers dates, times, decimals, measurements, ordinals, durations, abbreviations, and
Euro amounts.
