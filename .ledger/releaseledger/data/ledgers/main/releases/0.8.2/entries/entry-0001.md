---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.8.2
kind: changed
summary:
  German synthesis defaults to the verified Martin v1.2 fp32 profile with automatic
  language-aware model and voice resolution
status: accepted
audience: null
scopes: []
source_refs:
  - git:97f9ddb6808f2afb9c2eb1fb199249f8a4163de4
  - git:c9e6a0868fd24540984b82bcc3268e1ee3e97dec
paths:
  - pykokoro/model_profiles.py
  - pykokoro/pipeline_config.py
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
