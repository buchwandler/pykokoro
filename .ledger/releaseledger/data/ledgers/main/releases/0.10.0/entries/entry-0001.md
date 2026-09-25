---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 2
entry_id: entry-0001
release_version: 0.10.0
kind: changed
summary: Changed PyKokoro to render prepared requests as independent results
status: accepted
audience: null
scopes: []
source_refs:
  - git:7824a27c37ff13b422e7e11933b0cd62f3fca579
  - git:35a9c05a06d835b11641e0b4038bd05ad0a3a6de
  - git:f8ff25efa9e3ad870ebee1c5675f8fb2ffbe9170
  - git:b436299354407edcf98ac24850747726ca6dfbac
  - git:4c55705ff3fd6f477ecff6a04d638cd89d31b6b5
  - git:d6767c8f661c3ea33b995d44e92e167622ca3cd9
  - git:2741b94e51fe9b2e4a520ccbf088dd310e6861bd
  - git:f0833aae4a7b4a277199f7c003c149bd7ec80608
  - git:6550f14c4f5c1cacbe69c6d3b6aad7be06fdaf9f
  - git:219229ec6980cbedd0f141e44468f290a2e75476
paths:
  - README.md
  - docs/breaking-change-0.10.0.md
  - pykokoro/synthesizer.py
  - pykokoro/synthesis_types.py
issues: []
prs: []
sources: []
contributors: []
breaking: true
internal: false
order: 1
---

Removed KokoroPipeline, PipelineConfig, UtterPlan planning, AudioCompose composition,
and SSMD parsing from PyKokoro. Callers now provide prepared text, explicit language,
voice, and optional source-aligned context; they own document orchestration and
cross-request composition.
