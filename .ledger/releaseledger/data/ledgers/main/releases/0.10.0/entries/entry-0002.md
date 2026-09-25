---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0002
release_version: 0.10.0
kind: added
summary:
  Added opt-in token-safe splitting for oversized requests with one rendered result
status: accepted
audience: null
scopes: []
source_refs:
  - git:7a9ce60fd9d44a84bc9f58dfb86ed5db7f00a231
paths:
  - pykokoro/request_renderer.py
  - pykokoro/synthesis_config.py
  - tests/test_long_text_splitting.py
issues: []
prs: []
sources:
  - tl:task-0112
contributors: []
breaking: false
internal: false
order: 2
---

The default long_text_split="none" path does not import PhraseSplit and raises
SynthesisInputTooLongError when the model limit is exceeded. With
long_text_split="sentence", PhraseSplit loads lazily only for oversized input, packs
sentence spans into token-safe chunks, then falls back to clauses and safe word
boundaries. Pronunciation overrides and linguistic annotations are rebased, trace
metadata reports actual chunking, and the chunks are joined into one RenderedSegment.
SpaCy is optional; a single word that cannot fit safely still raises
SynthesisInputTooLongError.
