---
schema_version: 1
object_type: change
file_version: v2
change_id: change-0039
task_id: task-0001
implementation_run: run-0003
timestamp: "2026-07-26T11:09:38Z"
kind: scan
path: .
summary:
  Reconciled all Phase 1-7 source, tests, packaging, docs, tooling, and CI changes
  before implementation finish.
git_commit: null
git_diff_stat: "branch: main\nstatus:\nM .codecrate.toml\n M
  .github/workflows/tests.yml\n\
  \ M .pre-commit-config.yaml\n D .ruff.toml\n M AGENTS.md\n M README.md\n M
  docs/installation.rst\n\
  \ M docs/make.py\n M examples/cpu_benchmark.py\n M
  examples/phoneme_diff_comparison.py\n\
  \ M pykokoro/__init__.py\n M pykokoro/audio_generator.py\n M
  pykokoro/generation_config.py\n\
  \ M pykokoro/onnx_backend.py\n M pykokoro/pipeline.py\n M
  pykokoro/pipeline_config.py\n\
  \ M pykokoro/runtime/cache.py\n M pykokoro/short_sentence_handler.py\n M
  pykokoro/stages/g2p/kokorog2p.py\n\
  \ M pykokoro/trim.py\n M pykokoro/voice_manager.py\n M pyproject.toml\n M setup.py\n\
  \ M tests/test_generation_config.py\n M tests/test_short_sentence_handler.py\n??\
  \ .taskledger.toml\n?? PYKOKORO_MAINTAINER_REVIEW-4.md\n?? plan.md\n??
  pykokoro/config_types.py\n\
  ?? tests/test_import_boundaries.py\n??
  tests/test_maintainer_review4_download_cache.py\n\
  ?? tests/test_maintainer_review4_reproductions.py\n??
  tests/test_packaging_metadata.py\n\
  diff_stat:\n.codecrate.toml                      |   3 +-\n
  .github/workflows/tests.yml\
  \          |  10 +-\n .pre-commit-config.yaml              |   2 +-\n .ruff.toml\
  \                           |  28 ----\n AGENTS.md                            |\
  \   2 +-\n README.md                            |  13 +-\n docs/installation.rst\
  \                |  15 ++-\n docs/make.py                         |   5 +-\n
  examples/cpu_benchmark.py\
  \            |   2 +-\n examples/phoneme_diff_comparison.py  |   3 +-\n
  pykokoro/__init__.py\
  \                 |  41 ++++--\n pykokoro/audio_generator.py          |   3 +-\n\
  \ pykokoro/generation_config.py        |  58 ++++----\n pykokoro/onnx_backend.py\
  \             | 255 ++++++++++++++++++++++++++++-------\n pykokoro/pipeline.py \
  \                |  93 +++++++++----\n pykokoro/pipeline_config.py          |  \
  \ 6 +-\n pykokoro/runtime/cache.py            |  13 +-\n
  pykokoro/short_sentence_handler.py\
  \   |  37 +++++\n pykokoro/stages/g2p/kokorog2p.py     |  81 ++++++++---\n
  pykokoro/trim.py\
  \                     |  24 ++--\n pykokoro/voice_manager.py            |   5 +-\n\
  \ pyproject.toml                       |  15 +--\n setup.py                    \
  \         |   5 +-\n tests/test_generation_config.py      |  14 ++\n
  tests/test_short_sentence_handler.py\
  \ |  24 ++++\n 25 files changed, 536 insertions(+), 221 deletions(-)"
command: git branch --show-current && git status --short && git diff --stat
before_hash: null
after_hash: null
exit_code: null
---

Reconciled all Phase 1-7 source, tests, packaging, docs, tooling, and CI changes before
implementation finish.
