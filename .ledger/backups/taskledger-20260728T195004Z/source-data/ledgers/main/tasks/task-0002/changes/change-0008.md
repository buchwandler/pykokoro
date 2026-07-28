---
schema_version: 1
object_type: change
file_version: v2
change_id: change-0008
task_id: task-0002
implementation_run: run-0002
timestamp: "2026-07-26T14:16:03Z"
kind: scan
path: .
summary:
  Reconciled the user-applied SSMD patch plus the changelog and Ruff cleanup with the
  completed implementation checklist.
git_commit: null
git_diff_stat: "branch: main\nstatus:\nM  .codecrate.toml\n M README.md\n M
  docs/basic_usage.rst\n\
  \ M docs/pipeline_stages.rst\n M examples/english.py\n M examples/podcast.py\n M\
  \ examples/prosody_demo.py\n M examples/say_as_demo.py\n M examples/ssmd_demo.py\n\
  \ M pykokoro/generation_config.py\n M pykokoro/phoneme_dictionary.py\n M
  pykokoro/ssmd_parser.py\n\
  \ M tests/test_phoneme_dictionary.py\n M tests/test_ssmd.py\n?? .taskledger.toml\n\
  ?? CHANGELOG.md\n?? PYKOKORO_SSMD_MAINTAINER_HANDOFF.md\n?? plan.md\n??
  pykokoro_ssmd_corrections.patch\n\
  diff_stat:\nREADME.md                        |  36 ++--\n docs/basic_usage.rst \
  \            |   2 +-\n docs/pipeline_stages.rst         |  12 +-\n
  examples/english.py\
  \              |  12 +-\n examples/podcast.py              | 100 +++++----\n
  examples/prosody_demo.py\
  \         | 428 +++++++--------------------------------\n examples/say_as_demo.py\
  \          | 114 +++++------\n examples/ssmd_demo.py            |  47 +++--\n
  pykokoro/generation_config.py\
  \    |   2 +-\n pykokoro/phoneme_dictionary.py   |  16 +-\n pykokoro/ssmd_parser.py\
  \          | 190 ++++++++++-------\n tests/test_phoneme_dictionary.py |  12 ++\n\
  \ tests/test_ssmd.py               |  77 ++++++-\n 13 files changed, 457
  insertions(+),\
  \ 591 deletions(-)"
command: git branch --show-current && git status --short && git diff --stat
before_hash: null
after_hash: null
exit_code: null
---

Reconciled the user-applied SSMD patch plus the changelog and Ruff cleanup with the
completed implementation checklist.
