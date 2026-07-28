---
schema_version: 1
object_type: change
file_version: v2
change_id: change-0048
task_id: task-0001
implementation_run: run-0005
timestamp: "2026-07-26T11:35:23Z"
kind: scan
path: .
summary:
  Scanned the completed Phase 1-7 implementation and final validation-gate changes.
git_commit: null
git_diff_stat: "branch: main\nstatus:\nM .codecrate.toml\n M
  .github/workflows/tests.yml\n\
  \ M .pre-commit-config.yaml\n D .ruff.toml\n M AGENTS.md\n M README.md\n M
  docs/installation.rst\n\
  \ M docs/make.py\n M examples/automatic_pauses_demo.py\n M
  examples/backend_comparison.py\n\
  \ M examples/boundary_detection_analysis.py\n M examples/contractions_advanced.py\n\
  \ M examples/cpu_benchmark.py\n M examples/english.py\n M
  examples/generation_config_demo.py\n\
  \ M examples/gpu_benchmark.py\n M examples/headings_demo.py\n M
  examples/hf_v11zh_demo.py\n\
  \ M examples/hindi.py\n M examples/japanese.py\n M examples/mixed_language.py\n\
  \ M examples/model_source_comparison.py\n M examples/optimal_phoneme_length_demo.py\n\
  \ M examples/pauses_demo.py\n M examples/pauses_with_splitting.py\n M
  examples/phoneme_diff_comparison.py\n\
  \ M examples/pipeline_stage_showcase.py\n M examples/podcast.py\n M
  examples/prosody_demo.py\n\
  \ M examples/provider_config_demo.py\n M examples/provider_info.py\n M
  examples/say_as_demo.py\n\
  \ M examples/short_sentence_demo.py\n M examples/short_sentence_randomized_demo.py\n\
  \ M examples/token_length_effects.py\n M examples/voice_slerp_demo.py\n M
  metrics/find_short_sentence_phrase_candidates.py\n\
  \ M metrics/rank_short_sentence_phrases_across_voice_list.py\n M
  pykokoro/__init__.py\n\
  \ M pykokoro/audio_generator.py\n M pykokoro/debug/segment_invariants.py\n M
  pykokoro/generation_config.py\n\
  \ M pykokoro/mixed_language_handler.py\n M pykokoro/onnx_backend.py\n M
  pykokoro/onnx_session.py\n\
  \ M pykokoro/phoneme_dictionary.py\n M pykokoro/pipeline.py\n M
  pykokoro/pipeline_config.py\n\
  \ M pykokoro/prosody.py\n M pykokoro/provider_config.py\n M
  pykokoro/runtime/cache.py\n\
  \ M pykokoro/runtime/spans.py\n M pykokoro/say_as.py\n M
  pykokoro/short_sentence_cutters/__init__.py\n\
  \ M pykokoro/short_sentence_cutters/energy_valley.py\n M
  pykokoro/short_sentence_cutters/shared.py\n\
  \ M pykokoro/short_sentence_cutters/vad.py\n M pykokoro/short_sentence_handler.py\n\
  \ M pykokoro/ssmd_parser.py\n M pykokoro/stages/audio_postprocessing/noop.py\n M\
  \ pykokoro/stages/audio_postprocessing/onnx.py\n M
  pykokoro/stages/doc_parsers/plain.py\n\
  \ M pykokoro/stages/doc_parsers/ssmd.py\n M pykokoro/stages/g2p/kokorog2p.py\n M\
  \ pykokoro/stages/synth/noop.py\n M pykokoro/tokenizer.py\n M
  pykokoro/transcript.py\n\
  \ M pykokoro/trim.py\n M pykokoro/utils.py\n M pykokoro/voice_manager.py\n M
  pyproject.toml\n\
  \ M setup.py\n M tests/test_audio_splitting.py\n M tests/test_auto_pause_mode.py\n\
  \ M tests/test_bug_fixes.py\n M tests/test_downloads.py\n M
  tests/test_g2p_abbreviations.py\n\
  \ M tests/test_g2p_pause_propagation.py\n M tests/test_generation_config.py\n M\
  \ tests/test_mixed_language.py\n M tests/test_phrasplit_fallbacks.py\n M
  tests/test_phrasplit_overrides.py\n\
  \ M tests/test_pipeline_lifecycle.py\n M tests/test_pipeline_paragraphs.py\n M
  tests/test_pipeline_spacy.py\n\
  \ M tests/test_prosody.py\n M tests/test_session_manager.py\n M
  tests/test_short_sentence_handler.py\n\
  \ M tests/test_splitter_no_overlap.py\n M tests/test_ssmd.py\n M
  tests/test_ssmd_deterministic_breaks.py\n\
  \ M tests/test_tokenizer.py\n M tests/test_utils.py\n M tests/test_voice_manager.py\n\
  ?? .taskledger.toml\n?? PYKOKORO_MAINTAINER_REVIEW-4.md\n?? plan.md\n??
  pykokoro/artifact_manifest.py\n\
  ?? pykokoro/config_types.py\n?? tests/test_import_boundaries.py\n??
  tests/test_maintainer_review4_download_cache.py\n\
  ?? tests/test_maintainer_review4_reproductions.py\n??
  tests/test_packaging_metadata.py\n\
  diff_stat:\n.codecrate.toml                                    |   3 +-\n
  .github/workflows/tests.yml\
  \                        |  10 +-\n .pre-commit-config.yaml                    \
  \        |   2 +-\n .ruff.toml                                         |  28 --\n\
  \ AGENTS.md                                          |   2 +-\n README.md      \
  \                                    |  13 +-\n docs/installation.rst          \
  \                    |  15 +-\n docs/make.py                                   \
  \    |   5 +-\n examples/automatic_pauses_demo.py                  |   5 +-\n
  examples/backend_comparison.py\
  \                     |   8 +-\n examples/boundary_detection_analysis.py       \
  \     |   8 +-\n examples/contractions_advanced.py                  |   4 +-\n
  examples/cpu_benchmark.py\
  \                          |   2 +-\n examples/english.py                      \
  \          |   4 +-\n examples/generation_config_demo.py                 |   5 +-\n\
  \ examples/gpu_benchmark.py                          |   5 +-\n
  examples/headings_demo.py\
  \                          |   4 +-\n examples/hf_v11zh_demo.py                \
  \          |   4 +-\n examples/hindi.py                                  |   5 +-\n\
  \ examples/japanese.py                               |   5 +-\n
  examples/mixed_language.py\
  \                         |   4 +-\n examples/model_source_comparison.py       \
  \         |  54 +--\n examples/optimal_phoneme_length_demo.py            |   4 +-\n\
  \ examples/pauses_demo.py                            |   4 +-\n
  examples/pauses_with_splitting.py\
  \                  |  27 +-\n examples/phoneme_diff_comparison.py              \
  \  |  20 +-\n examples/pipeline_stage_showcase.py                |   4 +-\n
  examples/podcast.py\
  \                                |   4 +-\n examples/prosody_demo.py           \
  \                |  53 +--\n examples/provider_config_demo.py                  \
  \ |   9 +-\n examples/provider_info.py                          |   5 +-\n
  examples/say_as_demo.py\
  \                            |   4 +-\n examples/short_sentence_demo.py        \
  \            |  12 +-\n examples/short_sentence_randomized_demo.py         |   4\
  \ +-\n examples/token_length_effects.py                   |   8 +-\n
  examples/voice_slerp_demo.py\
  \                       |  24 +-\n metrics/find_short_sentence_phrase_candidates.py\
  \   |  39 +--\n ...ank_short_sentence_phrases_across_voice_list.py |  31 +-\n
  pykokoro/__init__.py\
  \                               |  41 ++-\n pykokoro/audio_generator.py        \
  \                |  99 ++----\n pykokoro/debug/segment_invariants.py           \
  \    |   9 +-\n pykokoro/generation_config.py                      |  61 ++--\n\
  \ pykokoro/mixed_language_handler.py                 |   8 +-\n
  pykokoro/onnx_backend.py\
  \                           | 363 +++++++++++++++------\n pykokoro/onnx_session.py\
  \                           |  23 +-\n pykokoro/phoneme_dictionary.py          \
  \           |  12 +-\n pykokoro/pipeline.py                               | 122\
  \ ++++---\n pykokoro/pipeline_config.py                        |   6 +-\n
  pykokoro/prosody.py\
  \                                |   8 +-\n pykokoro/provider_config.py        \
  \                |   3 +-\n pykokoro/runtime/cache.py                          |\
  \  17 +-\n pykokoro/runtime/spans.py                          |   3 +-\n
  pykokoro/say_as.py\
  \                                 |   8 +-\n
  pykokoro/short_sentence_cutters/__init__.py\
  \        |   4 +-\n pykokoro/short_sentence_cutters/energy_valley.py   |  18 +-\n\
  \ pykokoro/short_sentence_cutters/shared.py          |   4 +-\n
  pykokoro/short_sentence_cutters/vad.py\
  \             |  20 +-\n pykokoro/short_sentence_handler.py                 |  87\
  \ +++--\n pykokoro/ssmd_parser.py                            |  26 +-\n
  pykokoro/stages/audio_postprocessing/noop.py\
  \       |   6 +-\n pykokoro/stages/audio_postprocessing/onnx.py       |   4 +-\n\
  \ pykokoro/stages/doc_parsers/plain.py               |  46 +--\n
  pykokoro/stages/doc_parsers/ssmd.py\
  \                |  22 +-\n pykokoro/stages/g2p/kokorog2p.py                   |\
  \ 105 +++---\n pykokoro/stages/synth/noop.py                      |   4 +-\n
  pykokoro/tokenizer.py\
  \                              |  27 +-\n pykokoro/transcript.py               \
  \              |  20 +-\n pykokoro/trim.py                                   | \
  \ 24 +-\n pykokoro/utils.py                                  |  24 +-\n
  pykokoro/voice_manager.py\
  \                          |  51 +--\n pyproject.toml                          \
  \           |  15 +-\n setup.py                                           |   5\
  \ +-\n tests/test_audio_splitting.py                      |  72 +---\n
  tests/test_auto_pause_mode.py\
  \                      |  12 +-\n tests/test_bug_fixes.py                      \
  \      |  12 +-\n tests/test_downloads.py                            |  16 +-\n\
  \ tests/test_g2p_abbreviations.py                    |  10 +-\n
  tests/test_g2p_pause_propagation.py\
  \                |  12 +-\n tests/test_generation_config.py                    |\
  \  14 +\n tests/test_mixed_language.py                       |   8 +-\n
  tests/test_phrasplit_fallbacks.py\
  \                  |   8 +-\n tests/test_phrasplit_overrides.py                \
  \  |   4 +-\n tests/test_pipeline_lifecycle.py                   |   8 +-\n
  tests/test_pipeline_paragraphs.py\
  \                  |   7 +-\n tests/test_pipeline_spacy.py                     \
  \  |   4 +-\n tests/test_prosody.py                              |   8 +-\n
  tests/test_session_manager.py\
  \                      |  33 +-\n tests/test_short_sentence_handler.py         \
  \      |  24 ++\n tests/test_splitter_no_overlap.py                  |   4 +-\n\
  \ tests/test_ssmd.py                                 |  13 +-\n
  tests/test_ssmd_deterministic_breaks.py\
  \            |   5 +-\n tests/test_tokenizer.py                            |  16\
  \ +-\n tests/test_utils.py                                |  20 +-\n
  tests/test_voice_manager.py\
  \                        |  10 +-\n 94 files changed, 916 insertions(+), 1146
  deletions(-)"
command: git branch --show-current && git status --short && git diff --stat
before_hash: null
after_hash: null
exit_code: null
---

Scanned the completed Phase 1-7 implementation and final validation-gate changes.
