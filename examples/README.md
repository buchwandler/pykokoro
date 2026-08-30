# PyKokoro examples

Run maintained examples from the repository root after installing `pykokoro[cpu]`. The
scripts below use the current pipeline-first API.

## Basic pipeline

- `german.py` — German Martin v1.2; demonstrates automatic German model selection and
  requires ONNX Runtime/model assets.
- `german2.py` — German Kerstin/Crane; explicitly selects `de-crane`, opts into its
  experimental readiness status, and uses the shared native German G2P path.
- `german3.py` — German Thorsten; explicitly selects the ready `de-thorsten` model.
- `models_and_languages.py` — inspect the canonical runtime registry and display models,
  languages, providers, voices, qualities, frontend/layout status, and optionally
  synthesize one selected model. The default listing never downloads model weights.
- `play_audio.py` — generate a short waveform completely, then play it; install
  `pykokoro[cpu,playback]`.
- `play_streaming.py` — preferred low-startup-latency long-form sentence playback
  through one persistent bounded stream; install `pykokoro[cpu,playback]`.
- `play_paragraphs.py` — render larger paragraph chunks through one persistent stream;
  install `pykokoro[cpu,playback]`.
- `split_and_phonemize_demo.py` — custom document/G2P stage inspection; import-only
  until a backend is selected.
- `termux_android_onnx.py` — Android/Termux provider configuration; requires a
  compatible ONNX Runtime build and model assets.

The maintained analysis and language demos are also import-safe and indexed here:
`abbreviations.py`, `automatic_pauses_demo.py`, `backend_comparison.py`,
`boundary_detection_analysis.py`, `chinese.py`, `compare_prosody_algorithms.py`,
`contractions.py`, `contractions_advanced.py`, `cpu_benchmark.py`, `dash_variations.py`,
`english.py`, `french.py`, `headings_demo.py`, `hindi.py`, `homographs.py`,
`italian.py`, `japanese.py`, `korean.py`, `mixed_language.py`,
`optimal_phoneme_length_demo.py`, `paragraph_streaming.py`, `pauses_demo.py`,
`phoneme_print_demo.py`, `pipeline_g2p_onnx_minimal.py`, `podcast.py`, `portuguese.py`,
`prosody_algorithm_selection.py`, `prosody_demo.py`, `provider_info.py`,
`punctuation.py`, `punctuation_variations.py`, `repro_dup_words.py`, `say_as_demo.py`,
`short_sentence_demo.py`, `short_sentence_randomized_demo.py`,
`short_sentence_voices_demo.py`, `spanish.py`, `split_and_phonemize_demo.py`,
`ssmd_080_portable_podcast.py`, `ssmd_demo.py`, and `termux_android_onnx.py`.

## Paragraph and sentence long-form rendering

- Use `play_streaming.py` for direct playback with the lowest startup latency. It
  renders sentence units sequentially while one persistent stream consumes prior audio.
- Use the paragraph examples when larger chunks, file export, markers, or resumable
  manifests are more important than minimum startup latency.
- `paragraph_wave_export.py` — one WAV per paragraph with an atomic resumable manifest;
  requires ONNX Runtime, model assets, and `soundfile`.
- `paragraph_ssmd_voices.py` — SSMD 0.8 YAML voice bindings, pause defaults, and
  markers; requires ONNX Runtime and model assets.
- `paragraph_stream_to_chapter.py` — append units directly to one chapter WAV with a
  marker sidecar; requires ONNX Runtime and model assets.

Preparation is global: parsing, G2P, and phoneme preprocessing happen once, while
generation and postprocessing are bounded to the selected unit waveform and playback
queue. Persist or copy a result before advancing the iterator because advancing releases
the previous waveform. Descriptor hashes use the `pykokoro-audio-unit-v1` schema.

## Archived examples

The `legacy/` directory contains historical scripts that targeted the removed `Kokoro()`
/ `.create()` API. They are retained for migration reference only and are not part of
the maintained example surface. New examples must use `KokoroPipeline`,
`PipelineConfig`, and `GenerationConfig`.
