# PyKokoro examples

Run maintained examples from the repository root after installing `pykokoro[cpu]`. The
scripts below use the current pipeline-first API.

## Basic pipeline

- `german.py` — short German synthesis example; requires ONNX Runtime and model assets.
- `play_audio.py` — generate speech and play it directly; install the optional
  `sounddevice` dependency.
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

## Paragraph and long-form rendering

- `paragraph_wave_export.py` — one WAV per paragraph with an atomic resumable manifest;
  requires ONNX Runtime, model assets, and `soundfile`.
- `paragraph_ssmd_voices.py` — SSMD 0.8 YAML voice bindings, pause defaults, and
  markers; requires ONNX Runtime and model assets.
- `paragraph_stream_to_chapter.py` — append units directly to one chapter WAV with a
  marker sidecar; requires ONNX Runtime and model assets.

Preparation is global: parsing, G2P, and phoneme preprocessing happen once, while
generation and postprocessing are bounded to one selected paragraph waveform. Persist or
copy a result before advancing the iterator because advancing releases the previous
waveform. Descriptor hashes use the `pykokoro-audio-unit-v1` schema.

## Archived examples

The `legacy/` directory contains historical scripts that targeted the removed `Kokoro()`
/ `.create()` API. They are retained for migration reference only and are not part of
the maintained example surface. New examples must use `KokoroPipeline`,
`PipelineConfig`, and `GenerationConfig`.
