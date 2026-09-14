# Short-sentence benchmark tools

PyKokoro provides two separate benchmark tools. They answer different questions and
should not be combined.

## Engineering benchmark

`benchmarks/short_sentence_latency.py` compares the configured policies using a warm
pipeline matrix:

```text
disabled
energy-valley
timestamp-adaptive
wrap
```

It reports final cases separately from phrase attempts. The default output is a compact
table containing cases per policy, phrase-cut success rates, initial and retry success,
wrap fallbacks, actual successful cut strategies, failure reasons, and inference costs.
Raw rows can be exported independently:

```bash
python benchmarks/short_sentence_latency.py \
  --model-source github \
  --voices af_sarah \
  --output-format both \
  --jsonl artifacts/short_sentence_rows.jsonl \
  --summary-json artifacts/short_sentence_summary.json
```

Use `--dry-run` to inspect voices, languages, policies, scenario count, and expected
render count without loading model assets. `--log-level` controls benchmark logging.
This benchmark does not write listening WAVs.

Each row retains timing diagnostics, ONNX counters, wall time, RTF, boundary metrics,
configured policy, final outcome, actual cut strategy, retry count, fallback, and stage-
specific failure reasons. Wall-clock values are informational, not CI thresholds.

## Listening parameter sweep

`benchmarks/short_sentence_parameter_sweep.py` renders one fixed short segment for one
ordered parameter sweep. Each candidate is preceded by a spoken label and separated by
pauses in one combined WAV. Announcement rendering always disables short-sentence
handling so the parameter under test cannot change the label.

Example:

```bash
python benchmarks/short_sentence_parameter_sweep.py \
  --model-source github \
  --model-variant v1.0 \
  --voice af_sarah \
  --lang en-us \
  --text "Why?" \
  --cutter timestamp-adaptive \
  --parameter energy-threshold \
  --values 0.02,0.03,0.05,0.07,0.10
```

The default artifacts are:

```text
artifacts/short_sentence_parameter_sweep.wav
artifacts/short_sentence_parameter_sweep.json
```

Use `--phrase-template` to pin the same context phrase for every candidate,
`--label-template` to localize announcements, `--allow-retries N` only for robustness
experiments, and `--save-individual` for secondary per-candidate files. Retries default
to zero for valid acoustic comparisons. Use `--dry-run` to validate the candidate list
without loading a model.

Supported parameters are `energy-threshold`, `frame-duration-ms`, `min-silence-seconds`,
`search-radius-ms`, `context-guard-ms`, `analysis-window-ms`, and the categorical
`cutter` comparison. Starting ranges are suggestions, not optimum values:

```text
energy-threshold:     0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10
frame-duration-ms:    3, 4, 5, 6, 8, 10
min-silence-seconds:  0.010, 0.015, 0.020, 0.025, 0.030, 0.040
search-radius-ms:     20, 25, 35, 45, 60
context-guard-ms:     0, 4, 8, 12, 16
analysis-window-ms:   2, 3, 5, 8, 10
```

`silence-threshold` remains compatibility metadata and is not consumed by the current
cutters, so the sweep rejects it. The manifest records model, voice, language, speed,
text, phrase template, cutter, seed, fallback policy, sample rate, candidate values,
audio offsets, outcomes, strategies, retries, failure reasons, and raw boundary metrics.

The sweep is a listening aid, not an automatic optimizer. Compare candidates in printed
order and choose based on human perception. Smooth objective boundary metrics do not
replace listening for clipping, retained context, consonants, or prosody.
