# Short-sentence quality validation

The warm benchmark compares the four supported policies using the same persistent
pipelines:

```text
disabled
energy-valley
timestamp-adaptive
wrap
```

Run it with model assets available:

```bash
python benchmarks/short_sentence_latency.py --voices af_sarah
```

Use `--dry-run` to inspect the scenario matrix without loading model assets. The
benchmark includes the existing 13 demo texts and the required declarative,
question, exclamation, interjection, two-word, fragment, ellipsis, and near-minimum
scenarios. It reports structural counters for ONNX calls, phrase attempts, retries,
wrap fallbacks, strict cuts, adaptive cuts, and failures. Wall-clock values are
informational and are not CI pass/fail gates.

Each rendered row also reports objective boundary measurements:

- absolute amplitude at the returned left and right endpoints;
- adjacent sample slope at each endpoint;
- local RMS near each endpoint;
- distance from each timestamp anchor in samples and milliseconds;
- retained guard duration in milliseconds when timestamp metadata is available.

Review the generated outputs in an A/B listening pass. Check for missing initial
consonants, clipped final consonants, leaked neighboring context, clicks, abrupt
gating, and incorrect question or exclamation prosody. Compare old energy-valley,
new timestamp-adaptive, disabled, and wrap outputs. No human reference recording is
required, and no wall-clock improvement should be claimed without a real warm
benchmark run on the target hardware.
