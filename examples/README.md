# PyKokoro examples

Run examples from the repository root after installing `pykokoro[cpu]`.

- `prosody_demo.py` renders SSMD volume, pitch, rate, and combined metadata.
- `prosody_algorithm_selection.py` is a small diagnostic harness: it writes one neutral
  reference and one output per AudioSig prosody backend from the same waveform.
- `compare_prosody_algorithms.py` creates rate, pitch, and combined renders, diagnostic
  CSV/JSON metrics, and a randomized blind listening set.

## Comparing prosody algorithms

Use a known-good WAV when testing on Termux/Android:

```bash
python examples/prosody_algorithm_selection.py \
  --input-wav reference.wav
```

The diagnostic tool writes `reference.wav`, one output per backend, and `metrics.json`.
Listen to the reference first. The standard ONNX Runtime package reports Android as
unsupported, so source synthesis and AudioSig processing should be diagnosed separately.

The default diagnostic comparison modifies rate and pitch only. Positive gain is
excluded because it can push a full-scale TTS waveform above the PCM WAV range and make
every backend sound clipped. The writer rejects over-range samples instead of silently
clipping.

The comparison script accepts an existing WAV to avoid model synthesis:

```bash
python examples/compare_prosody_algorithms.py \
  --input-wav input.wav \
  --output-dir build/prosody-comparison
```

Objective metrics in the comparison output are diagnostic only; they do not measure
perceived naturalness.
