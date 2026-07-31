# PyKokoro examples

Run examples from the repository root after installing `pykokoro[cpu]`.

- `prosody_demo.py` renders SSMD volume, pitch, rate, and combined metadata.
- `prosody_algorithm_selection.py` renders one annotated utterance with each AudioSig
  prosody backend in strict mode.
- `compare_prosody_algorithms.py` creates identical rate, pitch, and combined renders,
  diagnostic CSV/JSON metrics, and a randomized blind listening set.

The comparison script accepts an existing WAV to avoid model synthesis:

```bash
python examples/compare_prosody_algorithms.py \
  --input-wav input.wav \
  --output-dir build/prosody-comparison
```

Objective metrics in the comparison output are diagnostic only; they do not measure
perceived naturalness.
