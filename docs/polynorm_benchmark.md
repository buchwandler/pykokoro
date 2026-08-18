# PolyNorm benchmark

PyKokoro includes a maintainer-focused benchmark for the end-to-end frontend path:

```text
PolyNorm text -> PyKokoro document parsing -> kokorog2p / Spokenform semantics -> Kokoro phonemes
```

The benchmark compares phonemizing the original PolyNorm text against phonemizing the
reviewed spoken-form target. It uses the real PyKokoro pipeline through G2P and swaps
only the downstream audio stages for no-op adapters, so it does not require ONNX
Runtime or Kokoro model downloads.

## License and pinned corpus

- Repository: `https://github.com/apple/ml-speech-polynorm-bench`
- Commit: `f3c67e047bea6b7c40bc2466c0fdaad51d8ce67d`
- License: `CC BY-NC-ND 4.0`

PyKokoro does **not** bundle the PolyNorm corpus. First download requires explicit
acknowledgement:

```bash
python -m benchmarks.polynorm_phoneme --accept-license
```

The benchmark caches the pinned JSONL files under a commit-scoped cache root and can be
re-run offline:

```bash
python -m benchmarks.polynorm_phoneme --offline --locale en-US
```

## Reproducing a single case

```bash
python -m benchmarks.polynorm_phoneme \
  --offline \
  --case en-US:1 \
  --pipeline both \
  --show-failures all
```

`--pipeline plain` runs the plain-text parser, `--pipeline ssmd` runs the SSMD parser,
and `--pipeline both` compares both frontends in one report.

## Metrics

The benchmark reports:

- `raw_phoneme_exact`
- `semantic_phoneme_exact`
- `token_exact`
- `token_error_rate`
- phoneme/token edit distance

`semantic_phoneme_exact` is the primary maintainer metric. Raw phoneme equality and
token equality remain visible so regressions in punctuation handling or tokenization are
not hidden.

## Reports

Each run writes:

- `summary.json`: aggregate-only metadata with no PolyNorm sentence text
- `failures.jsonl`
- `failures.md`

The failure reports are local diagnostics and are ignored by default because they
contain corpus text.

## Baseline and strict modes

The committed baseline file stores reviewed failure ids for regression gating:

```bash
python -m benchmarks.polynorm_phoneme \
  --offline \
  --baseline benchmarks/baselines/polynorm_phoneme.json
```

Baseline mode fails only on new reviewed failures. Strict mode fails on any remaining
semantic/token mismatch:

```bash
python -m benchmarks.polynorm_phoneme --offline --strict
```

You can compare two summaries directly:

```bash
python -m benchmarks.polynorm_compare before/summary.json after/summary.json
```

## Fault attribution

For benchmark failures, PyKokoro records:

- direct kokorog2p diagnostics
- optional direct Spokenform preparation output
- likely-owner classification

These signals are advisory only. They help decide whether to inspect PyKokoro
segmentation, kokorog2p, or Spokenform first.

## Limitations

- The benchmark does not prove general G2P correctness outside the pinned corpus.
- The external corpus is not downloaded during normal pytest runs.
- The default benchmark gate stops at phoneme/token output and does not compare audio.
