# PyKokoro hard-cases benchmark

`benchmarks.hard_cases` is a first-party, deterministic regression suite for difficult
English and German frontend cases. It deliberately has **no human-reference audio** and
does not measure MOS, naturalness, speaker similarity, or human-likeness.

## Levels

- `normalization`: Spokenform output, replacements, warnings, offsets, and language
  runs.
- `phoneme`: PyKokoro's real Spokenform/Phrasplit/KokoroG2P path with no-op downstream
  adapters; compares raw and semantic phonemes, tokens, edit distances, and critical
  spans.
- `plan`: segment offsets, sentence/clause boundaries, pauses, and SSMD-related
  metadata.
- `frontend`: the ordinary fast suite (normalization through phonemes).
- `acoustic`: optional waveform health, timing, pause, duration, and runtime
  diagnostics.
- `all`: runs all available levels.

Acoustic results are labelled **acoustic health**, **timing stability**, and **runtime
performance**. A passing result does not imply that speech sounds natural.

## Running

The built-in corpus is offline and packaged as JSONL:

```bash
python -m benchmarks.hard_cases --list-languages
python -m benchmarks.hard_cases --list-locales
python -m benchmarks.hard_cases --language en --locale en-US --level frontend
python -m benchmarks.hard_cases --language de --locale de-DE --level frontend
python -m benchmarks.hard_cases --case en_shared_001 --show-details
python -m benchmarks.hard_cases --language de --level acoustic --model v1.2-de-martin --lexicon gold
```

Use `--category`, `--limit`, `--frontend-variant`, `--ssmd`, `--results-dir`, and
`--render-audio` to reproduce a focused case. Generated `summary.json`, `cases.jsonl`,
`failures.jsonl`, `failures.md`, and `environment.json` belong under the requested
results directory (or `.benchmarks/hard_cases/`) and should not be committed.

## Corpus and ownership

Rows use schema version 1 and include language, optional explicit locale, category,
provenance, tags, and structured expectations. Shared rows use `locale: null`; they may
be selected for any compatible locale. The initial corpus contains more than 100 cases
per language and covers normalization, abbreviations, numbers, acronyms, names,
homographs/heteronyms, German compounds/prefixes/Denglisch, punctuation, dirty text,
SSMD, code-switching, and long-form interactions.

Failures are attributed to the earliest failed contract: `spokenform`, `phrasplit`,
`kokorog2p_or_spokenform`, `pykokoro_pipeline`, or `acoustic_model`. Baseline and
quarantine records track known failures without hiding new regressions.

## CI guidance

Pull requests should run schema/data/selection and the no-ONNX frontend subset.
Scheduled jobs can run the complete frontend corpus. Acoustic model/voice/lexicon
matrices are optional and should be isolated from normal unit tests so they never
trigger downloads. PolyNorm remains a separate external normalization benchmark.
