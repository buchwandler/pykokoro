# PyKokoro examples

These examples use the request-centric synthesis API. Run them from the repository root
with `pykokoro[cpu]` installed. Synthesis examples may download model assets on first
use; outputs are written below `example-artifacts/`.

## Request API examples

- `simple_synthesis.py` — synthesize one prepared string and save a floating-point WAV.
- `request_batch.py` — synthesize independent requests with per-request voices and save
  each result separately; PyKokoro does not concatenate the results.
- `pronunciation_overrides.py` — provide a source-aligned pronunciation-language
  override.
- `linguistic_annotations.py` — pass caller-owned POS/tag/lemma context without planner
  or spaCy objects.
- `models_and_languages.py` — inspect the model registry without loading weights, or
  select a runnable model and synthesize a sample with it.
- `all_voices.py` — optional, potentially slow showcase that renders independent
  requests and performs its own caller-side WAV assembly.

List or run the maintained examples:

```bash
python examples/run_all.py --list
python examples/run_all.py
python examples/run_all.py --include-optional
```

The runner stores each script's artifacts in its own directory. The optional all-voices
showcase can download multiple model/voice assets and take a long time on CPU.
