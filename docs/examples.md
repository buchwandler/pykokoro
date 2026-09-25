# Examples

The maintained scripts in `examples/` use the request-centric API. Run them from the
project root after installing `pykokoro[cpu]`:

```bash
python examples/run_all.py --list
python examples/run_all.py
```

Synthesis examples download model assets on first use. The sequential runner gives each
script its own output directory below `example-artifacts/` and continues after a
failure.

## Simple synthesis

[`examples/simple_synthesis.py`](../examples/simple_synthesis.py) renders prepared text
with `synthesize_text()` and writes the independent result as a float32 WAV.

## Independent request batches

[`examples/request_batch.py`](../examples/request_batch.py) passes two requests with
distinct voices to `synthesize_segments()` and writes each `RenderedSegment` separately.
It does not join them into one timeline.

## Pronunciation and linguistic context

- [`examples/pronunciation_overrides.py`](../examples/pronunciation_overrides.py)
  supplies an explicit source-aligned pronunciation-language span.
- [`examples/linguistic_annotations.py`](../examples/linguistic_annotations.py) passes
  POS, tag, and lemma context without planner or spaCy objects.

## Model inventory

[`examples/models_and_languages.py`](../examples/models_and_languages.py) displays
available model profiles without downloading weights. Pass `--model MODEL_ID` to
synthesize a sample with a selected runnable model; experimental frontends require
`--include-experimental`.

The optional [`examples/all_voices.py`](../examples/all_voices.py) showcase can take a
long time and download several model/voice assets. It requests each voice independently
and assembles its own WAV as caller-side example code; this composition behavior is not
part of PyKokoro's engine API.
