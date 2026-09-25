# Short-sentence synthesis

Short-sentence handling is an engine-local Kokoro behavior. When enabled, the renderer
can try its configured phrase/cut strategy and fall back to a wrapped request result. It
must still return one waveform for the public request; callers do not receive an
engine-authored pause or cross-request timeline.

Use `ShortSentenceConfig` through `SynthesisConfig` when a model/application needs
explicit short-sentence policy:

```python
from pykokoro import GenerationConfig, ShortSentenceConfig, SynthesisConfig

config = SynthesisConfig(
    generation=GenerationConfig(lang="en-us"),
    short_sentence_config=ShortSentenceConfig(resolve_mode="wrap"),
)
```

The default behavior depends on the selected model and runtime profile. Prefer the
defaults unless a reproducible use case requires a specific strategy.
`return_trace=True` attaches request-local diagnostic events where available. Internal
attempts, retries, trimming, and chunk stitching are implementation details and do not
change the one-request/one-result contract.

Short-sentence handling is separate from editorial pauses, cross-request silence,
playback rate, and final composition. Those policies remain with the caller.
