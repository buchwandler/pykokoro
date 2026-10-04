# Examples

The maintained scripts in `examples/` use the request-centric API. Install one provider
(for example, `pykokoro[cpu]`) before running a synthesis example. Uncached synthesis
may download model assets; discovery-only scripts are identified below. Outputs go to
`example-artifacts/` when run directly, and to `example-artifacts/<script>/` through the
runner.

## Runner groups

```bash
python examples/run_all.py --list
python examples/run_all.py --group core
python examples/run_all.py --group feature
python examples/run_all.py --group language-showcase
python examples/run_all.py --group optional-heavy
python examples/run_all.py --include-optional
```

The default group is `core`. Feature examples and the language showcase are separate
from it; `optional-heavy` is never added unless selected explicitly or
`--include-optional` is used. The runner processes scripts sequentially, isolates each
script's output directory, and continues after a script fails. Use `--list` to inspect a
group without running it.

## Core request examples

| Script / command                                                                                                        | Purpose and public API                                                                                                                                                                       | Assets, network, and cost                                                                                                                                                  | Expected output                                                                                                                      |
| ----------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| [`simple_synthesis.py`](../examples/simple_synthesis.py)<br>`python examples/simple_synthesis.py`                       | Smallest request using `SynthesisConfig`, `KokoroSynthesizer.synthesize_text()`, and `RenderedSegment.save_wav()`.                                                                           | First run may download one model/voice asset; low CPU cost.                                                                                                                | `hello.wav`                                                                                                                          |
| [`request_batch.py`](../examples/request_batch.py)<br>`python examples/request_batch.py`                                | Two explicit `SynthesisSegment` requests through `synthesize_segments()`. Results are saved independently; PyKokoro does not concatenate them or insert silence.                             | First run may download model assets; low-to-medium CPU cost for two short requests.                                                                                        | `greeting-a.wav`, `greeting-b.wav`                                                                                                   |
| [`pronunciation_overrides.py`](../examples/pronunciation_overrides.py)<br>`python examples/pronunciation_overrides.py`  | Contrasts a source-span `PronunciationOverride(language=...)` with a direct phoneme span.                                                                                                    | First run may download model assets; low-to-medium cost for two short requests.                                                                                            | `language-override.wav`, `phoneme-span-override.wav`                                                                                 |
| [`linguistic_tokens.py`](../examples/linguistic_tokens.py)<br>`python examples/linguistic_tokens.py`                    | Passes a source-aligned `LinguisticToken` through the canonical `tokens=` field.                                                                                                             | First run may download model assets; low CPU cost.                                                                                                                         | `contextual-pronunciation.wav`                                                                                                       |
| [`long_text.py`](../examples/long_text.py)<br>`python examples/long_text.py`                                            | Sends one oversized original request with `long_text_split="sentence"`; engine-managed chunks return as one result.                                                                          | First run may download model assets; higher CPU cost for the long passage.                                                                                                 | `long_text.wav`                                                                                                                      |
| [`voice_blend.py`](../examples/voice_blend.py)<br>`python examples/voice_blend.py`                                      | Shows structured and parsed `VoiceBlend` construction and synthesizes one blended voice.                                                                                                     | First run may download assets for the selected model/voices; low-to-medium CPU cost.                                                                                       | `voice_blend.wav`                                                                                                                    |
| [`result_metadata.py`](../examples/result_metadata.py)<br>`python examples/result_metadata.py`                          | Prints request/result fields, token IDs, diagnostics, trace summary, synthesis identity, word timings, and voice-level applications.                                                         | First run may download model assets; low CPU cost for one request.                                                                                                         | `result_metadata.wav` and printed metadata                                                                                           |
| [`models_and_languages.py`](../examples/models_and_languages.py)<br>`python examples/models_and_languages.py --offline` | Uses `discover_models()` to print runtime model/profile, language, and voice inventory without creating an ONNX session. `--model MODEL_ID` optionally synthesizes using a selected profile. | Default discovery may refresh registry metadata over the network; `--offline` uses cached metadata. No model download unless `--model` is supplied; discovery cost is low. | Default: printed inventory only. With `--model`: `<model>_<language>_<voice>.wav` under `example-artifacts/model_language_outputs/`. |

## Feature examples

| Script / command                                                                                                                                     | Purpose and public API                                                                                                                                       | Assets, network, and cost                                                                           | Expected output                                                           |
| ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| [`english.py`](../examples/english.py)<br>`python examples/english.py`                                                                               | Synthesizes prepared English text, calls `prepare()`, and reuses its phonemes in a whole-request phoneme request.                                            | First run may download model assets; two short inferences, low-to-medium CPU cost.                  | `english_demo.wav`, `english_phonemes_demo.wav`                           |
| [`language_routing.py`](../examples/language_routing.py)<br>`python examples/language_routing.py`                                                    | Contrasts automatic G2P language routing with an explicit deterministic language override on the same source span. Acoustic model and voice remain explicit. | First run may download model assets; low-to-medium cost for two short requests.                     | `automatic-routing.wav`, `explicit-german-span.wav`                       |
| [`frontend_and_lexicons.py`](../examples/frontend_and_lexicons.py)<br>`python examples/frontend_and_lexicons.py`                                     | Runs offline `discover_lexicons()` and constructs an installed-only `TokenizerConfig`; it does not synthesize.                                               | No model assets or lexicon data are installed; offline discovery avoids metadata refresh. Low cost. | Printed lexicon inventory and frontend configuration; no file.            |
| [`asset_progress.py`](../examples/asset_progress.py)<br>`python examples/asset_progress.py`<br>`python examples/asset_progress.py --custom-callback` | Demonstrates `ConsoleAssetProgress` or a typed `AssetProgressEvent` callback during managed model asset installation.                                        | May download model assets on first use; low CPU cost for one short request.                         | `asset_progress.wav` plus progress events                                 |
| [`error_handling.py`](../examples/error_handling.py)<br>`python examples/error_handling.py`                                                          | Shows narrow catches for invalid language, voice, pronunciation, and oversized input errors.                                                                 | First run may download model assets on the valid path; low CPU cost for one short request.          | Normally `error_handling.wav`; a caught failure prints a message instead. |

## Input-driven example

[`reference_voice.py`](../examples/reference_voice.py) requires a user-supplied
recording and its exact English transcript, so it is not run by the unattended example
groups:

```bash
python examples/reference_voice.py reference.wav \
  "The exact words spoken in the reference." \
  "New words spoken with the enrolled voice."
```

The script enrolls the recording, saves and reloads `ReferenceVoice`, then synthesizes
the target text. The [reference voice guide](reference_voice.md) documents audio
constraints, model binding, privacy, and permissions.

## Language showcase

These scripts are separated from the default runner group because each performs
synthesis; uncached runs may download model assets. Each is a short-to-medium
single-language showcase unless noted. Run one directly with the command shown.

| Script / command                                                                      | Purpose and public API                                                                                                     | Assets, network, and cost                                                                     | Expected output                            |
| ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------ |
| [`chinese.py`](../examples/chinese.py)<br>`python examples/chinese.py`                | Synthesizes Mandarin with explicit `zh` language and the `v1.1-zh` model variant.                                          | May download the Chinese model/voice assets; medium CPU cost.                                 | `chinese_demo.wav`                         |
| [`contractions.py`](../examples/contractions.py)<br>`python examples/contractions.py` | Exercises English contractions and past-tense endings through multiple requests; joins audio in this caller-owned example. | May download model assets; higher CPU cost than a single short sample.                        | `contractions_demo.wav` (caller-assembled) |
| [`french.py`](../examples/french.py)<br>`python examples/french.py`                   | Synthesizes a prepared French passage with an explicit French voice/language.                                              | May download model assets; medium CPU cost.                                                   | `french_demo.wav`                          |
| [`italian.py`](../examples/italian.py)<br>`python examples/italian.py`                | Synthesizes an Italian passage using the request API.                                                                      | May download model assets; medium CPU cost.                                                   | `italian_demo.wav`                         |
| [`japanese.py`](../examples/japanese.py)<br>`python examples/japanese.py`             | Synthesizes a Japanese passage with an explicit `ja` request language.                                                     | May download model assets; medium CPU cost.                                                   | `japanese_demo.wav`                        |
| [`korean.py`](../examples/korean.py)<br>`python examples/korean.py`                   | Demonstrates experimental Korean phonemization and a parsed `VoiceBlend`; pronunciation may be inaccurate.                 | Requires a compatible experimental frontend/profile and may download assets; medium CPU cost. | `korean_demo.wav`                          |
| [`portuguese.py`](../examples/portuguese.py)<br>`python examples/portuguese.py`       | Synthesizes a Brazilian Portuguese passage with an explicit voice and language.                                            | May download model assets; medium CPU cost.                                                   | `portuguese_demo.wav`                      |
| [`spanish.py`](../examples/spanish.py)<br>`python examples/spanish.py`                | Synthesizes a prepared Spanish passage with the request API.                                                               | May download model assets; medium CPU cost.                                                   | `spanish_demo.wav`                         |

## Optional heavy examples

| Script / command                                                                                                                                    | Purpose and public API                                                                                                                                                          | Assets, network, and cost                                                                                                                                                        | Expected output                                                                                   |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| [`all_voices.py`](../examples/all_voices.py)<br>`python examples/all_voices.py --list-only` (inventory)<br>`python examples/all_voices.py` (render) | Uses model discovery to enumerate profiles, then submits one independent synthesis request per selected voice. The script itself assembles a comparison WAV on the caller side. | Inventory may refresh registry metadata. Rendering can download many model/voice assets, use substantial storage, and run for a long time on CPU; list-only does not synthesize. | `all_voices.wav`; with `--compare-leveling`, `all_voices_raw.wav` and `all_voices_calibrated.wav` |
| [`short_sentence_demo.py`](../examples/short_sentence_demo.py)<br>`python examples/short_sentence_demo.py`                                          | Compares disabled, wrap, phrase, and randomized-phrase `ShortSentenceConfig` modes; joins comparison sections in example code, not in the engine.                               | May download model assets; high CPU cost (many short requests across four modes).                                                                                                | `short_sentence_demo.wav` (caller-assembled)                                                      |

Every runnable script is listed above. `__init__.py`, `_output.py`, and `run_all.py` are
support modules rather than synthesis examples. For API details, see the
[reference](api_reference.md).
