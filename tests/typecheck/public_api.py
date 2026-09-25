from typing import assert_type

from pykokoro import (
    GenerationConfig,
    KokoroSynthesizer,
    RenderedSegment,
    SynthesisConfig,
    SynthesisSegment,
    discover_models,
)

config = SynthesisConfig(voice="af_sarah", generation=GenerationConfig(lang="en-us"))
request = SynthesisSegment("line-1", "Hello.", "en-us", voice="af_sarah")
assert callable(discover_models)
assert_type(SynthesisConfig, type[SynthesisConfig])
assert_type(KokoroSynthesizer, type[KokoroSynthesizer])
assert_type(RenderedSegment, type[RenderedSegment])
assert_type(config, SynthesisConfig)
assert_type(request, SynthesisSegment)
