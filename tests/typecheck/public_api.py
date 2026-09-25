from collections.abc import Callable
from typing import assert_type

from pykokoro import (
    AssetProgressCallback,
    AssetProgressEvent,
    ConsoleAssetProgress,
    GenerationConfig,
    KokoroSynthesizer,
    LexiconCapabilities,
    LexiconDiscoveryResult,
    ModelCapabilities,
    ModelDiscoveryResult,
    RenderedSegment,
    SynthesisConfig,
    SynthesisSegment,
    VoiceCapabilities,
    discover_lexicons,
    discover_models,
)

config = SynthesisConfig(voice="af_sarah", generation=GenerationConfig(lang="en-us"))
request = SynthesisSegment("line-1", "Hello.", "en-us", voice="af_sarah")

assert_type(AssetProgressCallback, type[AssetProgressCallback])
assert_type(AssetProgressEvent, type[AssetProgressEvent])
assert_type(ConsoleAssetProgress, type[ConsoleAssetProgress])
assert_type(ModelCapabilities, type[ModelCapabilities])
assert_type(ModelDiscoveryResult, type[ModelDiscoveryResult])
assert_type(VoiceCapabilities, type[VoiceCapabilities])
model_discoverer: Callable[..., ModelDiscoveryResult] = discover_models
assert_type(model_discoverer, Callable[..., ModelDiscoveryResult])
assert_type(LexiconCapabilities, type[LexiconCapabilities])
assert_type(LexiconDiscoveryResult, type[LexiconDiscoveryResult])
lexicon_discoverer: Callable[..., LexiconDiscoveryResult] = discover_lexicons
assert_type(lexicon_discoverer, Callable[..., LexiconDiscoveryResult])
assert_type(KokoroSynthesizer, type[KokoroSynthesizer])
assert_type(RenderedSegment, type[RenderedSegment])
assert_type(SynthesisConfig, type[SynthesisConfig])
assert_type(request, SynthesisSegment)
assert_type(config, SynthesisConfig)
