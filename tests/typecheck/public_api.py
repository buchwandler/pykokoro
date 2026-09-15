from typing import assert_type

from pykokoro import (
    KokoroPipeline,
    PipelineConfig,
    PreparedAudioUnits,
    PreparedFrontend,
    discover_models,
    resolve_pipeline_config,
)

config = PipelineConfig()
resolved = resolve_pipeline_config(config)
assert_type(PipelineConfig, type[PipelineConfig])
assert_type(KokoroPipeline, type[KokoroPipeline])
assert_type(PreparedAudioUnits, type[PreparedAudioUnits])
assert_type(PreparedFrontend, type[PreparedFrontend])
assert_type(config, PipelineConfig)
assert_type(resolved, PipelineConfig)
