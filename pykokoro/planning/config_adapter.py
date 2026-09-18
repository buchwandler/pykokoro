from __future__ import annotations

from typing import Any

from utterplan import LinguisticsConfig, PauseConfig, PlannerConfig, SSMDConfig

from ..pipeline_config import PipelineConfig, require_document_language


def planner_config_from_pipeline(
    cfg: PipelineConfig,
    *,
    unit: str,
) -> PlannerConfig:
    """Map PyKokoro planning settings to an immutable UtterPlan config."""
    generation = cfg.generation
    tokenizer = cfg.tokenizer_config
    pause_mode = generation.pause_mode
    pauses = PauseConfig(
        mode=pause_mode,
        weak=generation.pause_parenthetical,
        clause=generation.pause_clause,
        sentence=generation.pause_sentence,
        paragraph=generation.pause_paragraph,
        parenthetical=generation.pause_parenthetical,
        voice_change=generation.pause_parenthetical,
    )
    linguistics = LinguisticsConfig(
        use_spacy=None if tokenizer is None else tokenizer.use_spacy,
        spacy_model=None if tokenizer is None else tokenizer.spacy_model,
        spacy_model_size=(
            None
            if tokenizer is None or tokenizer.spacy_model_size is None
            else tokenizer.spacy_model_size
        ),
        require_spacy=False if tokenizer is None else tokenizer.use_spacy is True,
    )
    pause_defaults: dict[str, Any] | None = None
    if cfg.ssmd.pause_defaults is not None:
        pause_defaults = {
            name: value
            for name in ("enabled", "sentence", "paragraph", "voice_change")
            if (value := getattr(cfg.ssmd.pause_defaults, name)) is not None
        }
    return PlannerConfig(
        language=require_document_language(cfg),
        document_format="ssmd",
        text_preparation="spokenform",
        unit=unit,
        pauses=pauses,
        linguistics=linguistics,
        ssmd=SSMDConfig(
            parse_header=cfg.ssmd.parse_header,
            strict_header=cfg.ssmd.strict_header,
            unknown_header=cfg.ssmd.unknown_header,
            pause_defaults=pause_defaults,
        ),
        overlap_mode=cfg.overlap_mode,
    )


PLANNING_OVERRIDE_KEYS = frozenset(
    {
        "lang",
        "generation",
        "ssmd",
        "overlap_mode",
        "unit",
        "document_format",
        "text_preparation",
        "pause_mode",
        "pause_weak",
        "pause_clause",
        "pause_sentence",
        "pause_paragraph",
    }
)
