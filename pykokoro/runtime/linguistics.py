"""Pipeline-owned linguistic resources and provider-neutral analyses."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, cast

from ..spacy_models import SpacyModelSize, resolve_spacy_model
from .language_plan import LanguageRun


@dataclass(frozen=True, slots=True)
class TokenAnnotation:
    """Provider-neutral lexical annotation aligned to one input text."""

    start: int
    end: int
    text: str | None = None
    pos: str | None = None
    tag: str | None = None
    lemma: str | None = None
    language: str | None = None


@dataclass(slots=True)
class LinguisticAnalysis:
    """One analysis and its provider document for a single text run."""

    language: str
    text: str
    doc: object | None
    annotations: tuple[TokenAnnotation, ...]
    model_name: str | None = None


@dataclass(slots=True)
class PreparedRunAnalysis:
    """Request-local Pass-A or Pass-B analysis associated with a language run."""

    run: LanguageRun
    text: str
    doc: object | None
    annotations: tuple[TokenAnnotation, ...]
    model_name: str | None = None


@dataclass(slots=True)
class LinguisticRequestState:
    """Ephemeral linguistic state for one document preparation request."""

    source_plan: tuple[LanguageRun, ...] = ()
    prepared_plan: tuple[LanguageRun, ...] = ()
    source_analysis: list[PreparedRunAnalysis] = field(default_factory=list)
    prepared_analysis: list[PreparedRunAnalysis] = field(default_factory=list)

    def release_source_docs(self) -> None:
        """Release Pass-A documents after preparation has consumed them."""
        for analysis in self.source_analysis:
            analysis.doc = None

    def release_docs(self) -> None:
        """Release provider documents while preserving lightweight annotations."""
        for analysis in (*self.source_analysis, *self.prepared_analysis):
            analysis.doc = None


class LinguisticResourcePool:
    """Cache loaded spaCy Language pipelines by concrete resource identity."""

    def __init__(self) -> None:
        self._spacy: dict[tuple[str, str], object] = {}

    def get_spacy_pipeline(
        self,
        *,
        language: str,
        model: str | None = None,
        model_size: SpacyModelSize | None = None,
        require: bool = False,
    ) -> object | None:
        """Load or reuse a local spaCy pipeline without downloading models."""
        model_name = self._resolve_model_name(language, model, model_size, require=require)
        if model_name is None:
            return None
        key = (language.lower().replace("_", "-"), model_name)
        if key in self._spacy:
            return self._spacy[key]
        try:
            import spacy
        except ImportError as exc:
            if require:
                raise RuntimeError("spaCy is required but is not installed") from exc
            return None
        try:
            pipeline = spacy.load(model_name)
        except (OSError, ImportError, ValueError) as exc:
            if require:
                raise RuntimeError(f"Requested spaCy model {model_name!r} is unavailable") from exc
            return None
        self._spacy[key] = pipeline
        return pipeline

    def analyze(
        self,
        text: str,
        *,
        language: str,
        model: str | None = None,
        model_size: SpacyModelSize | None = None,
        require: bool = False,
    ) -> LinguisticAnalysis | None:
        """Analyze text with a cached local pipeline, or return no-model fallback."""
        pipeline = self.get_spacy_pipeline(
            language=language,
            model=model,
            model_size=model_size,
            require=require,
        )
        if pipeline is None:
            return None
        doc = cast(Any, pipeline)(text)
        annotations = tuple(
            TokenAnnotation(
                start=int(token.idx),
                end=int(token.idx + len(token.text)),
                text=str(token.text),
                pos=getattr(token, "pos_", None) or None,
                tag=getattr(token, "tag_", None) or None,
                lemma=getattr(token, "lemma_", None) or None,
                language=language,
            )
            for token in doc
        )
        return LinguisticAnalysis(
            language=language,
            text=text,
            doc=doc,
            annotations=annotations,
            model_name=self._pipeline_name(pipeline),
        )

    def clear(self) -> None:
        """Release all cached language pipelines."""
        self._spacy.clear()

    @staticmethod
    def _pipeline_name(pipeline: object) -> str | None:
        meta = getattr(pipeline, "meta", None)
        if isinstance(meta, dict):
            name = meta.get("name")
            version = meta.get("version")
            if name and version:
                return f"{name}-{version}"
            if name:
                return str(name)
        return getattr(pipeline, "lang", None)

    @staticmethod
    def _resolve_model_name(
        language: str,
        model: str | None,
        model_size: SpacyModelSize | None,
        *,
        require: bool,
    ) -> str | None:
        if model:
            return model
        if model_size:
            return resolve_spacy_model(language, size=model_size)
        for size in ("trf", "lg", "md", "sm"):
            candidate = resolve_spacy_model(language, size=size)
            if importlib.util.find_spec(candidate) is not None:
                return candidate
        if require:
            raise RuntimeError(f"No installed spaCy model is available for {language!r}")
        return None


__all__ = [
    "LinguisticAnalysis",
    "LinguisticRequestState",
    "LinguisticResourcePool",
    "PreparedRunAnalysis",
    "TokenAnnotation",
]
