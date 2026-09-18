from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

from ...constants import ESPEAK_ONLY_LANGUAGES, MAX_PHONEME_LENGTH, SUPPORTED_LANGUAGES
from ...language_detection import ResolvedLanguageDetection, resolve_language_detection
from ...lexicon_data import create_g2p_with_lexphon_retry
from ...runtime.cache import cache_from_dir, make_g2p_key
from ...runtime.spans import slice_boundaries, slice_spans
from ...spacy_models import SpacyModelSize, make_spacy_model_request, spacy_selection_metadata
from ...ssmd_config import resolve_document_voice
from ...types import (
    AnnotationSpan,
    G2PAlignmentToken,
    PhonemeSegment,
    TraceEvent,
    _model_span_token_count,
)
from ..protocols import DocumentResult, G2PAdapter

if TYPE_CHECKING:
    from kokorog2p.base import G2PBase

    from ...generation_config import GenerationConfig
    from ...pipeline_config import PipelineConfig
    from ...types import Segment, Trace


_LANGUAGE_ALIASES = {
    "en": "en-us",
    "fr": "fr-fr",
    "cmn": "zh",
}


def canonicalize_g2p_language(language: str) -> str:
    """Normalize and validate one document or span language for G2P.

    This is the renderer-local equivalent of language_plan.canonicalize_language.
    Use this for G2P-layer language normalization, not plan-level normalization.
    """
    if not isinstance(language, str):
        raise TypeError(f"language must be a string, got {type(language)!r}")
    normalized = language.strip().lower().replace("_", "-")
    if not normalized:
        raise ValueError("language must not be empty")
    normalized = _LANGUAGE_ALIASES.get(normalized, normalized)
    supported = set(SUPPORTED_LANGUAGES) | set(ESPEAK_ONLY_LANGUAGES)
    base_language = normalized.split("-", 1)[0]
    if normalized not in supported and base_language not in supported:
        raise ValueError(f"Unsupported language {language!r}")
    return normalized


def _context_token_value(token: Any, name: str, default: object) -> object:
    if isinstance(token, dict):
        return token.get(name, default)
    value = getattr(token, name, None)
    if value is not None:
        return value
    metadata = getattr(token, "meta", None)
    if isinstance(metadata, dict) and name in metadata:
        return metadata[name]
    return default


def _context_token_int(token: Any, name: str) -> int | None:
    value = _context_token_value(token, name, None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


@dataclass(frozen=True)
class _CachedContextToken:
    text: str
    phonemes: str
    whitespace: str
    char_start: int | None
    char_end: int | None
    model_token_count: int | None
    model_span_token_count: int | None


@dataclass(frozen=True)
class _ContextModelGeometry:
    speech_count: int
    span_count: int


@dataclass(frozen=True)
class _CachedContextResult:
    phonemes: str
    ids: tuple[int, ...]
    tokens: tuple[_CachedContextToken, ...]


@dataclass(frozen=True, slots=True)
class _G2PTokenAnnotation:
    """Segment-local token annotation for G2P processing.

    Coordinates are relative to the segment's start, not the full document.
    These are "sliced" coordinates after subtracting segment.char_start.
    Do not confuse with UtterPlan's plan-spoken coordinates.
    """

    start: int
    end: int
    text: str | None = None
    pos: str | None = None
    tag: str | None = None
    lemma: str | None = None
    language: str | None = None


class KokoroG2PAdapter(G2PAdapter):
    _cache_schema = 11

    def __init__(self) -> None:
        self._g2p: ModuleType | None = None
        self._g2p_instances: dict[tuple[tuple[str, object], ...], G2PBase] = {}

        self._context_cache: OrderedDict[tuple[object, ...], _CachedContextResult] = OrderedDict()
        self._context_cache_max_entries = 256

    def _load(self) -> ModuleType:
        if self._g2p is not None:
            return self._g2p
        try:
            import kokorog2p
        except ModuleNotFoundError as exc:
            if exc.name == "kokorog2p":
                raise RuntimeError(
                    "kokorog2p is not installed; install the pykokoro runtime dependencies."
                ) from exc
            raise RuntimeError(
                "kokorog2p is installed but failed to import one of its dependencies."
            ) from exc
        except Exception as exc:
            raise RuntimeError("kokorog2p is installed but failed to initialize.") from exc
        self._g2p = kokorog2p
        return self._g2p

    def phonemize(
        self,
        segments: list[Segment],
        doc: DocumentResult,
        cfg: PipelineConfig,
        trace: Trace,
    ) -> list[PhonemeSegment]:
        g2p = self._load()
        cache = cache_from_dir(cfg.cache_dir)
        generation = cfg.generation
        model_version = self._get_model_version(cfg)
        profile, resolved_backend, phoneme_postprocess = self._resolve_frontend_contract(cfg)
        frontend = profile.frontend if profile is not None else None
        resolved_routing = resolve_language_detection(cfg.language_detection, doc.header)
        if generation.is_phonemes:
            effective_routing = ResolvedLanguageDetection(
                source=resolved_routing.source,
            )
        else:
            effective_routing = resolved_routing
        doc.metadata["language_detection"] = {
            "mode": resolved_routing.mode,
            "languages": list(resolved_routing.languages),
            "source": resolved_routing.source,
        }
        route_diagnostics: list[dict[str, Any]] = []
        doc.metadata["g2p_language_routes"] = route_diagnostics
        out: list[PhonemeSegment] = []

        for segment in segments:
            span_warnings: list[str] = []
            span_list = slice_spans(
                doc.annotation_spans,
                segment.char_start,
                segment.char_end,
                overlap_mode=cfg.overlap_mode,
                warnings=span_warnings,
            )
            seg_boundaries = slice_boundaries(
                doc.boundary_events,
                segment.char_start,
                segment.char_end,
                doc_end=len(doc.clean_text),
            )
            phoneme_override = self._resolve_phoneme_override(
                doc.annotation_spans,
                segment,
                span_warnings,
            )
            overrides = self._prepared_overrides(doc.annotation_spans, segment, span_warnings)
            plan_directives = segment.meta.get("plan_directives")
            if isinstance(plan_directives, dict):
                plan_phonemes = plan_directives.get("ph")
                if phoneme_override is None and isinstance(plan_phonemes, str):
                    phoneme_override = plan_phonemes
                    overrides = [
                        AnnotationSpan(0, segment.char_end - segment.char_start, plan_directives)
                    ]
            trace.warnings.extend(span_warnings)
            annotations = self._prepared_annotations(doc, segment)

            lang = segment.meta.get("language") or generation.lang
            ssmd_metadata: dict[str, str] = {}
            if isinstance(plan_directives, dict):
                self._apply_span_metadata(plan_directives, ssmd_metadata)
            header_bindings = doc.header.get("voice_bindings", {})
            if not isinstance(header_bindings, Mapping):
                header_bindings = {}
            voice_reference = ssmd_metadata.get("voice_reference") or ssmd_metadata.get("voice")
            if voice_reference:
                resolution = resolve_document_voice(
                    voice_reference,
                    provider=cfg.ssmd.provider,
                    api_bindings=cfg.ssmd.voice_bindings,
                    header_bindings=header_bindings,
                )
                ssmd_metadata["voice_reference"] = resolution.reference
                ssmd_metadata["voice_name"] = resolution.target
                ssmd_metadata["voice_source"] = resolution.source
            for span in span_list:
                self._apply_span_metadata(span.attrs, ssmd_metadata)
            if not isinstance(lang, str) or not lang:
                raise ValueError("A language is required for each G2P segment")

            cache_key = make_g2p_key(
                text=segment.text,
                lang=lang,
                is_phonemes=generation.is_phonemes,
                tokenizer_config=asdict(cfg.tokenizer_config) if cfg.tokenizer_config else None,
                annotations=annotations,
                phoneme_override=phoneme_override,
                kokorog2p_version=getattr(g2p, "__version__", None),
                model_quality=cfg.model_quality,
                model_source=cfg.model_source,
                model_variant=cfg.model_variant,
                frontend=frontend,
                g2p_backend=resolved_backend,
                phoneme_postprocess=phoneme_postprocess,
                input_mode="prepared",
                preparation_backend=doc.preparation.backend if doc.preparation else "spokenform",
                preparation_version=doc.preparation.version if doc.preparation else None,
                language_routing=effective_routing.as_routing(),
            )
            cached = cache.get(cache_key)
            cached_payload = self._read_cache_payload(cached)
            alignment_tokens: list[G2PAlignmentToken] = []
            language_routes: list[dict[str, Any]] = []
            result_warnings: list[str] = []
            if cached_payload is not None:
                phonemes, tokens, alignment_tokens, result_warnings, language_routes = (
                    cached_payload
                )
                trace.warnings.extend(result_warnings)
            else:
                if cached is not None:
                    cache.delete(cache_key)
                if generation.is_phonemes:
                    phonemes = segment.text
                    tokens = g2p.phonemes_to_ids(phonemes, model=model_version)
                else:
                    g2p_instance = self._get_g2p_instance(lang, cfg)
                    self._record_selection(doc, lang, cfg, g2p_instance)

                    def resolve_language_g2p(candidate: str) -> Any:
                        instance = self._get_g2p_instance(candidate, cfg)
                        self._record_selection(doc, candidate, cfg, instance)
                        return instance

                    result = self._phonemize_prepared(
                        g2p,
                        segment.text,
                        lang,
                        overrides,
                        annotations,
                        g2p_instance,
                        g2p_resolver=resolve_language_g2p,
                        language_routing=effective_routing.as_routing(),
                        target_model=model_version,
                    )
                    phonemes = str(
                        getattr(result, "phonemes", None) or getattr(result, "phoneme", "")
                    )
                    tokens = getattr(result, "ids", None) or getattr(result, "token_ids", [])
                    alignment_tokens = self._normalize_alignment_tokens(
                        getattr(result, "tokens", []), segment, g2p, model_version
                    )
                    result_warnings = [str(warning) for warning in getattr(result, "warnings", [])]
                    language_routes = [
                        asdict(route) for route in getattr(result, "language_routes", [])
                    ]
                    if result_warnings:
                        trace.warnings.extend(result_warnings)
            global_routes = self._globalize_language_routes(language_routes, segment.char_start)
            route_diagnostics.extend(global_routes)
            if cfg.model_variant in {"de-thorsten", "de-crane"}:
                phonemes, tokens, alignment_tokens = self._normalize_german_short_u_payload(
                    str(phonemes), alignment_tokens, g2p, model_version
                )
            tokens = list(tokens)
            cache.set(
                cache_key,
                {
                    "schema": self._cache_schema,
                    "g2p_input_mode": "prepared",
                    "preparation_backend": doc.preparation.backend
                    if doc.preparation
                    else "spokenform",
                    "preparation_version": doc.preparation.version if doc.preparation else None,
                    "phonemes": str(phonemes),
                    "tokens": tokens,
                    "alignment_tokens": [token.to_dict() for token in alignment_tokens],
                    "warnings": result_warnings,
                    "language_routes": language_routes,
                    "effective_language_routing": effective_routing.as_routing(),
                },
            )

            plan_pauses = doc.metadata.get("utterplan_pauses", {})
            if isinstance(plan_pauses, dict) and segment.id in plan_pauses:
                pause_before, pause_after = plan_pauses[segment.id]
            else:
                pause_before, pause_after = self._resolve_pauses(seg_boundaries, generation)
            if any(
                boundary.attrs.get("deterministic_pause_boundary") == "true"
                for boundary in seg_boundaries
            ):
                ssmd_metadata["deterministic_pause_boundary"] = "true"
            phoneme_batches = self._split_phoneme_batches(
                g2p, str(phonemes), list(tokens), model_version, generation
            )
            total_batches = len(phoneme_batches)
            batch_alignments = self._partition_alignment_tokens(
                alignment_tokens, [len(batch_tokens) for _, batch_tokens, _ in phoneme_batches]
            )
            for idx, (batch_phonemes, batch_tokens, batch_pause_after) in enumerate(
                phoneme_batches, start=0
            ):
                batch_pause_before = pause_before if idx == 0 else 0.0
                if idx == total_batches - 1:
                    batch_pause_after = max(pause_after, batch_pause_after)
                phoneme_id = idx
                phoneme_segment_id = f"{segment.id}_ph{phoneme_id}"
                out.append(
                    PhonemeSegment(
                        id=phoneme_segment_id,
                        segment_id=segment.id,
                        phoneme_id=phoneme_id,
                        text=segment.text,
                        phonemes=str(batch_phonemes),
                        tokens=list(batch_tokens),
                        alignment_tokens=batch_alignments[idx],
                        lang=lang,
                        char_start=segment.char_start,
                        char_end=segment.char_end,
                        paragraph_idx=segment.paragraph_idx,
                        sentence_idx=segment.sentence_idx,
                        clause_idx=segment.clause_idx,
                        pause_before=batch_pause_before,
                        pause_after=batch_pause_after,
                        ssmd_metadata=ssmd_metadata or None,
                        voice_name=ssmd_metadata.get("voice_name"),
                        voice_language=ssmd_metadata.get("voice_language"),
                        voice_gender=ssmd_metadata.get("voice_gender"),
                        voice_variant=ssmd_metadata.get("voice_variant"),
                    )
                )

        trace.events.append(
            TraceEvent(
                stage="g2p",
                name="language_routing",
                ms=0.0,
                details={
                    "mode": effective_routing.mode,
                    "languages": list(effective_routing.languages),
                    "route_count": len(route_diagnostics),
                    "mixed_token_count": sum(
                        1 for route in route_diagnostics if len(route.get("fragments", ())) > 1
                    ),
                },
            )
        )
        return out

    @staticmethod
    def _prepared_overrides(
        spans: list[AnnotationSpan],
        segment: Segment,
        warnings: list[str],
    ) -> list[AnnotationSpan]:
        overrides: list[AnnotationSpan] = []
        segment_language = segment.meta.get("language")
        canonical_segment_language = (
            canonicalize_g2p_language(segment_language)
            if isinstance(segment_language, str) and segment_language
            else None
        )
        for span in spans:
            attrs = {
                key: value
                for key, value in span.attrs.items()
                if key in {"ph", "phonemes", "lang", "language"}
            }
            if "lang" not in attrs and "language" in attrs:
                attrs["lang"] = attrs.pop("language")
            if (
                "ph" not in attrs
                and "phonemes" not in attrs
                and canonical_segment_language is not None
                and span.char_start == segment.char_start
                and span.char_end == segment.char_end
                and "lang" in attrs
                and canonicalize_g2p_language(attrs["lang"]) == canonical_segment_language
            ):
                continue
            if (
                not attrs
                or span.char_end <= segment.char_start
                or span.char_start >= segment.char_end
            ):
                continue
            if "ph" in attrs or "phonemes" in attrs:
                if span.char_start != segment.char_start or span.char_end != segment.char_end:
                    continue
                overrides.append(AnnotationSpan(0, segment.char_end - segment.char_start, attrs))
            else:
                start = max(span.char_start, segment.char_start) - segment.char_start
                end = min(span.char_end, segment.char_end) - segment.char_start
                if start < end:
                    overrides.append(AnnotationSpan(start, end, attrs))
        return overrides

    @staticmethod
    def _prepared_annotations(doc: DocumentResult, segment: Segment) -> list[Any]:
        state = getattr(doc, "linguistic_state", None)
        if state is None:
            tokens = doc.metadata.get("utterplan_tokens")
            indices = segment.meta.get("plan_token_indices")
            if not isinstance(tokens, tuple) or not isinstance(indices, tuple):
                return []

            plan_annotations: list[Any] = []
            for index in indices:
                if not isinstance(index, int) or index < 0 or index >= len(tokens):
                    raise ValueError(
                        f"Plan segment {segment.id!r} references invalid token {index!r}"
                    )
                item = tokens[index]
                start = max(item.spoken_start, segment.char_start) - segment.char_start
                end = min(item.spoken_end, segment.char_end) - segment.char_start
                if start < end:
                    plan_annotations.append(
                        _G2PTokenAnnotation(
                            start=start,
                            end=end,
                            text=item.text,
                            pos=item.pos,
                            tag=item.tag,
                            lemma=item.lemma,
                            language=item.language,
                        )
                    )
            return plan_annotations
        annotations: list[Any] = []
        for analysis in getattr(state, "prepared_analysis", ()):
            analysis_language = canonicalize_g2p_language(analysis.run.language)
            run = analysis.run
            if run.char_start > segment.char_start or run.char_end < segment.char_end:
                continue
            for item in analysis.annotations:
                if item.end <= segment.char_start or item.start >= segment.char_end:
                    continue

                start = max(item.start, segment.char_start) - segment.char_start
                end = min(item.end, segment.char_end) - segment.char_start
                item_language = canonicalize_g2p_language(item.language) if item.language else None
                forwarded_language = (
                    item.language
                    if item_language is not None and item_language != analysis_language
                    else None
                )
                annotations.append(
                    _G2PTokenAnnotation(
                        start=start,
                        end=end,
                        text=item.text,
                        pos=item.pos,
                        tag=item.tag,
                        lemma=item.lemma,
                        language=forwarded_language,
                    )
                )
            break
        return annotations

    @staticmethod
    def _phonemize_prepared(
        g2p: Any,
        text: str,
        language: str,
        overrides: list[AnnotationSpan],
        annotations: list[Any],
        g2p_instance: Any,
        *,
        g2p_resolver: Any = None,
        language_routing: dict[str, object] | None = None,
        target_model: str | None = None,
    ) -> Any:
        prepared = getattr(g2p, "phonemize_prepared", None)
        if callable(prepared):
            return prepared(
                text,
                language=language,
                overrides=overrides or None,
                annotations=annotations or None,
                return_phonemes=True,
                return_ids=True,
                alignment="span",
                g2p=g2p_instance,
                overlap="split",
                g2p_resolver=g2p_resolver,
                language_routing=language_routing,
                target_model=target_model,
            )
        # Compatibility for test doubles and old installations; released PyKokoro
        # dependencies always provide the prepared entry point.
        return g2p.phonemize(
            text,
            language=language,
            return_phonemes=True,
            return_ids=True,
            g2p=g2p_instance,
        )

    @classmethod
    def _read_cache_payload(
        cls, cached: Any
    ) -> tuple[str, list[int], list[G2PAlignmentToken], list[str], list[dict[str, Any]]] | None:
        if (
            not isinstance(cached, dict)
            or cached.get("schema") != cls._cache_schema
            or cached.get("g2p_input_mode") != "prepared"
            or cached.get("preparation_backend") != "spokenform"
        ):
            return None
        phonemes = cached.get("phonemes")
        tokens = cached.get("tokens")
        raw_alignment = cached.get("alignment_tokens")
        warnings = cached.get("warnings")
        raw_routes = cached.get("language_routes")
        if not isinstance(phonemes, str) or not isinstance(tokens, list):
            return None
        if not all(isinstance(token, int) and not isinstance(token, bool) for token in tokens):
            return None
        if (
            not isinstance(raw_alignment, list)
            or not isinstance(warnings, list)
            or not isinstance(raw_routes, list)
            or not all(isinstance(route, dict) for route in raw_routes)
        ):
            return None
        if not all(isinstance(warning, str) for warning in warnings):
            return None
        alignment: list[G2PAlignmentToken] = []
        for item in raw_alignment:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                return None
            phoneme_text = item.get("phonemes")
            whitespace = item.get("whitespace", "")
            if not isinstance(phoneme_text, str) or not isinstance(whitespace, str):
                return None
            provenance_keys = (
                "pronunciation_source",
                "pronunciation_provider",
                "pronunciation_lexicon_id",
                "pronunciation_requested_language",
                "pronunciation_source_ipa",
            )
            if any(
                item.get(key) is not None and not isinstance(item.get(key), str)
                for key in provenance_keys
            ):
                return None
            language_markers = item.get("pronunciation_language_markers")
            if language_markers is not None and (
                not isinstance(language_markers, list)
                or not all(isinstance(marker, dict) for marker in language_markers)
            ):
                return None
            alignment.append(
                G2PAlignmentToken(
                    text=item["text"],
                    phonemes=phoneme_text,
                    whitespace=whitespace,
                    char_start=item.get("char_start"),
                    char_end=item.get("char_end"),
                    model_token_count=item.get("model_token_count"),
                    model_span_token_count=item.get("model_span_token_count"),
                    pronunciation_source=item.get("pronunciation_source"),
                    pronunciation_provider=item.get("pronunciation_provider"),
                    pronunciation_lexicon_id=item.get("pronunciation_lexicon_id"),
                    pronunciation_requested_language=item.get("pronunciation_requested_language"),
                    pronunciation_source_ipa=item.get("pronunciation_source_ipa"),
                    pronunciation_language_markers=(
                        [dict(marker) for marker in language_markers]
                        if language_markers is not None
                        else None
                    ),
                )
            )
        return phonemes, tokens, alignment, warnings, [dict(route) for route in raw_routes]

    @staticmethod
    def _globalize_language_routes(
        routes: list[dict[str, Any]], offset: int
    ) -> list[dict[str, Any]]:
        global_routes: list[dict[str, Any]] = []
        for raw_route in routes:
            route = dict(raw_route)
            for key in ("char_start", "char_end"):
                if isinstance(route.get(key), int):
                    route[key] += offset
            fragments = []
            for raw_fragment in route.get("fragments", []):
                if not isinstance(raw_fragment, dict):
                    continue
                fragment = dict(raw_fragment)
                for key in ("char_start", "char_end"):
                    if isinstance(fragment.get(key), int):
                        fragment[key] += offset
                fragments.append(fragment)
            route["fragments"] = fragments
            global_routes.append(route)
        return global_routes

    @staticmethod
    def _normalize_german_short_u_payload(
        phonemes: str,
        alignment_tokens: list[G2PAlignmentToken],
        g2p: Any,
        model_version: str,
    ) -> tuple[str, list[int], list[G2PAlignmentToken]]:
        """Apply the model-vocabulary compatibility transform to every route."""
        cleaned_phonemes = phonemes.replace("ʏ", "y")
        tokens = list(g2p.phonemes_to_ids(cleaned_phonemes, model=model_version))
        cleaned_alignment = [
            replace(
                token,
                phonemes=token.phonemes.replace("ʏ", "y"),
                model_token_count=len(
                    g2p.phonemes_to_ids(token.phonemes.replace("ʏ", "y"), model=model_version)
                ),
            )
            for token in alignment_tokens
        ]
        return cleaned_phonemes, tokens, cleaned_alignment

    @staticmethod
    def _normalize_alignment_tokens(
        raw_tokens: Any, segment: Segment, g2p: Any, model_version: str
    ) -> list[G2PAlignmentToken]:
        if not isinstance(raw_tokens, (list, tuple)):
            return []
        normalized: list[G2PAlignmentToken] = []
        cursor = 0
        for raw_token in raw_tokens:
            if isinstance(raw_token, dict):
                text = raw_token.get("text")
                phonemes = raw_token.get("phonemes")
                whitespace = raw_token.get("whitespace") or ""
                raw_start = raw_token.get("char_start")
                raw_end = raw_token.get("char_end")
            else:
                metadata = getattr(raw_token, "meta", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                text = getattr(raw_token, "text", None)
                phonemes = getattr(raw_token, "phonemes", None) or metadata.get("phonemes")
                whitespace = (
                    getattr(raw_token, "whitespace", None) or metadata.get("whitespace") or ""
                )
                raw_start = getattr(raw_token, "char_start", None)
                raw_end = getattr(raw_token, "char_end", None)
            if isinstance(raw_token, dict):
                metadata = raw_token
            if not isinstance(text, str) or not isinstance(phonemes, str):
                continue
            if not isinstance(whitespace, str):
                whitespace = str(whitespace)
            pronunciation_source = metadata.get("pronunciation_source")
            if not isinstance(pronunciation_source, str):
                pronunciation_source = None
            pronunciation_provider = metadata.get("pronunciation_provider")
            if not isinstance(pronunciation_provider, str):
                pronunciation_provider = None
            pronunciation_lexicon_id = metadata.get("pronunciation_lexicon_id")
            if not isinstance(pronunciation_lexicon_id, str):
                pronunciation_lexicon_id = None
            pronunciation_requested_language = metadata.get("pronunciation_requested_language")
            if not isinstance(pronunciation_requested_language, str):
                pronunciation_requested_language = None
            pronunciation_source_ipa = metadata.get("pronunciation_source_ipa")
            if not isinstance(pronunciation_source_ipa, str):
                pronunciation_source_ipa = None
            raw_language_markers = metadata.get("pronunciation_language_markers")
            pronunciation_language_markers = (
                [dict(marker) for marker in raw_language_markers]
                if isinstance(raw_language_markers, list)
                and all(isinstance(marker, dict) for marker in raw_language_markers)
                else None
            )
            if (
                isinstance(raw_start, int)
                and not isinstance(raw_start, bool)
                and isinstance(raw_end, int)
                and not isinstance(raw_end, bool)
            ):
                if not 0 <= raw_start <= raw_end <= len(segment.text):
                    continue
                char_start = segment.char_start + raw_start
                char_end = segment.char_start + raw_end
                cursor = max(cursor, raw_end)
            else:
                found = segment.text.find(text, cursor) if text else cursor
                local_start = found if found >= 0 else cursor
                char_start = segment.char_start + local_start
                char_end = char_start + len(text)
                cursor = max(cursor, local_start + len(text))
            raw_model_span_count = _context_token_int(raw_token, "model_span_token_count")
            try:
                model_token_count = len(g2p.phonemes_to_ids(phonemes, model=model_version))
            except (AttributeError, TypeError, ValueError, RuntimeError):
                model_token_count = None
            normalized.append(
                G2PAlignmentToken(
                    text=text,
                    phonemes=phonemes,
                    whitespace=whitespace,
                    char_start=char_start,
                    char_end=char_end,
                    model_token_count=model_token_count,
                    model_span_token_count=raw_model_span_count,
                    pronunciation_source=pronunciation_source,
                    pronunciation_provider=pronunciation_provider,
                    pronunciation_lexicon_id=pronunciation_lexicon_id,
                    pronunciation_requested_language=pronunciation_requested_language,
                    pronunciation_source_ipa=pronunciation_source_ipa,
                    pronunciation_language_markers=pronunciation_language_markers,
                )
            )
        return normalized

    @staticmethod
    def _partition_alignment_tokens(
        alignment: list[G2PAlignmentToken], batch_counts: list[int]
    ) -> list[list[G2PAlignmentToken]]:
        if not alignment:
            return [[] for _ in batch_counts]
        if len(batch_counts) == 1:
            return [list(alignment)]
        if any(token.model_token_count is None for token in alignment):
            return [[] for _ in batch_counts]
        result: list[list[G2PAlignmentToken]] = []
        token_index = 0
        for batch_count in batch_counts:
            current: list[G2PAlignmentToken] = []
            consumed = 0
            while token_index < len(alignment) and consumed < batch_count:
                token = alignment[token_index]
                current.append(token)
                consumed += _model_span_token_count(token) or 0
                token_index += 1
            if consumed != batch_count:
                return [[] for _ in batch_counts]
            result.append(current)
        return result if token_index == len(alignment) else [[] for _ in batch_counts]

    @staticmethod
    def _resolve_frontend_contract(
        cfg: PipelineConfig,
    ) -> tuple[Any | None, str, str | None]:
        from ...model_profiles import get_model_profile
        from ...tokenizer import TokenizerConfig

        tokenizer_config = cfg.tokenizer_config or TokenizerConfig()
        profile = (
            get_model_profile(cfg.model_variant, cfg.model_source)
            if cfg.model_variant and cfg.model_source
            else None
        )
        backend = tokenizer_config.backend
        if profile is not None and profile.g2p_backend is not None:
            backend = profile.g2p_backend
        postprocess = (
            "german-short-u-to-y" if cfg.model_variant in {"de-thorsten", "de-crane"} else None
        )
        return profile, backend, postprocess

    @staticmethod
    def _g2p_kwargs_for_language(
        lang: str,
        cfg: PipelineConfig,
        backend: str,
        profile: Any | None,
        model_version: str,
    ) -> dict[str, Any]:
        from kokorog2p.language_codes import normalize_language_code
        from kokorog2p.lexicons import normalize_lexicon_selection

        from ...tokenizer import (
            TokenizerConfig,
            _default_lexicons_for_language,
            _effective_lexicons,
            _legacy_fallback_kwargs,
        )

        tokenizer_config = cfg.tokenizer_config or TokenizerConfig()
        kokorog2p_lang = SUPPORTED_LANGUAGES.get(lang, lang)
        default_language = normalize_language_code(cfg.generation.lang or lang)
        candidate_language = normalize_language_code(kokorog2p_lang)
        lexicons = _effective_lexicons(tokenizer_config)
        if lexicons is None and backend == "kokorog2p":
            lexicons = _default_lexicons_for_language(candidate_language)
        if candidate_language != default_language and lexicons is not None:
            try:
                lexicons = normalize_lexicon_selection(candidate_language, lexicons)
            except ValueError:
                lexicons = None
        candidate_backend = backend if candidate_language == default_language else "kokorog2p"
        request = make_spacy_model_request(
            model=tokenizer_config.spacy_model,
            size=tokenizer_config.spacy_model_size,
        )
        kwargs: dict[str, Any] = {
            "language": kokorog2p_lang,
            "version": "1.0" if model_version == "nabra-82m-v0.1" else model_version,
            "phoneme_quotes": "curly",
            **_legacy_fallback_kwargs(tokenizer_config.fallback),
            "use_spacy": tokenizer_config.use_spacy,
            "spacy_model": request.model,
            "spacy_model_size": request.size,
            "backend": candidate_backend,
            "lexicons": lexicons,
        }
        if profile is not None and profile.variant == "ar-nabra":
            kwargs["model_profile"] = "nabra-82m-v0.1"
        return kwargs

    def _get_g2p_instance(self, lang: str, cfg: PipelineConfig) -> G2PBase:
        from ...frontend_contracts import require_frontend
        from ...tokenizer import TokenizerConfig

        profile, backend, _ = self._resolve_frontend_contract(cfg)
        model_version = self._get_model_version(cfg, lang)
        if profile is not None:
            require_frontend(profile.variant, allow_experimental=cfg.allow_experimental_frontend)
        kwargs = self._g2p_kwargs_for_language(lang, cfg, backend, profile, model_version)
        tokenizer_config = cfg.tokenizer_config or TokenizerConfig()
        lexicon_data_policy = tokenizer_config.lexicon_data_policy
        cache_key = tuple(sorted(kwargs.items())) + (
            ("__pykokoro_lexicon_data_policy", lexicon_data_policy),
        )
        if cache_key in self._g2p_instances:
            return self._g2p_instances[cache_key]

        g2p_module = self._load()
        g2p_instance = create_g2p_with_lexphon_retry(
            g2p_module,
            language=kwargs["language"],
            config=tokenizer_config,
            kwargs=kwargs,
        )
        self._g2p_instances[cache_key] = g2p_instance
        return g2p_instance

    def phonemize_context(self, text: str, language: str, cfg: PipelineConfig) -> Any:
        """Phonemize synthetic short-sentence context with a bounded LRU cache."""
        cache_key = self._context_cache_key(text, language, cfg)
        cached = self._context_cache.get(cache_key)
        if cached is not None:
            self._context_cache.move_to_end(cache_key)
            return cached

        g2p_module = self._load()
        g2p_instance = self._get_g2p_instance(language, cfg)
        model_version = self._get_model_version(cfg, language)
        result = self._phonemize_prepared(
            g2p_module,
            text,
            language,
            [],
            [],
            g2p_instance,
            target_model=model_version,
        )
        normalized = self._normalize_context_result(
            result, g2p_module=g2p_module, model_version=model_version
        )
        self._context_cache[cache_key] = normalized
        self._context_cache.move_to_end(cache_key)
        while len(self._context_cache) > self._context_cache_max_entries:
            self._context_cache.popitem(last=False)
        return normalized

    def _context_cache_key(
        self, text: str, language: str, cfg: PipelineConfig
    ) -> tuple[object, ...]:
        profile, backend, _ = self._resolve_frontend_contract(cfg)
        model_version = self._get_model_version(cfg, language)
        kwargs = self._g2p_kwargs_for_language(language, cfg, backend, profile, model_version)
        tokenizer_config = cfg.tokenizer_config
        lexicon_data_policy = (
            tokenizer_config.lexicon_data_policy if tokenizer_config is not None else None
        )
        return (
            text,
            language,
            tuple(sorted(kwargs.items())),
            ("lexicon_data_policy", lexicon_data_policy),
            ("generation_lang", cfg.generation.lang),
            ("target_model", model_version),
            ("allow_experimental_frontend", cfg.allow_experimental_frontend),
        )

    @staticmethod
    def _explicit_context_model_geometry(
        tokens: list[_CachedContextToken],
        ids: tuple[int, ...],
    ) -> list[_ContextModelGeometry] | None:
        geometry: list[_ContextModelGeometry] = []
        for token in tokens:
            speech_count = getattr(token, "model_token_count", None)
            span_count = getattr(token, "model_span_token_count", None)
            if (
                not isinstance(speech_count, int)
                or isinstance(speech_count, bool)
                or speech_count < 0
                or not isinstance(span_count, int)
                or isinstance(span_count, bool)
                or span_count < speech_count
            ):
                return None
            geometry.append(_ContextModelGeometry(speech_count, span_count))
        if sum(item.span_count for item in geometry) != len(ids):
            return None
        return geometry

    @staticmethod
    def _derive_context_model_geometry(
        tokens: list[_CachedContextToken],
        ids: tuple[int, ...],
        g2p_module: Any | None,
        model_version: str | None,
    ) -> list[_ContextModelGeometry | None]:
        """Derive exact speech and span positions in one phrase-wide coordinate system."""
        tokenizer = getattr(g2p_module, "phonemes_to_ids", None)
        if callable(tokenizer) and tokens:
            prefix = ""
            previous_count = 0
            geometry: list[_ContextModelGeometry] = []
            try:
                for token in tokens:
                    speech_prefix = prefix + token.phonemes
                    speech_end_count = len(tokenizer(speech_prefix, model=model_version))
                    full_prefix = speech_prefix + token.whitespace
                    full_end_count = len(tokenizer(full_prefix, model=model_version))
                    speech_count = speech_end_count - previous_count
                    span_count = full_end_count - previous_count
                    if speech_count < 0 or span_count < speech_count:
                        raise ValueError("non-monotonic contextual model geometry")
                    geometry.append(_ContextModelGeometry(speech_count, span_count))
                    prefix = full_prefix
                    previous_count = full_end_count
            except (AttributeError, TypeError, ValueError, RuntimeError):
                geometry = []
            else:
                if previous_count == len(ids):
                    return cast(list[_ContextModelGeometry | None], geometry)
        explicit = KokoroG2PAdapter._explicit_context_model_geometry(tokens, ids)
        if explicit is not None:
            return cast(list[_ContextModelGeometry | None], explicit)
        return [None for _ in tokens]

    @staticmethod
    def _derive_context_model_spans(
        tokens: list[_CachedContextToken],
        ids: tuple[int, ...],
        g2p_module: Any | None,
        model_version: str | None,
    ) -> list[int | None]:
        """Compatibility wrapper returning only contextual total-span counts."""
        geometry = KokoroG2PAdapter._derive_context_model_geometry(
            tokens, ids, g2p_module, model_version
        )
        return [item.span_count if item is not None else None for item in geometry]

    @staticmethod
    def _normalize_context_result(
        result: Any,
        *,
        g2p_module: Any | None = None,
        model_version: str | None = None,
    ) -> _CachedContextResult:
        phonemes = getattr(result, "phonemes", None) or getattr(result, "phoneme", "")
        ids = getattr(result, "ids", None)
        if ids is None:
            ids = getattr(result, "token_ids", ())
        normalized_tokens: list[_CachedContextToken] = []
        for token in getattr(result, "tokens", ()) or ():
            token_phonemes = str(
                _context_token_value(token, "phonemes", None)
                or _context_token_value(token, "phoneme", "")
            )
            model_token_count = _context_token_int(token, "model_token_count")
            if model_token_count is not None and model_token_count < 0:
                model_token_count = None
            explicit_span = _context_token_int(token, "model_span_token_count")
            if explicit_span is not None and explicit_span < 0:
                explicit_span = None
            if model_token_count is None and token_phonemes and g2p_module is not None:
                try:
                    model_token_count = len(
                        g2p_module.phonemes_to_ids(token_phonemes, model=model_version)
                    )
                except (AttributeError, TypeError, ValueError, RuntimeError):
                    model_token_count = None
            normalized_tokens.append(
                _CachedContextToken(
                    text=str(_context_token_value(token, "text", "")),
                    phonemes=token_phonemes,
                    whitespace=str(_context_token_value(token, "whitespace", "") or ""),
                    char_start=_context_token_int(token, "char_start"),
                    char_end=_context_token_int(token, "char_end"),
                    model_token_count=model_token_count,
                    model_span_token_count=explicit_span,
                )
            )
        ids_tuple = tuple(int(token_id) for token_id in (ids or ()))
        geometry = KokoroG2PAdapter._derive_context_model_geometry(
            normalized_tokens, ids_tuple, g2p_module, model_version
        )
        normalized_tokens = [
            replace(
                token,
                model_token_count=item.speech_count if item is not None else None,
                model_span_token_count=item.span_count if item is not None else None,
            )
            for token, item in zip(normalized_tokens, geometry, strict=True)
        ]
        return _CachedContextResult(
            phonemes=str(phonemes),
            ids=ids_tuple,
            tokens=tuple(normalized_tokens),
        )

    def _record_selection(
        self, doc: DocumentResult, lang: str, cfg: PipelineConfig, g2p_instance: G2PBase
    ) -> None:
        from ...tokenizer import TokenizerConfig

        tokenizer_config = cfg.tokenizer_config or TokenizerConfig()
        request = make_spacy_model_request(
            model=tokenizer_config.spacy_model,
            size=tokenizer_config.spacy_model_size,
        )
        selected_model = getattr(g2p_instance, "spacy_model", None)
        selected_size = None
        if isinstance(selected_model, str):
            suffix = selected_model.rsplit("_", 1)[-1]
            if suffix in {"sm", "md", "lg", "trf"}:
                selected_size = cast(SpacyModelSize, suffix)
        models = doc.metadata.setdefault("spacy_models", {})
        if not isinstance(models, dict):
            models = {}
            doc.metadata["spacy_models"] = models
        g2p_models = models.setdefault("g2p", {})
        if not isinstance(g2p_models, dict):
            g2p_models = {}
            models["g2p"] = g2p_models
        g2p_models[lang] = spacy_selection_metadata(
            language=lang,
            request=request,
            selected_model=selected_model,
            selected_size=selected_size,
        )

    @staticmethod
    def _get_model_version(cfg: PipelineConfig, lang: str | None = None) -> str:
        from ...model_profiles import get_model_profile
        from ...pipeline_config import resolve_model_defaults

        if cfg.generation.lang is None and lang is not None:
            cfg = replace(cfg, generation=replace(cfg.generation, lang=lang))
        cfg = resolve_model_defaults(cfg)
        assert cfg.model_variant is not None
        assert cfg.model_source is not None
        return get_model_profile(cfg.model_variant, cfg.model_source).tokenizer_vocab_version

    def _split_phoneme_batches(
        self,
        g2p_module: Any,
        phonemes: str,
        tokens: list[int],
        model_version: str,
        generation: GenerationConfig,
    ) -> list[tuple[str, list[int], float]]:
        if not tokens:
            return [(phonemes, tokens, 0.0)]
        if len(tokens) <= MAX_PHONEME_LENGTH:
            return [(phonemes, tokens, 0.0)]
        if generation.pause_mode == "auto":
            clause_batches = self._split_phoneme_batches_by_clause(
                g2p_module, phonemes, model_version
            )
            if clause_batches:
                last_idx = len(clause_batches) - 1
                return [
                    (
                        batch_phonemes,
                        batch_tokens,
                        generation.pause_clause if idx < last_idx else 0.0,
                    )
                    for idx, (batch_phonemes, batch_tokens) in enumerate(clause_batches)
                ]
        batches: list[tuple[str, list[int], float]] = []
        for start in range(0, len(tokens), MAX_PHONEME_LENGTH):
            chunk_tokens = tokens[start : start + MAX_PHONEME_LENGTH]
            chunk_phonemes = g2p_module.ids_to_phonemes(chunk_tokens, model=model_version)
            batches.append((chunk_phonemes, chunk_tokens, 0.0))
        return batches

    def _split_phoneme_batches_by_clause(
        self, g2p_module: Any, phonemes: str, model_version: str
    ) -> list[tuple[str, list[int]]]:
        clause_boundaries = [match.end() for match in re.finditer(r"[,;:]", phonemes)]
        if not clause_boundaries:
            return []
        parts: list[str] = []
        start = 0
        for end in clause_boundaries:
            parts.append(phonemes[start:end])
            start = end
        if start < len(phonemes):
            parts.append(phonemes[start:])

        batches: list[tuple[str, list[int]]] = []
        current = ""
        for part in parts:
            candidate = f"{current}{part}" if current else part
            candidate_tokens = g2p_module.phonemes_to_ids(candidate, model=model_version)
            if len(candidate_tokens) > MAX_PHONEME_LENGTH:
                if not current:
                    return []
                current_tokens = g2p_module.phonemes_to_ids(current, model=model_version)
                if len(current_tokens) > MAX_PHONEME_LENGTH:
                    return []
                batches.append((current, current_tokens))
                current = part
            else:
                current = candidate

        if current:
            current_tokens = g2p_module.phonemes_to_ids(current, model=model_version)
            if len(current_tokens) > MAX_PHONEME_LENGTH:
                return []
            batches.append((current, current_tokens))

        if len(batches) <= 1:
            return []
        return batches

    def _apply_span_metadata(self, attrs: dict[str, str], metadata: dict[str, str]) -> None:
        if not attrs:
            return
        if "voice" in attrs:
            metadata["voice"] = attrs["voice"]
            metadata["voice_name"] = attrs["voice"]
        if "voice_name" in attrs:
            metadata["voice"] = attrs["voice_name"]
            metadata["voice_name"] = attrs["voice_name"]
        if "voice_reference" in attrs:
            metadata["voice_reference"] = attrs["voice_reference"]
        if "voice_source" in attrs:
            metadata["voice_source"] = attrs["voice_source"]
        for key in ("voice_language", "voice_gender", "voice_variant"):
            if key in attrs:
                metadata[key] = attrs[key]
        for key in (
            "emphasis",
            "audio_src",
            "audio_alt_text",
            "audio_clip_begin",
            "audio_clip_end",
            "audio_speed",
            "audio_repeat_dur",
            "audio_sound_level",
            "audio_repeat_count",
        ):
            if key in attrs:
                metadata[key] = attrs[key]
        if "markers" in attrs:
            metadata["markers"] = attrs["markers"]
        if "prosody_rate" in attrs:
            metadata["prosody_rate"] = attrs["prosody_rate"]
        if "rate" in attrs:
            metadata.setdefault("prosody_rate", attrs["rate"])
        if "prosody_pitch" in attrs:
            metadata["prosody_pitch"] = attrs["prosody_pitch"]
        if "pitch" in attrs:
            metadata.setdefault("prosody_pitch", attrs["pitch"])
        if "prosody_volume" in attrs:
            metadata["prosody_volume"] = attrs["prosody_volume"]
        if "volume" in attrs:
            metadata.setdefault("prosody_volume", attrs["volume"])
        if "lang" in attrs:
            metadata["lang"] = attrs["lang"]
        if "ph" in attrs:
            metadata["ph"] = attrs["ph"]
        if "phonemes" in attrs:
            metadata.setdefault("ph", attrs["phonemes"])

    def _resolve_phoneme_override(
        self,
        spans: list[AnnotationSpan],
        segment: Segment,
        warnings: list[str],
    ) -> str | None:
        phoneme_override = None
        for span in spans:
            if "ph" not in span.attrs and "phonemes" not in span.attrs:
                continue
            if span.char_start == segment.char_start and span.char_end == segment.char_end:
                override_value = span.attrs.get("ph") or span.attrs.get("phonemes")
                if phoneme_override and override_value != phoneme_override:
                    warnings.append(
                        "Multiple phoneme override spans match segment "
                        f"{segment.char_start}:{segment.char_end}."
                    )
                phoneme_override = override_value
            elif span.char_end > segment.char_start and span.char_start < segment.char_end:
                warnings.append(
                    "Skipped phoneme override span at "
                    f"{span.char_start}:{span.char_end} for segment "
                    f"{segment.char_start}:{segment.char_end}."
                )
        return phoneme_override

    def _resolve_pauses(self, boundaries, generation):
        """Read already-reduced document boundaries without re-deciding semantics."""
        pause_before = 0.0
        pause_after = 0.0
        for boundary in boundaries:
            if boundary.kind != "pause":
                continue
            duration = boundary.duration_s
            if duration is None:
                strength = boundary.attrs.get("strength")
                duration = {
                    "c": generation.pause_clause,
                    "s": generation.pause_sentence,
                    "p": generation.pause_paragraph,
                    "w": 0.15,
                    "n": 0.0,
                }.get(strength)
            if duration is None:
                continue
            if boundary.pos == 0:
                pause_before = max(pause_before, duration)
            else:
                pause_after = max(pause_after, duration)
        return pause_before, pause_after
