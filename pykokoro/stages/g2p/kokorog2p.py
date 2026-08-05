from __future__ import annotations

import re
from dataclasses import asdict
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

from ...constants import MAX_PHONEME_LENGTH, SUPPORTED_LANGUAGES
from ...runtime.cache import cache_from_dir, make_g2p_key
from ...runtime.spans import slice_boundaries, slice_spans
from ...spacy_models import SpacyModelSize, make_spacy_model_request, spacy_selection_metadata
from ...types import PhonemeSegment
from ..protocols import DocumentResult, G2PAdapter

if TYPE_CHECKING:
    from kokorog2p.base import G2PBase

    from ...generation_config import GenerationConfig
    from ...pipeline_config import PipelineConfig
    from ...types import AnnotationSpan, Segment, Trace


class KokoroG2PAdapter(G2PAdapter):
    _cache_schema = 1

    def __init__(self) -> None:
        self._g2p: ModuleType | None = None
        self._g2p_instances: dict[tuple[tuple[str, object], ...], G2PBase] = {}

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
            trace.warnings.extend(span_warnings)

            lang = generation.lang
            ssmd_metadata: dict[str, str] = {}
            for span in span_list:
                span_lang = span.attrs.get("lang")
                if span_lang:
                    lang = span_lang
                self._apply_span_metadata(span.attrs, ssmd_metadata)

            cache_key = make_g2p_key(
                text=segment.text,
                lang=lang,
                is_phonemes=generation.is_phonemes,
                tokenizer_config=asdict(cfg.tokenizer_config) if cfg.tokenizer_config else None,
                phoneme_override=phoneme_override,
                kokorog2p_version=getattr(g2p, "__version__", None),
                model_quality=cfg.model_quality,
                model_source=cfg.model_source,
                model_variant=cfg.model_variant,
            )
            cached = cache.get(cache_key)
            cached_payload = self._read_cache_payload(cached)
            if cached_payload is not None:
                phonemes, tokens, cached_warnings = cached_payload
                trace.warnings.extend(cached_warnings)
            else:
                if cached is not None:
                    cache.delete(cache_key)
                result_warnings: list[str] = []
                if generation.is_phonemes:
                    phonemes = segment.text
                    tokens = g2p.phonemes_to_ids(phonemes, model=model_version)
                elif phoneme_override:
                    phonemes = phoneme_override
                    tokens = g2p.phonemes_to_ids(phonemes, model=model_version)
                else:
                    g2p_instance = self._get_g2p_instance(lang, cfg)
                    self._record_selection(doc, lang, cfg, g2p_instance)
                    result = g2p.phonemize(
                        segment.text,
                        language=lang,
                        return_phonemes=True,
                        return_ids=True,
                        g2p=g2p_instance,
                    )
                    phonemes = str(
                        getattr(result, "phonemes", None) or getattr(result, "phoneme", "")
                    )
                    tokens = getattr(result, "ids", None) or getattr(result, "token_ids", [])
                    result_warnings = [str(warning) for warning in getattr(result, "warnings", [])]
                    if result_warnings:
                        trace.warnings.extend(result_warnings)
                cache.set(
                    cache_key,
                    {
                        "schema": self._cache_schema,
                        "phonemes": str(phonemes),
                        "tokens": list(tokens),
                        "warnings": result_warnings,
                    },
                )

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

        return out

    @classmethod
    def _read_cache_payload(cls, cached: Any) -> tuple[str, list[int], list[str]] | None:
        if not isinstance(cached, dict) or cached.get("schema") != cls._cache_schema:
            return None
        phonemes = cached.get("phonemes")
        tokens = cached.get("tokens")
        warnings = cached.get("warnings")
        if not isinstance(phonemes, str) or not isinstance(tokens, list):
            return None
        if not all(isinstance(token, int) and not isinstance(token, bool) for token in tokens):
            return None
        if not isinstance(warnings, list) or not all(
            isinstance(warning, str) for warning in warnings
        ):
            return None
        return phonemes, tokens, warnings

    def _get_g2p_instance(self, lang: str, cfg: PipelineConfig) -> G2PBase:
        from ...tokenizer import TokenizerConfig

        tokenizer_config = cfg.tokenizer_config or TokenizerConfig()
        kokorog2p_lang = SUPPORTED_LANGUAGES.get(lang, lang)
        version = self._get_model_version(cfg)
        request = make_spacy_model_request(
            model=tokenizer_config.spacy_model,
            size=tokenizer_config.spacy_model_size,
        )

        kwargs: dict[str, Any] = {
            "language": kokorog2p_lang,
            "version": version,
            "phoneme_quotes": "curly",
            "use_goruut_fallback": tokenizer_config.use_goruut_fallback,
            "use_espeak_fallback": tokenizer_config.use_espeak_fallback,
            "use_spacy": tokenizer_config.use_spacy,
            "spacy_model": request.model,
            "spacy_model_size": request.size,
            "backend": tokenizer_config.backend,
            "load_gold": tokenizer_config.load_gold,
            "load_silver": tokenizer_config.load_silver,
        }

        cache_key = tuple(sorted(kwargs.items()))
        if cache_key in self._g2p_instances:
            return self._g2p_instances[cache_key]

        g2p_module = self._load()
        g2p_instance = g2p_module.get_g2p(**kwargs)
        self._g2p_instances[cache_key] = g2p_instance
        return g2p_instance

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
    def _get_model_version(cfg: PipelineConfig) -> str:
        from ...model_profiles import get_model_profile
        from ...pipeline_config import resolve_model_defaults

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
