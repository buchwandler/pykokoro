from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pykokoro.generation_config import GenerationConfig
from pykokoro.prepared_g2p import PreparedG2PAdapter
from pykokoro.synthesis_config import SynthesisConfig, resolve_synthesis_config
from pykokoro.synthesis_types import SynthesisSegment
from pykokoro.tokenizer import TokenizerConfig
from pykokoro.types import PhonemeSegment, Trace

LANGUAGE_TO_G2P = {"en-US": "en-us", "en-GB": "en-gb", "de-DE": "de-de"}


@dataclass(frozen=True, slots=True)
class FrontendVariant:
    """A named request-frontend configuration, independent of acoustic inference."""

    id: str = "default"
    language: str = "en-us"
    options: Mapping[str, Any] = field(default_factory=dict)

    def tokenizer_config(self, backend: str = "kokorog2p") -> TokenizerConfig:
        options = dict(self.options)
        options.setdefault("backend", backend)
        options["use_spacy"] = False
        return TokenizerConfig(**options)


@dataclass(frozen=True, slots=True)
class FrontendResult:
    text: str
    clean_text: str
    source_text: str | None
    segments: tuple[SynthesisSegment, ...]
    phoneme_segments: tuple[PhonemeSegment, ...]
    trace: Trace
    document_metadata: Mapping[str, Any]


class NoOnnxFrontend:
    """Prepare one request through PyKokoro's public G2P adapter without loading ONNX."""

    def __init__(
        self,
        locale: str,
        *,
        backend: str = "kokorog2p",
        variant: FrontendVariant | None = None,
    ) -> None:
        if locale not in LANGUAGE_TO_G2P:
            raise ValueError(f"unsupported locale: {locale!r}")
        language = LANGUAGE_TO_G2P[locale]
        selected = variant or FrontendVariant(language=language)
        if selected.language != language:
            raise ValueError("frontend variant language does not match locale")
        self.locale = locale
        self.variant = selected
        self.config = resolve_synthesis_config(
            SynthesisConfig(
                generation=GenerationConfig(lang=language),
                tokenizer_config=selected.tokenizer_config(backend),
                return_trace=True,
            ),
            language=language,
        )
        self.g2p = PreparedG2PAdapter()

    def run(self, text: str) -> FrontendResult:
        request = SynthesisSegment("hard-case", text, self.variant.language)
        prepared = self.g2p.phonemize(request, self.config)
        phoneme_segment = PhonemeSegment(
            id=f"{request.id}:phoneme:0",
            segment_id=request.id,
            phoneme_id=0,
            text=text,
            phonemes=prepared.phonemes,
            tokens=list(prepared.token_ids),
            lang=request.language,
            char_start=0,
            char_end=len(text),
            alignment_tokens=list(prepared.alignment_tokens),
        )
        return FrontendResult(
            text=text,
            clean_text=text,
            source_text=text,
            segments=(request,),
            phoneme_segments=(phoneme_segment,),
            trace=Trace(warnings=list(prepared.diagnostics)),
            document_metadata={},
        )

    def close(self) -> None:
        pass

    def __enter__(self) -> NoOnnxFrontend:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


PyKokoroFrontend = NoOnnxFrontend

__all__ = [
    "FrontendResult",
    "FrontendVariant",
    "LANGUAGE_TO_G2P",
    "NoOnnxFrontend",
    "PyKokoroFrontend",
]
