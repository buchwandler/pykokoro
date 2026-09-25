class PyKokoroError(Exception):
    """Base exception for PyKokoro."""


KokoroError = PyKokoroError


class SynthesisError(PyKokoroError):
    """Base exception for synthesis failures."""


class ConfigurationError(PyKokoroError, ValueError):
    """Invalid or inconsistent configuration."""


class InvalidRequestError(SynthesisError, ValueError):
    """A synthesis request contains invalid caller-provided data."""


class EmptyTextError(InvalidRequestError):
    """Synthesis text is empty or whitespace-only."""


class InvalidLanguageError(InvalidRequestError):
    """The requested language is unsupported."""


class InvalidVoiceError(InvalidRequestError):
    """The requested voice is invalid or unavailable."""


class InvalidModelError(InvalidRequestError):
    """The requested model is invalid or unavailable."""


class InvalidPronunciationError(InvalidRequestError):
    """Pronunciation overrides are invalid or ambiguous."""


class InvalidLinguisticTokensError(InvalidRequestError):
    """Linguistic token annotations are invalid or ambiguous."""


class SynthesisInputTooLongError(SynthesisError, ValueError):
    """Tokenized request exceeds model capacity."""

    def __init__(
        self,
        message: str | None = None,
        *,
        text_length: int | None = None,
        token_count: int | None = None,
        max_tokens: int | None = None,
        model_id: str | None = None,
    ) -> None:
        self.text_length = text_length
        self.token_count = token_count
        self.max_tokens = max_tokens
        self.model_id = model_id
        if message is None:
            message = (
                f"request has {token_count} model tokens, model maximum is {max_tokens} "
                f"for {model_id!r}; split the request externally"
            )
        super().__init__(message)


class UnsupportedFeatureError(SynthesisError):
    """Requested engine feature is unsupported by a model or runtime profile."""


CapabilityError = UnsupportedFeatureError


class SynthesisStateError(SynthesisError, RuntimeError):
    """Synthesis engine is not in a state that accepts requests."""


class AlignmentError(SynthesisError):
    """Alignment failed in a way that cannot be recovered."""


class BackendError(SynthesisError):
    """Synthesis backend failed (ONNX runtime, model I/O, etc.)."""
