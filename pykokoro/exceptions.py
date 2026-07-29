class KokoroError(Exception):
    """Base exception for pykokoro."""


class ConfigurationError(KokoroError):
    """Invalid or inconsistent configuration."""


class SSMDDocumentError(ConfigurationError):
    """Invalid SSMD 0.8 document metadata or renderer profile value."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        line: int | None = None,
        column: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.line = line
        self.column = column


class CapabilityError(KokoroError):
    """Requested SSMD/SSML feature is unsupported by selected backend/profile."""


class AlignmentError(KokoroError):
    """Annotation/token alignment failed in a way that can't be recovered."""


class BackendError(KokoroError):
    """Synthesis backend failed (onnx runtime, model I/O, etc.)."""
