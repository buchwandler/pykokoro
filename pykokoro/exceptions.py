class KokoroError(Exception):
    """Base exception for pykokoro."""


class ConfigurationError(KokoroError):
    """Invalid or inconsistent configuration."""


class CapabilityError(KokoroError):
    """Requested engine feature is unsupported by a model or runtime profile."""


class AlignmentError(KokoroError):
    """Alignment failed in a way that cannot be recovered."""


class BackendError(KokoroError):
    """Synthesis backend failed (onnx runtime, model I/O, etc.)."""
