class TeleOCRError(Exception):
    """Base exception for TeleOCR failures."""


class PipelineError(TeleOCRError):
    """A document could not be converted into a complete result."""


class LayoutParseError(PipelineError):
    """The model returned an invalid layout record."""


class BlockCropError(PipelineError):
    """A detected block could not be cropped safely."""


class PostProcessError(PipelineError):
    """Recognized content could not be structurally normalized."""
