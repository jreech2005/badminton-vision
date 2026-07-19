"""Typed exceptions raised by every fail-loud gate in the pipeline."""


class BadmintonVisionError(Exception):
    """Family root so the CLI edge can catch this project's errors as one class."""


class DataContractError(BadmintonVisionError):
    """Input data violates the ingestion contract (shape, columns, counts, lockdown)."""


class LabelMapError(BadmintonVisionError):
    """A stroke label is not covered by the versioned label map."""


class LeakGuardError(BadmintonVisionError):
    """Code requested data that would leak outcome or future information."""


class ValidationGateError(BadmintonVisionError):
    """A record failed a rulebook, physics, or consistency gate."""
