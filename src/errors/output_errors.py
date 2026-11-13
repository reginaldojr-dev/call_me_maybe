"""Custom exception class for output validation and file writing errors."""


class OutputError(Exception):
    """Raised when output validation fails
        or the output file cannot be saved."""
