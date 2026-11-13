"""Custom exception classes for input file reading and JSON parsing."""


class InputFileError(Exception):
    """Raised when an input file cannot be found or read."""


class InvalidJsonError(Exception):
    """Raised when an input file contains malformed JSON."""
