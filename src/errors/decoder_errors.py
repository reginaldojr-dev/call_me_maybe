"""Custom exception classes for JSON state machine and schema rule errors."""


class JsonStateError(Exception):
    """Raised when a token violates JSON syntax or state transitions."""


class SchemaRulesError(Exception):
    """Raised when a key or type violates the function schema rules."""
