"""Custom exception class for LLM function selection failures."""


class FunctionSelectionError(Exception):
    """Raised when the LLM fails to select a valid function definition."""
