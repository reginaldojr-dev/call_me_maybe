"""JSON parsing utilities with custom domain error handling."""

import json
from typing import Any

from src.errors.input_errors import InvalidJsonError


def parse_json(content: str) -> Any:
    """Parse a JSON string and return the resulting Python object."""
    try:
        return json.loads(content)
    except json.JSONDecodeError as error:
        raise InvalidJsonError(f"Invalid JSON content: {error}") from error
