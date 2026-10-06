import json
from src.errors.input_errors import InvalidJsonError


def parse_json(content: str) -> object:
    try:
        return json.loads(content)
    except json.JSONDecodeError as error:
        raise InvalidJsonError(
            f"Error decoding JSON: {error}"
            ) from error
