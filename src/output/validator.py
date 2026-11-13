"""Output validation using Pydantic and function schema rules."""

from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from src.decoder.schema_rules import SchemaRules
from src.errors.output_errors import OutputError


class FunctionCallOutput(BaseModel):
    """Pydantic model enforcing the exact top-level output structure."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]


def validate_output(
    result: dict[str, Any],
    schema_rules: SchemaRules,
) -> dict[str, Any]:
    """Validate that a function call dictionary matches the required schema."""
    if not isinstance(result, dict):
        raise OutputError("Output must be a dictionary.")

    try:
        validated_model = FunctionCallOutput.model_validate(result)
    except ValidationError as error:
        raise OutputError(f"Invalid output structure: {error}") from error

    parameters = validated_model.parameters
    valid_keys = schema_rules.get_all_keys()
    for key in parameters.keys():
        if key not in valid_keys:
            raise OutputError(f"Unexpected parameter key: {key}.")

    all_parameters = schema_rules.parameters
    for key, expected_type in all_parameters.items():
        if key not in parameters:
            raise OutputError(f"Missing parameter: {key}.")

        value = parameters[key]
        mapping_dict: dict[str, type] = {
            "string": str,
            "number": float,
            "boolean": bool,
        }
        if expected_type not in mapping_dict:
            raise OutputError(
                f"Unknown expected type: {expected_type} for key: {key}."
            )
        if not isinstance(value, mapping_dict[expected_type]) or (
            expected_type == "number" and isinstance(value, bool)
        ):
            raise OutputError(
                f"Parameter '{key}' is expected to be of type "
                f"'{expected_type}', "
                f"but got value '{value}' of type '{type(value).__name__}'."
            )

    return result
