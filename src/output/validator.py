from typing import Any
from src.decoder.schema_rules import SchemaRules
from src.errors.output_errors import OutputError


def validate_output(
    result: dict[str, Any], schema_rules: SchemaRules
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise OutputError("Output must be a dictionary.")

    required_keys = {"prompt", "name", "parameters"}
    if set(result.keys()) != required_keys:
        raise OutputError(
            f"Output keys are not as expected. Expected: {required_keys}, "
            f"but got: {set(result.keys())}."
        )

    if not isinstance(result["prompt"], str):
        raise OutputError("The 'prompt' key must be a string.")
    if not isinstance(result["name"], str):
        raise OutputError("The 'name' key must be a string.")
    if not isinstance(result["parameters"], dict):
        raise OutputError("The 'parameters' key must be a dictionary.")

    valid_keys = schema_rules.get_valid_keys()
    for key in result["parameters"].keys():
        if key not in valid_keys:
            raise OutputError(f"Unexpected parameter key: {key}.")

    all_parameters = schema_rules.parameters
    for key, expected_type in all_parameters.items():
        if key not in result["parameters"]:
            raise OutputError(f"Missing parameter: {key}.")

        value = result["parameters"][key]
        mapping_dict: dict[str, type | tuple[type, ...]] = {
            "string": str,
            "number": (int, float),
            "boolean": bool,
        }
        if expected_type not in mapping_dict:
            raise OutputError(f"Unknown expected type: "
                              f"{expected_type} for key: {key}.")
        if not isinstance(value, mapping_dict[expected_type]) or (
            expected_type == "number" and isinstance(value, bool)
        ):
            raise OutputError(
                f"Parameter '{key}' is expected to be of type "
                f"'{expected_type}', "
                f"but got value '{value}' of type '{type(value).__name__}'."
            )

    return result
