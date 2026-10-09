from models.function_definition import FunctionDefinition


def build_schema_rules(
    function: FunctionDefinition
) -> dict[str, str]:
    return {
        name: parameter.type
        for name, parameter in function.parameters.items()
    }
