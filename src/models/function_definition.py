from pydantic import BaseModel, TypeAdapter


class ParameterDefinition(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ParameterDefinition


def validate_functions(data: object) -> list[FunctionDefinition]:
    return TypeAdapter(list[FunctionDefinition]).validate_python(data)
