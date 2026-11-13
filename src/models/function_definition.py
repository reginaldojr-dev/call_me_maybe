"""Pydantic models for validating function definitions from JSON input."""

from typing import Any

from pydantic import BaseModel, TypeAdapter


class ParameterDefinition(BaseModel):
    """Represent the schema type of a single function parameter or return."""

    type: str


class FunctionDefinition(BaseModel):
    """Represent a callable function specification and its parameter schema."""

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ParameterDefinition


def validate_functions(data: Any) -> list[FunctionDefinition]:
    """Validate raw JSON data into a list of FunctionDefinition objects."""
    adapter = TypeAdapter(list[FunctionDefinition])
    return adapter.validate_python(data)
