"""Pydantic model for validating natural language test prompts."""

from typing import Any

from pydantic import BaseModel, TypeAdapter


class PromptDefinition(BaseModel):
    """Represent a single natural language prompt entry."""

    prompt: str


def validate_prompts(data: Any) -> list[PromptDefinition]:
    """Validate raw JSON data into a list of PromptDefinition objects."""
    adapter = TypeAdapter(list[PromptDefinition])
    return adapter.validate_python(data)
