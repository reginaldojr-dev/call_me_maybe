from pydantic import BaseModel, TypeAdapter


class PromptDefinition(BaseModel):
    prompt: str


def validate_prompts(data: object) -> list[PromptDefinition]:
    return TypeAdapter(
        list[PromptDefinition]
        ).validate_python(data)
