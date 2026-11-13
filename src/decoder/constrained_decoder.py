"""Constrained decoding loop for generating schema-compliant JSON arguments."""

import json
from typing import Any

from llm_sdk.llm_sdk import Small_LLM_Model

from src.decoder.json_state import DecoderState, JsonState, next_state
from src.decoder.schema_rules import SchemaRules
from src.decoder.token_filter import mask_logits, select_next_token
from src.errors.decoder_errors import JsonStateError
from src.llm.model import encode_text, get_next_token_logits
from src.llm.vocab import invert_vocab
from src.models.function_definition import FunctionDefinition


def build_parameters_prompt(
    function: FunctionDefinition,
    prompt: str,
) -> str:
    """Build the prompt guiding the LLM to extract function arguments."""
    params_desc = ", ".join(
        f'"{name}": {param.type}'
        for name, param in function.parameters.items()
    )
    return (
        "Extract the arguments for the selected function as a JSON object.\n"
        f"Function: {function.name}\n"
        f"Description: {function.description}\n"
        f"Required JSON keys and types: {{{params_desc}}}\n"
        f"User request: {prompt}\n"
        "JSON:"
    )


def generate_parameters(
    llm: Small_LLM_Model,
    vocab: dict[str, int],
    schema_rules: SchemaRules,
    prompt: str,
    max_tokens: int = 128,
) -> dict[str, Any]:
    """Generate a valid JSON dictionary of parameters token by token."""
    state = DecoderState(json_state=JsonState.START)
    generated_text = ""
    id_to_token = invert_vocab(vocab)

    for _ in range(max_tokens):
        if state.json_state == JsonState.DONE:
            break

        input_ids = encode_text(llm, prompt + generated_text)
        logits = get_next_token_logits(llm, input_ids)
        masked_logits = mask_logits(state, schema_rules, logits, vocab)
        next_token_id = select_next_token(masked_logits)
        next_token = id_to_token[next_token_id]

        expected_type = (
            schema_rules.get_expected_type(state)
            if state.json_state == JsonState.EXPECT_VALUE
            else None
        )
        state = next_state(state, next_token, expected_type)
        generated_text += next_token

    if state.json_state != JsonState.DONE:
        raise JsonStateError("Maximum token limit reached before JSON end.")

    parsed: dict[str, Any] = json.loads(generated_text)
    for key, param_type in schema_rules.parameters.items():
        if (
            param_type == "number"
            and key in parsed
            and isinstance(parsed[key], (int, float))
            and not isinstance(parsed[key], bool)
        ):
            parsed[key] = float(parsed[key])

    return parsed
