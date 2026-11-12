import json
from typing import Any

from src.decoder.json_state import DecoderState, JsonState, next_state
from src.decoder.schema_rules import SchemaRules
from src.decoder.token_filter import (mask_logits,
                                      select_next_token)


def generate_parameters(
        llm,
        vocab: dict[str, int],
        schema_rules: SchemaRules,
        prompt: str
) -> dict[str, Any]:
    state = DecoderState(json_state=JsonState.START)
    generated_text = ""
    while state.json_state != JsonState.DONE:
        logits = llm(prompt + generated_text)
        masked_logits = mask_logits(state, schema_rules, logits, vocab)
        next_token_id = select_next_token(masked_logits)
        next_token = [
            token
            for token, token_id in vocab.items()
            if token_id == next_token_id
        ][0]
        generated_text += next_token
        state = next_state(
            state, next_token, schema_rules.get_expected_type(state)
        )

    return json.loads(generated_text)
