from src.decoder.json_state import DecoderState, JsonState, next_state
from src.decoder.schema_rules import SchemaRules
from src.errors.decoder_errors import JsonStateError


def is_token_allowed(
    state: DecoderState, token: str, schema_rules: SchemaRules
) -> bool:
    if state.json_state == JsonState.EXPECT_VALUE:
        expected_type = schema_rules.get_expected_type(state)
    else:
        expected_type = None

    try:
        new_state = next_state(state, token, expected_type)
    except JsonStateError:
        return False

    if new_state.json_state == JsonState.EXPECT_COLON:
        if new_state.current_key is None:
            return False
        if (
            not schema_rules.is_valid_key(new_state.current_key)
            or new_state.current_key in new_state.used_keys
        ):
            return False

    if new_state.json_state == JsonState.DONE:
        missing_parameters = schema_rules.get_missing_parameters(new_state)
        if missing_parameters:
            return False

    return True


def get_allowed_tokens(
    state: DecoderState, schema_rules: SchemaRules, vocab: dict[str, int]
) -> list[int]:
    allowed_tokens = []
    for token, token_id in vocab.items():
        if is_token_allowed(state, token, schema_rules):
            allowed_tokens.append(token_id)
    return allowed_tokens


def mask_logits(
    state: DecoderState,
    schema_rules: SchemaRules,
    logits: list[float],
    vocab: dict[str, int]
) -> list[float]:
    allowed_token_ids = set(get_allowed_tokens(state, schema_rules, vocab))
    masked_logits = [
        logit if token_id in allowed_token_ids else float("-inf")
        for token_id, logit in enumerate(logits)
    ]
    return masked_logits


def select_next_token(logits: list[float]) -> int:
    max_logit = max(logits)
    return logits.index(max_logit)
