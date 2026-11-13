"""Token filtering and logit masking for constrained JSON decoding."""

from src.decoder.json_state import DecoderState, JsonState, next_state
from src.decoder.schema_rules import SchemaRules
from src.errors.decoder_errors import JsonStateError
from src.llm.model import get_best_token_id


def is_token_allowed(
    state: DecoderState,
    token: str,
    schema_rules: SchemaRules,
) -> bool:
    """Check if a token preserves valid JSON syntax and schema compliance."""
    if state.json_state == JsonState.EXPECT_VALUE:
        expected_type = schema_rules.get_expected_type(state)
    else:
        expected_type = None

    try:
        new_state = next_state(state, token, expected_type)
    except JsonStateError:
        return False

    if new_state.json_state == JsonState.EXPECT_KEY:
        missing = schema_rules.get_missing_parameters(new_state)
        if not missing:
            return False
        if new_state.buffer.startswith('"'):
            key_prefix = new_state.buffer[1:]
            if not any(key.startswith(key_prefix) for key in missing):
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
    state: DecoderState,
    schema_rules: SchemaRules,
    vocab: dict[str, int],
) -> list[int]:
    """Return all token IDs from vocab allowed in the current state."""
    allowed_tokens: list[int] = []
    for token, token_id in vocab.items():
        if is_token_allowed(state, token, schema_rules):
            allowed_tokens.append(token_id)
    return allowed_tokens


def mask_logits(
    state: DecoderState,
    schema_rules: SchemaRules,
    logits: list[float],
    vocab: dict[str, int],
) -> list[float]:
    """Set logits of invalid tokens to negative infinity."""
    allowed_token_ids = set(get_allowed_tokens(state, schema_rules, vocab))
    if not allowed_token_ids:
        raise JsonStateError("No valid tokens available for current state.")
    return [
        logit if token_id in allowed_token_ids else float("-inf")
        for token_id, logit in enumerate(logits)
    ]


def select_next_token(logits: list[float]) -> int:
    """Select the token ID with the highest logit score."""
    return get_best_token_id(logits)