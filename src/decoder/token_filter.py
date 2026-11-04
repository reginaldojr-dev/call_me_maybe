from src.decoder.json_state import DecoderState, next_state
from src.decoder.schema_rules import SchemaRules

def is_token_allowed(
    state: DecoderState, token: str, schema_rules: SchemaRules
) -> bool:

    try:
        next_state(state, token, "VALID")
