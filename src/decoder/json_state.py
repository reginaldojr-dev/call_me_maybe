"""JSON state machine and partial value validators for constrained decoding."""

from enum import Enum

from pydantic import BaseModel, Field

from src.errors.decoder_errors import JsonStateError, SchemaRulesError


class JsonState(str, Enum):
    """Structural states of a flat JSON object."""

    START = "START"
    EXPECT_KEY = "EXPECTED_KEY"
    EXPECT_COLON = "EXPECT_COLON"
    EXPECT_VALUE = "EXPECT_VALUE"
    EXPECT_COMMA_OR_END = "EXPECT_COMMA_OR_END"
    DONE = "DONE"


class ValidationState(str, Enum):
    """Validation status for complete or partial JSON tokens."""

    VALID = "VALID"
    POSSIBLE = "POSSIBLE"
    INVALID = "INVALID"


class DecoderState(BaseModel):
    """Immutable state tracking JSON structure, keys, and token buffer."""

    json_state: JsonState
    current_key: str | None = None
    used_keys: set[str] = Field(default_factory=set)
    buffer: str = ""

    def set_json_state(self, state: JsonState) -> "DecoderState":
        """Return a copy of the state with an updated JsonState."""
        return self.model_copy(update={"json_state": state})

    def append_buffer(self, text: str) -> "DecoderState":
        """Return a copy of the state with text appended to the buffer."""
        return self.model_copy(update={"buffer": self.buffer + text})

    def clear_buffer(self) -> "DecoderState":
        """Return a copy of the state with an empty buffer."""
        return self.model_copy(update={"buffer": ""})

    def set_current_key(self, key: str | None) -> "DecoderState":
        """Return a copy of the state with an updated current_key."""
        return self.model_copy(update={"current_key": key})

    def mark_key_as_used(self, key: str) -> "DecoderState":
        """Return a copy of the state with key added to used_keys."""
        new_used_keys = set(self.used_keys)
        new_used_keys.add(key)
        return self.model_copy(update={"used_keys": new_used_keys})


def _finalize_value(
    state: DecoderState,
    next_json_state: JsonState,
) -> DecoderState:
    """Mark current_key as used, reset buffer/key, and set next JsonState."""
    updated = state
    if state.current_key is not None:
        updated = updated.mark_key_as_used(state.current_key)
    return (
        updated.set_current_key(None)
        .clear_buffer()
        .set_json_state(next_json_state)
    )


def next_state(
    state: DecoderState,
    token: str,
    expected_type: str | None = None,
) -> DecoderState:
    """Advance the JSON state machine given the
    current state and next token."""
    if token == "":
        raise JsonStateError("Empty token is not allowed")

    clean_token = token.lstrip(" ") if state.buffer == "" else token
    if clean_token == "":
        raise JsonStateError("Whitespace-only token is not allowed")

    if state.json_state == JsonState.START and clean_token == "{":
        return state.set_json_state(JsonState.EXPECT_KEY)

    if state.json_state == JsonState.EXPECT_KEY:
        if state.buffer == "" and not state.used_keys and clean_token == "}":
            return state.set_json_state(JsonState.DONE)
        candidate = state.buffer + clean_token
        key_status = validate_string(candidate)
        if key_status == ValidationState.VALID:
            extracted_key = candidate[1:-1]
            return (
                state.set_current_key(extracted_key)
                .clear_buffer()
                .set_json_state(JsonState.EXPECT_COLON)
            )
        if key_status == ValidationState.POSSIBLE:
            return state.append_buffer(clean_token)

    if state.json_state == JsonState.EXPECT_COLON and clean_token == ":":
        return state.set_json_state(JsonState.EXPECT_VALUE)

    if state.json_state == JsonState.EXPECT_VALUE:
        if expected_type is None:
            raise JsonStateError(
                "Expected type is None while expecting a value"
            )

        if expected_type == "number" and state.buffer != "":
            stripped = token.lstrip(" ")
            if (
                stripped in (",", "}")
                and validate_number(state.buffer) == ValidationState.VALID
            ):
                target = (
                    JsonState.EXPECT_KEY
                    if stripped == ","
                    else JsonState.DONE
                )
                return _finalize_value(state, target)

        candidate = state.buffer + clean_token
        status = validate_value(candidate, expected_type)

        if status == ValidationState.VALID:
            if expected_type == "number":
                return state.append_buffer(clean_token)
            return _finalize_value(state, JsonState.EXPECT_COMMA_OR_END)

        if status == ValidationState.POSSIBLE:
            return state.append_buffer(clean_token)

    if state.json_state == JsonState.EXPECT_COMMA_OR_END:
        if clean_token == ",":
            return state.set_json_state(JsonState.EXPECT_KEY)
        if clean_token == "}":
            return state.set_json_state(JsonState.DONE)

    raise JsonStateError(
        f"Unexpected token {token!r} in state {state.json_state.value}"
    )


def validate_string(value: str) -> ValidationState:
    """Validate whether value is a complete or partial JSON string."""
    if not value.startswith('"'):
        return ValidationState.INVALID
    if "\n" in value or "\r" in value or "\t" in value:
        return ValidationState.INVALID

    escaped = False
    quote_positions: list[int] = []
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            quote_positions.append(index)

    if escaped:
        return ValidationState.POSSIBLE
    if len(quote_positions) == 1:
        return ValidationState.POSSIBLE
    if len(quote_positions) == 2 and quote_positions[1] == len(value) - 1:
        return ValidationState.VALID
    return ValidationState.INVALID


def validate_number(value: str) -> ValidationState:
    """Validate whether value is a complete or partial JSON number."""
    if value == "-":
        return ValidationState.POSSIBLE

    number = value[1:] if value.startswith("-") else value

    if not number or number.count(".") > 1:
        return ValidationState.INVALID

    if number.endswith("."):
        prefix = number[:-1]
        if prefix.isdigit():
            return ValidationState.POSSIBLE
        return ValidationState.INVALID

    if "." in number:
        number = number.replace(".", "", 1)

    if number.isdigit():
        return ValidationState.VALID

    return ValidationState.INVALID


def validate_boolean(value: str) -> ValidationState:
    """Validate whether value is a complete or partial JSON boolean."""
    if value in ("true", "false"):
        return ValidationState.VALID
    if value in ("t", "tr", "tru", "f", "fa", "fal", "fals"):
        return ValidationState.POSSIBLE
    return ValidationState.INVALID


def validate_value(value: str, expected_type: str) -> ValidationState:
    """Validate a candidate value against the expected schema type."""
    if expected_type == "string":
        return validate_string(value)
    if expected_type == "number":
        return validate_number(value)
    if expected_type == "boolean":
        return validate_boolean(value)
    raise SchemaRulesError(f"Unsupported parameter type: {expected_type}")
