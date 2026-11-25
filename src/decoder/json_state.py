"""JSON state machine and partial value validators for constrained decoding."""

from enum import Enum
from typing import ClassVar

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


class JsonRules(BaseModel):
    """Static validation rules for JSON tokens.

    Constants are declared as ``ClassVar`` so Pydantic does not treat them
    as model fields.
    """

    SIMPLE_ESCAPES: ClassVar[frozenset[str]] = frozenset('"\\/bfnrt')
    HEX_DIGITS: ClassVar[frozenset[str]] = frozenset(
        "0123456789abcdefABCDEF"
    )
    ASCII_DIGITS: ClassVar[frozenset[str]] = frozenset("0123456789")


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
) -> DecoderState | None:
    """Advance the JSON state machine given the current state and next token.

    Returns a new ``DecoderState`` when the token is acceptable, or ``None``
    when the token would break JSON structure or the expected schema. Raising
    exceptions is reserved for programming errors (e.g. missing
    ``expected_type`` while a value is required).
    """
    if token == "":
        return None

    clean_token = token.lstrip(" ") if state.buffer == "" else token
    if clean_token == "":
        return None

    if state.json_state == JsonState.START:
        if clean_token == "{":
            return state.set_json_state(JsonState.EXPECT_KEY)
        return None

    if state.json_state == JsonState.EXPECT_KEY:
        if state.buffer == "" and not state.used_keys:
            if clean_token.strip() == "}":
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
        return None

    if state.json_state == JsonState.EXPECT_COLON:
        if clean_token.strip() == ":":
            return state.set_json_state(JsonState.EXPECT_VALUE)
        return None

    if state.json_state == JsonState.EXPECT_VALUE:
        if expected_type is None:
            raise JsonStateError(
                "Expected type is None while expecting a value"
            )

        if expected_type in ("number", "integer") and state.buffer != "":
            stripped = token.lstrip(" ")
            if (
                stripped in (",", "}")
                and validate_value(state.buffer, expected_type)
                == ValidationState.VALID
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
            if expected_type in ("number", "integer"):
                return state.append_buffer(clean_token)
            return _finalize_value(state, JsonState.EXPECT_COMMA_OR_END)

        if status == ValidationState.POSSIBLE:
            return state.append_buffer(clean_token)

        return None

    if state.json_state == JsonState.EXPECT_COMMA_OR_END:
        stripped = clean_token.strip()
        if stripped == ",":
            return state.set_json_state(JsonState.EXPECT_KEY)
        if stripped == "}":
            return state.set_json_state(JsonState.DONE)
        return None

    return None


def validate_string(value: str) -> ValidationState:
    """Validate whether value is a complete or partial JSON string.

    Returns:
        VALID: the value is a complete, well-formed JSON string.
        POSSIBLE: the value is a prefix of a valid JSON string.
        INVALID: the value can never become a valid JSON string.
    """
    if not value.startswith('"'):
        return ValidationState.INVALID

    i = 1
    n = len(value)

    while i < n:
        ch = value[i]

        if ord(ch) < 0x20:
            return ValidationState.INVALID

        if ch == '"':
            return (
                ValidationState.VALID
                if i == n - 1
                else ValidationState.INVALID
            )

        if ch == "\\":
            if i + 1 >= n:
                return ValidationState.POSSIBLE

            esc = value[i + 1]

            if esc in JsonRules.SIMPLE_ESCAPES:
                i += 2
                continue

            if esc == "u":
                hex_part = value[i + 2:i + 6]
                if len(hex_part) < 4:
                    if all(c in JsonRules.HEX_DIGITS for c in hex_part):
                        return ValidationState.POSSIBLE
                    return ValidationState.INVALID
                if not all(c in JsonRules.HEX_DIGITS for c in hex_part):
                    return ValidationState.INVALID
                i += 6
                continue

            return ValidationState.INVALID

        i += 1

    return ValidationState.POSSIBLE


def validate_number(value: str) -> ValidationState:
    """Validate whether value is a complete or partial JSON number.

    Accepts only ASCII digits, rejects leading zeros (except for ``0``
    itself), and supports partial prefixes used during constrained decoding.

    Returns:
        VALID: the value is a complete, well-formed JSON number.
        POSSIBLE: the value is a prefix of a valid JSON number.
        INVALID: the value can never become a valid JSON number.
    """
    if value == "":
        return ValidationState.INVALID
    if value == "-":
        return ValidationState.POSSIBLE

    body = value[1:] if value.startswith("-") else value
    if body == "":
        return ValidationState.INVALID

    if "." in body:
        int_part, _, frac_part = body.partition(".")
        if "." in frac_part:
            return ValidationState.INVALID
    else:
        int_part, frac_part = body, None

    if not int_part:
        return ValidationState.INVALID
    if int_part != "0" and int_part.startswith("0"):
        return ValidationState.INVALID
    if not all(c in JsonRules.ASCII_DIGITS for c in int_part):
        return ValidationState.INVALID

    if frac_part is None:
        return ValidationState.VALID
    if frac_part == "":
        return ValidationState.POSSIBLE
    if not all(c in JsonRules.ASCII_DIGITS for c in frac_part):
        return ValidationState.INVALID
    return ValidationState.VALID


def validate_integer(value: str) -> ValidationState:
    """Validate whether value is a complete or partial JSON integer.

    Accepts only ASCII digits with an optional leading minus sign.
    Rejects decimals, exponents, leading zeros and Unicode digits.

    Returns:
        VALID: the value is a complete, well-formed JSON integer.
        POSSIBLE: the value is a prefix of a valid JSON integer.
        INVALID: the value can never become a valid JSON integer.
    """
    if value == "":
        return ValidationState.INVALID
    if value == "-":
        return ValidationState.POSSIBLE

    body = value[1:] if value.startswith("-") else value
    if body == "":
        return ValidationState.INVALID

    if body != "0" and body.startswith("0"):
        return ValidationState.INVALID
    if not all(c in JsonRules.ASCII_DIGITS for c in body):
        return ValidationState.INVALID

    return ValidationState.VALID


def validate_boolean(value: str) -> ValidationState:
    """Validate whether value is a complete or partial JSON boolean."""
    if value in ("true", "false"):
        return ValidationState.VALID
    if value in ("t", "tr", "tru", "f", "fa", "fal", "fals"):
        return ValidationState.POSSIBLE
    return ValidationState.INVALID


def validate_null(value: str) -> ValidationState:
    """Validate whether value is a complete or partial JSON null."""
    if value == "null":
        return ValidationState.VALID
    if value in ("n", "nu", "nul"):
        return ValidationState.POSSIBLE
    return ValidationState.INVALID


def validate_value(value: str, expected_type: str) -> ValidationState:
    if expected_type == "string":
        return validate_string(value)
    if expected_type == "number":
        return validate_number(value)
    if expected_type == "integer":
        return validate_integer(value)
    if expected_type == "boolean":
        return validate_boolean(value)
    if expected_type == "null":
        return validate_null(value)
    raise SchemaRulesError(
        f"Unsupported parameter type: {expected_type}"
    )
