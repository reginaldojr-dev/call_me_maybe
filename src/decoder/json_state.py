from enum import Enum
from pydantic import BaseModel, Field
from errors.decoder_errors import JsonStateError, SchemaRulesError


class JsonState(str, Enum):
    START = "START"
    EXPECT_KEY = "EXPECTED_KEY"
    EXPECT_COLON = "EXPECT_COLON"
    EXPECT_VALUE = "EXPECT_VALUE"
    EXPECT_COMMA_OR_END = "EXPECT_COMMA_OR_END"
    DONE = "DONE"


class ValidationState(str, Enum):
    VALID = "VALID"
    POSSIBLE = "POSSIBLE"
    INVALID = "INVALID"


class DecoderState(BaseModel):
    json_state: JsonState
    current_key: str | None = None
    used_keys: set[str] = Field(default_factory=set)
    buffer: str = ""

    def transition(
        self,
        json_state: JsonState,
        current_key: str | None = None,
        buffer: str = ""
    ) -> "DecoderState":
        

def next_state(
    state: DecoderState,
    token: str,
    expected_type: str | None = None
) -> DecoderState:
    if state.json_state == JsonState.START:
        if token == "{":
            return DecoderState(
                json_state=JsonState.EXPECT_KEY,
                current_key=state.current_key,
                used_keys=state.used_keys,
                buffer=""
            )
    raise JsonStateError(
        f"Unexpected token {token!r} in state {state.json_state}"
    )

    if state.json_state == JsonState.EXPECT_KEY and token.startswith('"'):
        return DecoderState(json_state=JsonState.EXPECT_COLON)

    if state.json_state == JsonState.EXPECT_COLON and token == ":":
        return DecoderState(json_state=JsonState.EXPECT_VALUE)

    if state.json_state == JsonState.EXPECT_VALUE:
        if token.startswith('"'):
            return DecoderState(json_state=JsonState.EXPECT_COMMA_OR_END)
        if token.lstrip("-").replace(".", "", 1).isdigit():
            return DecoderState(json_state=JsonState.EXPECT_COMMA_OR_END)
        if token in ("true", "false"):
            return DecoderState(json_state=JsonState.EXPECT_COMMA_OR_END)

    if state.json_state == JsonState.EXPECT_COMMA_OR_END:
        if token == ",":
            return DecoderState(json_state=JsonState.EXPECT_KEY)
        if token == "}":
            return DecoderState(json_state=JsonState.DONE)

    raise JsonStateError(
        f"Unexpected token {token!r} in state {state}"
        )


def validate_string(value: str) -> ValidationState:
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        return ValidationState.VALID
    if value.startswith('"') and not value.endswith('"'):
        return ValidationState.POSSIBLE
    return ValidationState.INVALID


def validate_number(value: str) -> ValidationState:
    if value == "-":
        return ValidationState.POSSIBLE

    number = value

    if number.startswith("-"):
        number = number[1:]

    if number.count(".") > 1:
        return ValidationState.INVALID

    if number.endswith("."):
        number = number[:-1]

        if number.isdigit():
            return ValidationState.POSSIBLE

        return ValidationState.INVALID

    if "." in number:
        number = number.replace(".", "", 1)

    if number.isdigit():
        return ValidationState.VALID

    return ValidationState.INVALID


def validate_boolean(value: str) -> ValidationState:
    if value == "true" or value == "false":
        return ValidationState.VALID
    if value == "t" or value == "tr" or value == "tru":
        return ValidationState.POSSIBLE
    if value == "f" or value == "fa" or value == "fal" or value == "fals":
        return ValidationState.POSSIBLE
    return ValidationState.INVALID


def validate_value(value: str, expected_type: str) -> ValidationState:
    if expected_type == "string":
        return validate_string(value)
    if expected_type == "number":
        return validate_number(value)
    if expected_type == "boolean":
        return validate_boolean(value)
    raise SchemaRulesError(f"Unsupported parameter type: {expected_type}")
