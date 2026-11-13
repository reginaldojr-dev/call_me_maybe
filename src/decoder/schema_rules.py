"""Schema validation rules for function parameters during decoding."""

from pydantic import BaseModel

from src.decoder.json_state import DecoderState
from src.errors.decoder_errors import SchemaRulesError
from src.models.function_definition import FunctionDefinition


class SchemaRules(BaseModel):
    """Hold expected parameter names and types for a function."""

    parameters: dict[str, str]

    @classmethod
    def from_function(
        cls,
        function: FunctionDefinition,
    ) -> "SchemaRules":
        """Create SchemaRules from a FunctionDefinition instance."""
        return cls(
            parameters={
                name: parameter.type
                for name, parameter in function.parameters.items()
            }
        )

    def get_all_keys(self) -> set[str]:
        """Return the set of all valid parameter keys."""
        return set(self.parameters.keys())

    def get_missing_parameters(
        self,
        state: DecoderState,
    ) -> set[str]:
        """Return required parameter keys not yet used in state."""
        return set(self.parameters.keys()) - state.used_keys

    def is_valid_key(self, key: str) -> bool:
        """Check whether a key exists in the function schema."""
        return key in self.parameters

    def get_expected_type(self, state: DecoderState) -> str:
        """Return the expected schema type for state.current_key."""
        if state.current_key is None:
            raise SchemaRulesError("current_key does not exist")

        if state.current_key not in self.parameters:
            raise SchemaRulesError("current_key does not exist in schema")

        return self.parameters[state.current_key]
