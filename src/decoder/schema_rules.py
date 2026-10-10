from src.models.function_definition import FunctionDefinition
from src.errors.decoder_errors import SchemaRulesError
from src.decoder.json_state import DecoderState
from pydantic import BaseModel


class SchemaRules(BaseModel):
    parameters: dict[str, str]

    @classmethod
    def from_function(
        cls,
        function: FunctionDefinition
    ) -> "SchemaRules":
        return cls(
            parameters={
                name: parameter.type
                for name, parameter in function.parameters.items()
            }
        )

    def get_all_keys(self) -> set[str]:
        return set(self.parameters.keys())

    def get_missing_parameters(
        self,
        state: DecoderState
    ) -> set[str]:
        return set(self.parameters.keys()) - state.used_keys

    def is_valid_key(self, key: str) -> bool:
        return key in self.parameters

    def get_expected_type(self, state: DecoderState) -> str:
        if state.current_key is None:
            raise SchemaRulesError("current_key does not exist")

        if state.current_key not in self.parameters:
            raise SchemaRulesError("current_key does not exist in schema")

        return (self.parameters[state.current_key])
