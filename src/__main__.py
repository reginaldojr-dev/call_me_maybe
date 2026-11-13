"""CLI entry point and end-to-end orchestration for Call Me Maybe."""

import argparse
import sys
from pathlib import Path
from typing import Any

from llm_sdk.llm_sdk import Small_LLM_Model
from pydantic import ValidationError

from src.decoder.constrained_decoder import (
    build_parameters_prompt,
    generate_parameters,
)
from src.decoder.schema_rules import SchemaRules
from src.errors.decoder_errors import JsonStateError, SchemaRulesError
from src.errors.functions_selection_errors import FunctionSelectionError
from src.errors.input_errors import InputFileError, InvalidJsonError
from src.errors.output_errors import OutputError
from src.input.loader import load_file
from src.input.parser import parse_json
from src.llm.function_selector import (
    select_function_with_llm,
    validate_selected_function,
)
from src.llm.vocab import get_vocab_path, load_vocab
from src.models.function_definition import validate_functions
from src.models.prompt_definition import validate_prompts
from src.output.validator import validate_output
from src.output.writer import write_output


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for input, function definitions, and output."""
    parser = argparse.ArgumentParser(
        description="Translate natural language prompts into function calls."
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json",
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON file.",
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the functions definition JSON file.",
    )
    return parser.parse_args()


def normalize_vocab(raw_vocab: dict[str, int]) -> dict[str, int]:
    """Convert BPE markers into regular characters in vocab keys."""
    normalized: dict[str, int] = {}
    for token_str, token_id in raw_vocab.items():
        clean_str = (
            token_str.replace("Ġ", " ")
            .replace("Ċ", "\n")
            .replace("ĉ", "\t")
            .replace("č", "\r")
        )
        if clean_str not in normalized:
            normalized[clean_str] = token_id
    return normalized


def main() -> None:
    """Execute the complete constrained function calling pipeline."""
    args = parse_args()

    try:
        prompts_content = load_file(Path(args.input))
        prompts_data = parse_json(prompts_content)
        prompts = validate_prompts(prompts_data)

        functions_content = load_file(Path(args.functions_definition))
        functions_data = parse_json(functions_content)
        functions = validate_functions(functions_data)

        model = Small_LLM_Model()
        raw_vocab = load_vocab(get_vocab_path(model))
        vocab = normalize_vocab(raw_vocab)

        results: list[dict[str, Any]] = []
        for item in prompts:
            selected_name = select_function_with_llm(
                model, item.prompt, functions
            )
            selected_function = validate_selected_function(
                selected_name, functions
            )
            schema_rules = SchemaRules.from_function(selected_function)

            parameters_prompt = build_parameters_prompt(
                selected_function, item.prompt
            )
            parameters = generate_parameters(
                llm=model,
                vocab=vocab,
                schema_rules=schema_rules,
                prompt=parameters_prompt,
            )

            call_result: dict[str, Any] = {
                "prompt": item.prompt,
                "name": selected_function.name,
                "parameters": parameters,
            }
            validated_result = validate_output(call_result, schema_rules)
            results.append(validated_result)

        write_output(results, args.output)
    except (
        InputFileError,
        InvalidJsonError,
        ValidationError,
        FunctionSelectionError,
        JsonStateError,
        SchemaRulesError,
        OutputError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        print(f"Unexpected error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
