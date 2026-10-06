import argparse
import sys
from pathlib import Path
from pydantic import ValidationError
from src.input.loader import load_file
from src.input.parser import parse_json
from src.errors.input_errors import InvalidJsonError, InputFileError
from src.models.prompt_definition import validate_prompts
from src.models.function_definition import validate_functions

parser = argparse.ArgumentParser()

parser.add_argument(
    "--output",
    default="data/output/function_calls.json"
    )
parser.add_argument(
    "--input",
    default="data/input/function_calling_tests.json"
    )
parser.add_argument(
    "--functions_definition",
    default="data/input/functions_definition.json"
    )

args = parser.parse_args()

try:
    input_path = Path(args.input)
    file_content = load_file(input_path)
    prompts_data = parse_json(file_content)
    prompts = validate_prompts(prompts_data)
    function_definition = Path(args.functions_definition)
    function_content = load_file(function_definition)
    functions_data = parse_json(function_content)
    functions = validate_functions(functions_data)
except (InputFileError, InvalidJsonError, ValidationError) as error:
    print(error)
    sys.exit(1)
