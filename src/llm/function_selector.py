from src.models.function_definition import FunctionDefinition
from llm_sdk import Small_LLM_Model
from src.llm.model import (encode_text, get_next_token_logits,
                           get_best_token_id)
from src.errors.functions_selection_errors import FunctionSelectionError


def build_function_context(
    functions: list[FunctionDefinition]
) -> str:
    context = ""
    for function in functions:
        context += (
            f"{function.name}, {function.description},"
            f"{function.parameters}\n"
        )
    return context


def build_function_prompt(
    context: str,
    prompt: str
) -> str:
    return (
        "Choose the function that best matches the user's request.\n"
        "Return only the exact function name from the available functions.\n\n"
        "Available functions:\n"
        f"{context}\n"
        f"User request: {prompt}\n"
        "Function:"
        )


def tokenize_function_names(
    model: Small_LLM_Model,
    functions: list[FunctionDefinition]
) -> dict[str, list[int]]:
    function_tokens = {
            function.name: encode_text(model, " " + function.name + "\n")
            for function in functions
        }
    return function_tokens


def get_valid_token_ids(
    remaining_functions,
    position
) -> set[int]:
    return {
        tokens[position]
        for tokens in remaining_functions.values()
        if position < len(tokens)
    }


def mask_invalid_logits(
    logits: list[float],
    valid_token_ids: set[int]
) -> list[float]:
    return [
        logit if token_id in valid_token_ids else float("-inf")
        for token_id, logit in enumerate(logits)
    ]


def filter_remaining_functions(
    remaining_functions: dict[str, list[int]],
    position: int,
    best_token_id: int
) -> dict[str, list[int]]:
    return {
        name: tokens
        for name, tokens in remaining_functions.items()
        if (
            position < len(tokens)
            and tokens[position] == best_token_id
            )
    }


def get_completed_function(
    remaining_functions: dict[str, list[int]],
    position: int
) -> str | None:
    completed = [
        name
        for name, tokens in remaining_functions.items()
        if position == len(tokens)
    ]

    if completed:
        return completed[0]

    return None


def select_function_with_llm(
    model: Small_LLM_Model,
    prompt: str,
    functions: list[FunctionDefinition]
) -> str:
    context = build_function_context(functions)
    full_prompt = build_function_prompt(context, prompt)

    prompt_tokens = encode_text(model, full_prompt)

    function_tokens = tokenize_function_names(model, functions)

    remaining_functions = function_tokens
    selected_tokens: list[int] = []

    while remaining_functions:
        position = len(selected_tokens)

        completed_function = get_completed_function(
            remaining_functions,
            position
        )
        if completed_function is not None:
            return completed_function

        valid_token_ids = get_valid_token_ids(
            remaining_functions,
            position
        )

        logits = get_next_token_logits(
            model,
            prompt_tokens + selected_tokens
        )

        masked_logits = mask_invalid_logits(logits, valid_token_ids)

        best_token_id = get_best_token_id(masked_logits)
        selected_tokens.append(best_token_id)

        remaining_functions = filter_remaining_functions(
            remaining_functions, position, best_token_id
        )

    raise FunctionSelectionError("LLM could not select a valid function")


def validate_selected_function(
    selected_name: str,
    functions: list[FunctionDefinition]
) -> FunctionDefinition:
    for function in functions:
        if function.name == selected_name:
            return function

    raise FunctionSelectionError(
        "LLM could not select a valid function"
    )
