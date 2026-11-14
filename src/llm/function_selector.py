"""Constrained prefix decoding for selecting a function with the LLM."""

from llm_sdk import Small_LLM_Model

from src.errors.functions_selection_errors import FunctionSelectionError
from src.llm.model import encode_text, get_best_token_id, get_next_token_logits
from src.models.function_definition import FunctionDefinition


def build_function_context(functions: list[FunctionDefinition]) -> str:
    """Format available function definitions into a readable text catalog."""
    lines: list[str] = []
    for func in functions:
        params = ", ".join(
            f"{name}: {param.type}"
            for name, param in func.parameters.items()
        )
        line = (
            f"- {func.name}({params}) -> {func.returns.type}: "
            f"{func.description}"
        )
        lines.append(line)
    return "\n".join(lines)


def build_function_prompt(
    prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """Build the prompt asking the LLM to choose one function name."""
    context = build_function_context(functions)
    return (
        "Select the correct function for the user request.\n\n"
        f"Available functions:\n{context}\n\n"
        f"User request: {prompt}\n"
        "Function name: "
    )


def tokenize_function_names(
    model: Small_LLM_Model,
    functions: list[FunctionDefinition],
) -> dict[str, list[int]]:
    """Encode each candidate function name into its list of token IDs."""
    return {func.name: encode_text(model, func.name) for func in functions}


def get_valid_token_ids(
    remaining_functions: dict[str, list[int]],
    position: int,
) -> set[int]:
    """Return the set of valid token IDs at the current prefix position."""
    valid_token_ids: set[int] = set()

    for token_ids in remaining_functions.values():
        if position < len(token_ids):
            valid_token_ids.add(token_ids[position])

    return valid_token_ids


def mask_invalid_logits(
    logits: list[float],
    valid_token_ids: set[int],
) -> list[float]:
    """Mask logits by setting disallowed token IDs to negative infinity."""
    return [
        logit if index in valid_token_ids else float("-inf")
        for index, logit in enumerate(logits)
    ]


def filter_remaining_functions(
    remaining_functions: dict[str, list[int]],
    position: int,
    chosen_token_id: int,
) -> dict[str, list[int]]:
    """Keep only candidate functions matching chosen_token_id at position."""
    return {
        name: token_ids
        for name, token_ids in remaining_functions.items()
        if position < len(token_ids)
        and token_ids[position] == chosen_token_id
    }


def get_completed_function(
    remaining_functions: dict[str, list[int]],
    chosen_tokens: list[int],
) -> str | None:
    """Return the function name if chosen_tokens matches a complete name."""
    for name, token_ids in remaining_functions.items():
        if token_ids == chosen_tokens:
            return name
    return None


def select_function_with_llm(
    model: Small_LLM_Model,
    prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """Select the best function name for prompt using constrained logits."""
    if not functions:
        raise FunctionSelectionError("functions list cannot be empty")
    if not prompt.strip():
        raise FunctionSelectionError("prompt cannot be empty")

    selection_prompt = build_function_prompt(prompt, functions)
    input_ids = encode_text(model, selection_prompt)
    remaining_functions = tokenize_function_names(model, functions)

    chosen_tokens: list[int] = []
    position = 0

    while remaining_functions:
        valid_token_ids = get_valid_token_ids(remaining_functions, position)
        if not valid_token_ids:
            break

        logits = get_next_token_logits(model, input_ids + chosen_tokens)
        masked_logits = mask_invalid_logits(logits, valid_token_ids)
        chosen_token_id = get_best_token_id(masked_logits)

        chosen_tokens.append(chosen_token_id)
        remaining_functions = filter_remaining_functions(
            remaining_functions, position, chosen_token_id
        )

        completed_function = get_completed_function(
            remaining_functions, chosen_tokens
        )
        if completed_function is not None:
            return completed_function
        position += 1

    raise FunctionSelectionError("LLM could not select a valid function")


def validate_selected_function(
    function_name: str,
    functions: list[FunctionDefinition],
) -> FunctionDefinition:
    """Return the FunctionDefinition matching function_name or raise error."""
    cleaned_name = function_name.strip()
    if not cleaned_name:
        raise FunctionSelectionError("Selected function name cannot be empty")
    for function in functions:
        if function.name == cleaned_name:
            return function
    raise FunctionSelectionError(
        f"Selected function '{cleaned_name}' does not exist"
    )
