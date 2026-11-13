*This project has been created as part of the 42 curriculum by rgoulart.*

# Call Me Maybe — Introduction to Function Calling in LLMs

## Description

**Call Me Maybe** is a command-line tool that translates natural language requests into structured, schema-compliant JSON function calls using a small language model (`Qwen/Qwen3-0.6B`, with approximately $0.6\text{B}$ parameters).

Instead of answering a user prompt in free-form text (e.g., returning `"The sum of 40 and 2 is 42"`), the system identifies the appropriate tool from a provided catalog of function definitions and extracts typed arguments in a machine-executable format:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 40.0,
    "b": 2.0
  }
}
```

Small language models often struggle to spontaneously produce valid JSON when relying on prompting alone. To achieve $100\%$ structural and semantic validity, this project implements **Constrained Decoding**: at every generation step, the model's raw probability scores (logits) are filtered against a JSON state machine and the target function's parameter schema, setting invalid token logits to $-\infty$ before selecting the next token.

---

## Instructions

### Prerequisites

* **Python:** `3.10` or higher
* **Package Manager:** [`uv`](https://docs.astral.sh/uv/)
* **Local SDK:** `llm_sdk` directory in the repository root

### Installation

Synchronize the virtual environment and install all required dependencies (`pydantic`, `numpy`, `llm_sdk`, `flake8`, `mypy`):

```bash
uv sync
```

Or using the provided `Makefile`:

```bash
make install
```

### Execution

Run the module using `uv`:

```bash
uv run python -m src \
  [--functions_definition <function_definition_file>] \
  [--input <input_file>] \
  [--output <output_file>]
```

By default, if no arguments are passed, the program reads from `data/input/functions_definition.json` and `data/input/function_calling_tests.json`, and writes the output to `data/output/function_calls.json`:

```bash
make run
```

### Makefile Targets

* `make install`: Install project dependencies via `uv sync`.
* `make run`: Execute the main pipeline (`uv run python -m src`).
* `make debug`: Run the main script in interactive debug mode using `pdb`.
* `make clean`: Remove temporary caches (`__pycache__`, `.mypy_cache`, `.pytest_cache`).
* `make lint`: Run `flake8` and `mypy` with all mandatory flags required by the subject.
* `make lint-strict`: Run `flake8` and `mypy --strict`.

---

## Algorithm Explanation

The generation pipeline is divided into two constrained decoding phases: **Function Selection** and **Parameter Extraction**.

### Phase 1: Constrained Function Selection (`src/llm/function_selector.py`)

1. **Dynamic Context Construction:** `build_function_context()` and `build_function_prompt()` serialize all candidate `FunctionDefinition` objects (names, descriptions, parameter names, and types) alongside the user's natural language prompt.
2. **Candidate Name Tokenization:** `tokenize_function_names()` encodes each available function name into its exact sequence of token IDs using `Small_LLM_Model.encode()`.
3. **Prefix-Constrained Decoding Loop:** Starting at token position $0$, the model computes next-token logits for the prompt plus the tokens chosen so far:
   * `get_valid_token_ids()` collects only the token IDs that match the current position of at least one remaining candidate function name.
   * `mask_invalid_logits()` sets the logit of every token ID outside this valid set to $-\infty$.
   * `get_best_token_id()` selects the highest-scoring valid token ($\arg\max$), appends it to the context, and `filter_remaining_functions()` prunes candidates that no longer match the chosen prefix.
   * The loop terminates as soon as `get_completed_function()` matches a complete function name, guaranteeing that the LLM can only output a valid function defined in the input catalog.

### Phase 2: Constrained Parameter Generation (`src/decoder/`)

Once a function is selected, `SchemaRules.from_function()` extracts the expected parameter keys and their target types (`"string"`, `"number"`, `"boolean"`). Parameter generation then proceeds token by token in `generate_parameters()`:

1. **JSON State Machine (`src/decoder/json_state.py`):**
   `DecoderState` tracks the current structural state (`JsonState.START` $\rightarrow$ `EXPECT_KEY` $\rightarrow$ `EXPECT_COLON` $\rightarrow$ `EXPECT_VALUE` $\rightarrow$ `EXPECT_COMMA_OR_END` $\rightarrow$ `DONE`), the `current_key` being populated, the set of already completed `used_keys`, and a string `buffer` for multi-token keys or values.
2. **Partial Value Validators:**
   Because BPE tokens can represent partial fragments, `validate_string()`, `validate_number()`, and `validate_boolean()` classify any candidate buffer into one of three states:
   * `ValidationState.POSSIBLE`: Valid prefix that requires additional tokens (e.g., `"hel`, `-`, `2.`, `tr`).
   * `ValidationState.VALID`: Complete literal satisfying the type (e.g., `"hello"`, `265`, `-3.14`, `true`).
   * `ValidationState.INVALID`: Syntactically or semantically illegal sequence.
3. **Token Filtering & Logit Masking (`src/decoder/token_filter.py`):**
   At each step, `is_token_allowed()` simulates calling `next_state()` with each vocabulary token:
   * Tokens that raise `JsonStateError` are rejected.
   * In `EXPECT_KEY`, only prefixes of remaining unused schema keys are allowed.
   * In `EXPECT_COLON`, the completed key must exist in `SchemaRules` and not be in `used_keys`.
   * In `DONE`, closing the object with `}` is only permitted when `get_missing_parameters()` is empty.
   * `mask_logits()` assigns $-\infty$ to all disallowed token IDs, and `select_next_token()` picks the valid token with the highest logit.
4. **Type Normalization & Output Validation (`src/output/validator.py`):**
   Once `JsonState.DONE` is reached, the generated JSON string is parsed with `json.loads()`, numeric arguments are normalized to Python `float` (producing `2.0` instead of `2` as required by the evaluation scale), and `validate_output()` verifies the final record with Pydantic (`FunctionCallOutput`).

---

## Design Decisions

* **Strict Separation of Concerns:** Syntax rules (`json_state.py`), schema rules (`schema_rules.py`), vocabulary filtering (`token_filter.py`), and LLM orchestration (`constrained_decoder.py`) are isolated into dedicated modules.
* **Immutable Decoder State with Pydantic:** `DecoderState` inherits from `pydantic.BaseModel` and uses `model_copy()` on every state transition. This allows `is_token_allowed()` to safely simulate thousands of candidate tokens from the vocabulary without mutating the active generation state.
* **Public-Only SDK Wrapper:** `src/llm/model.py` and `src/llm/vocab.py` interact exclusively with the public methods of `Small_LLM_Model` (`encode`, `decode`, `get_logits_from_input_ids`, and `get_path_to_vocab_file`).
* **Vocabulary Normalization & $O(1)$ Lookup:** The vocabulary file is loaded once at startup, BPE whitespace markers (`Ġ`, `Ċ`, `ĉ`, `č`) are normalized into standard characters, and `invert_vocab()` builds an $O(1)$ `id_to_token` dictionary.

---

## Performance Analysis

* **JSON & Schema Reliability ($100\%$):** Because invalid tokens receive a logit of $-\infty$ prior to selection, the decoder cannot emit malformed JSON, unknown parameter keys, wrong argument types, or close the JSON object before all required parameters are filled.
* **Accuracy ($\ge 90\%$):** Guided prefix selection for function names combined with schema-constrained parameter extraction achieves near-perfect accuracy on the provided test suite and general evaluation sets.
* **Execution Speed ($< 5\text{ minutes}$):** Loading the model and vocabulary once per run and keeping state transition checks lightweight ensures that all test prompts are processed well within the 5-minute threshold on standard hardware.

---

## Challenges Faced

1. **Subword BPE Tokenization & Whitespace Markers:** Tokenizer vocabularies represent leading spaces with `Ġ` and newlines with `Ċ`. Normalizing these markers and handling leading spaces outside string literals in `next_state()` was essential to avoid rejecting valid structural tokens.
2. **Multi-Digit Numbers vs. Delimiters:** Unlike strings (which have an explicit closing quote `"`) or booleans (`true`/`false`), numbers such as `265` are generated digit by digit (`"2"`, `"6"`, `"5"`). If the state machine transitioned immediately after the first valid digit `"2"`, subsequent digits would be blocked. We solved this by keeping `EXPECT_VALUE` active while digits/decimals accumulate in `buffer` and finalizing the number when `,` or `}` arrives.
3. **Preventing Premature Object Closure:** Small models sometimes attempt to emit `{}` or close the JSON object after the first parameter. Checking `schema_rules.get_missing_parameters()` before allowing `JsonState.DONE` guarantees every required argument is present.
4. **Python `bool` Subclassing `int`:** In Python, `isinstance(True, (int, float))` evaluates to `True`. In `validate_output()`, we explicitly enforce `float` for `"number"` parameters and reject `isinstance(value, bool)`.

---

## Testing Strategy

The implementation was validated across three layers:

1. **Unit Tests (`pytest`):** Deterministic tests covering `load_file`, `parse_json`, Pydantic models, `validate_string`/`validate_number`/`validate_boolean`, `next_state` transitions, `SchemaRules`, `mask_logits` ($-\infty$ masking), and `validate_output`.
2. **Edge Case & Error Handling Tests:** Tested with missing input files, malformed JSON files, empty function lists, multi-digit and negative floats, empty strings, and special characters to ensure graceful error messages and zero unhandled exceptions.
3. **Static Analysis & Linting:** Verified code formatting and type safety using `flake8` and `mypy` (`make lint` and `make lint-strict`).

---

## Example Usage

### Running with Default Paths

```bash
uv run python -m src
```

### Running with Custom Paths

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

### Example Output (`data/output/function_calls.json`)

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  },
  {
    "prompt": "What is the sum of 265 and 345?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 265.0,
      "b": 345.0
    }
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {
      "name": "shrek"
    }
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {
      "s": "hello"
    }
  }
]
```

---

## Resources

### Technical References

* [Willard & Louf (2023) — Efficient Guided Generation for Large Language Models (Outlines)](https://arxiv.org/abs/2307.09702)
* [JSON Specification (RFC 8259)](https://datatracker.ietf.org/doc/html/rfc8259)
* [Pydantic v2 Documentation](https://docs.pydantic.dev/latest/)
* [Qwen3 Technical Report & Tokenizer Documentation](https://huggingface.co/Qwen/Qwen3-0.6B)

### Use of AI

In accordance with the 42 AI instructions, AI tools were used as a learning and productivity assistant for the following tasks:

* **Conceptual Understanding:** Studying how BPE tokenizers, logits, and finite-state machines interact during constrained decoding.
* **Architecture & Debugging:** Reviewing module boundaries (`json_state`, `schema_rules`, `token_filter`) and diagnosing edge cases such as multi-digit number transitions and BPE whitespace prefixes.
* **Documentation & Testing:** Structuring PEP 257 docstrings, reviewing `mypy` type annotations, and drafting this `README.md`. All code and logic were manually reviewed, tested, and validated.
