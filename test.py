from pathlib import Path
from src.input.loader import load_file
from src.input.parser import parse_json
from src.models.function_definition import validate_functions
from llm_sdk import Small_LLM_Model
from src.llm.model import (
    encode_text,
    decode_tokens,
    get_next_token_logits,
    get_best_token_id,
    get_top_token_ids
)
from src.llm.vocab import get_vocab_path, load_vocab, invert_vocab
from src.llm.function_selector import select_function_with_llm


model = Small_LLM_Model()
tokens = encode_text(model, 'Hello')
logits = get_next_token_logits(model, tokens)
best_token = get_best_token_id(logits)
vocab_path = get_vocab_path(model)
vocab = load_vocab(vocab_path)
invert = invert_vocab(vocab)
top_logits = get_top_token_ids(logits, 5)

"""
"What is the sum of 2 and 3?" -> fn_add_numbers
"Greet john" -> fn_greet
"Reverse the string 'hello'" -> fn_reverse_string
"""

print(f"Tokenização de 'Hello' : {tokens}")
print(f"Quantidade de logits, ou seja, quantidade de possíveis "
      f"próximos tokens: {len(logits)}")
print(f"Índice do token com maior logit: {get_best_token_id(logits)}")
print(f"Decode do token original: '{decode_tokens(model, tokens)}'")
print(f"Decode do token com maior logit: "
      f"'{decode_tokens(model, [best_token])}'")
print(f"Caminho do vocab.json: {vocab_path}")
print(f"Quantidade de entradas: {len(vocab)}")
print(f"Valor associado a 'Hello': {vocab.get('Hello')}")
print(f"Valor associado a '9707': {invert.get(9707)}")
print(f"Tokens com os maiores 5 logits: {top_logits}")
print("ID           logit          texto")

for token_id in top_logits:
    print(
        f"{token_id} | {logits[token_id]} | "
        f"{repr(decode_tokens(model, [token_id]))}"
    )

function_path = Path("data/input/functions_definition.json")
function_content = load_file(function_path)
functions_data = parse_json(function_content)
functions = validate_functions(functions_data)

test_prompts = [
    "What is the sum of 2 and 3?",
    "Greet john",
    "Reverse the string 'hello'"
]

print("\nFunction selection:")

for prompt in test_prompts:
    selected = select_function_with_llm(
        model,
        prompt,
        functions
    )
    print(f"{prompt} -> {selected}")
