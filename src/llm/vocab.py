from llm_sdk import Small_LLM_Model
import json
from typing import cast


def get_vocab_path(model: Small_LLM_Model) -> str:
    return cast(str, model.get_path_to_vocab_file())


def load_vocab(path: str) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as file:
        return cast(dict[str, int], json.load(file))


def invert_vocab(vocab: dict[str, int]) -> dict[int, str]:
    return {value: key for key, value in vocab.items()}
