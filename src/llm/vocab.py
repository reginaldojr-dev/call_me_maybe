"""Vocabulary path resolution, loading, and inversion utilities."""

from pathlib import Path
from typing import Any

from llm_sdk import Small_LLM_Model

from src.input.loader import load_file
from src.input.parser import parse_json


def get_vocab_path(model: Small_LLM_Model) -> Path:
    """Retrieve the Path to the model's vocabulary JSON file."""
    return Path(model.get_path_to_vocab_file())


def load_vocab(vocab_path: Path) -> dict[str, int]:
    """Load and validate the token-to-ID dictionary from vocab_path."""
    raw_content = load_file(vocab_path)
    parsed: Any = parse_json(raw_content)

    if not isinstance(parsed, dict):
        raise ValueError("Vocabulary JSON must be an object/dictionary")

    vocab: dict[str, int] = {}
    for token, token_id in parsed.items():
        if not isinstance(token, str) or not isinstance(token_id, int):
            raise ValueError("Invalid token or token_id in vocabulary")
        vocab[token] = token_id

    return vocab


def invert_vocab(vocab: dict[str, int]) -> dict[int, str]:
    """Invert a token-to-ID mapping into an ID-to-token dictionary."""
    return {token_id: token for token, token_id in vocab.items()}
