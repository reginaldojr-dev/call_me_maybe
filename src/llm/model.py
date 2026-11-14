"""Wrapper functions around the public Small_LLM_Model SDK interface."""

from llm_sdk import Small_LLM_Model


def encode_text(model: Small_LLM_Model, text: str) -> list[int]:
    """Encode a text string into a flat list of token IDs."""
    tensor_ids = model.encode(text)
    ids: list[int] = tensor_ids[0].tolist()
    return ids


def decode_tokens(model: Small_LLM_Model, token_ids: list[int]) -> str:
    """Decode a list of token IDs back into a text string."""
    text: str = model.decode(token_ids)
    return text


def get_next_token_logits(
    model: Small_LLM_Model,
    input_ids: list[int],
) -> list[float]:
    """Return the raw next-token logits for the given input token IDs."""
    if not input_ids:
        raise ValueError("input_ids cannot be empty")
    logits: list[float] = model.get_logits_from_input_ids(input_ids)
    return logits


def get_best_token_id(logits: list[float]) -> int:
    """Return the index (token ID) of the maximum logit value."""
    if not logits:
        raise ValueError("logits cannot be empty")
    best_id = max(range(len(logits)), key=lambda i: logits[i])
    return best_id


def get_top_token_ids(logits: list[float], limit: int = 5) -> list[int]:
    """Return the top-k token IDs sorted by descending logit score."""
    if not logits:
        raise ValueError("logits cannot be empty")
    if limit <= 0:
        raise ValueError("limit must be greater than 0")
    sorted_ids = sorted(
        range(len(logits)), key=lambda i: logits[i], reverse=True
    )
    return sorted_ids[:limit]
