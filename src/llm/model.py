from typing import cast

from llm_sdk import Small_LLM_Model


def encode_text(model: Small_LLM_Model, text: str) -> list[int]:
    return cast(list[int], model.encode(text)[0].tolist())


def decode_tokens(model: Small_LLM_Model, tokens: list[int]) -> str:
    return cast(str, model.decode(tokens))


def get_next_token_logits(model: Small_LLM_Model, tokens: list[int]
                          ) -> list[float]:
    return cast(list[float], model.get_logits_from_input_ids(tokens))


def get_best_token_id(logits: list[float]) -> int:
    return max(range(len(logits)), key=lambda i: logits[i])


def get_top_token_ids(logits: list[float], limit: int) -> list[int]:
    in_order = sorted(
        range(len(logits)), key=lambda i: logits[i], reverse=True)
    return in_order[:limit]
