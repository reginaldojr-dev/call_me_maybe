from llm_sdk.llm_sdk import Small_LLM_Model


def encode_text(model: Small_LLM_Model, text: str) -> list[int]:
    return model.encode(text)[0].tolist()
