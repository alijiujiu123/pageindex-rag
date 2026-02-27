"""Embedding utilities for PageIndex RAG."""


def get_embedding(text: str, model: str, api_key: str, base_url: str = "https://api.openai.com/v1") -> list[float]:
    """Call OpenAI-compatible embedding API and return embedding vector."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding
