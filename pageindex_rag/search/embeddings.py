"""Embedding utilities for PageIndex RAG."""


def get_embedding(text: str, model: str, api_key: str) -> list[float]:
    """Call OpenAI embedding API and return embedding vector."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding
