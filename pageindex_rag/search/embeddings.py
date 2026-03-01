"""Embedding utilities for PageIndex RAG."""


def get_embedding(text: str, model: str, api_key: str, base_url: str = "https://api.openai.com/v1") -> list[float]:
    """Call OpenAI-compatible embedding API and return embedding vector."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


def get_embeddings_batch(
    texts: list[str], model: str, api_key: str, base_url: str = "https://api.openai.com/v1"
) -> list[list[float]]:
    """批量调用 embedding API，单次请求返回多个向量（替代 N 次串行调用）。

    OpenAI 单次最多 2048 个输入；一般 10-K 节点数 50~200，不会超限。
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
