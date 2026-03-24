from __future__ import annotations

from openai import OpenAI

from .config import settings, get_index

_openai = OpenAI(api_key=settings.openai_api_key)


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    """
    Embed the query and return the top_k most similar chunks from Pinecone.

    Each result dict has:
        score       float   cosine similarity (0-1, higher is more relevant)
        metadata    dict    {source_file, page, chunk_index, text}
    """
    k = top_k if top_k is not None else settings.top_k

    response = _openai.embeddings.create(
        input=[query],
        model=settings.embedding_model,
    )
    query_vector = response.data[0].embedding

    index = get_index()
    results = index.query(
        vector=query_vector,
        top_k=k,
        include_metadata=True,
    )

    return [
        {"score": match.score, "metadata": match.metadata}
        for match in results.matches
    ]
