from __future__ import annotations

from .config import settings, get_index, get_pc


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    """
    Embed the query using Pinecone Inference and return the top_k most similar chunks.

    Each result dict has:
        score       float   cosine similarity (0-1, higher is more relevant)
        metadata    dict    {source_file, page, chunk_index, text}
    """
    k = top_k if top_k is not None else settings.top_k

    pc = get_pc()
    response = pc.inference.embed(
        model=settings.embedding_model,
        inputs=[query],
        parameters={"input_type": "query", "truncate": "END"},
    )
    query_vector = response.data[0]["values"]

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
