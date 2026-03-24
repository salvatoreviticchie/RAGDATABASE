from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from .config import settings
from .retriever import retrieve

_openai = OpenAI(api_key=settings.openai_api_key)

_SYSTEM_PROMPT = """You are a precise document assistant. Answer ONLY using the context provided below.
If the answer cannot be found in the context, respond with: "I could not find this in the uploaded documents."
When you use information from the context, cite the source inline using the format [Source: <filename>, Page <n>]."""


@dataclass
class GeneratorResponse:
    answer: str
    sources: list[dict]  # each: {score, metadata: {source_file, page, chunk_index, text}}


def answer(query: str, top_k: int | None = None) -> GeneratorResponse:
    """Retrieve relevant chunks and generate a grounded answer with GPT-4o."""
    matches = retrieve(query, top_k=top_k)

    if not matches:
        return GeneratorResponse(
            answer="I could not find this in the uploaded documents.",
            sources=[],
        )

    # Build context block with source labels
    context_parts = []
    for m in matches:
        meta = m["metadata"]
        label = f"[Source: {meta['source_file']}, Page {meta['page']}]"
        context_parts.append(f"{meta['text']}\n{label}")
    context = "\n---\n".join(context_parts)

    user_message = f"Context:\n---\n{context}\n---\n\nQuestion: {query}"

    completion = _openai.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        max_tokens=1024,
    )

    return GeneratorResponse(
        answer=completion.choices[0].message.content,
        sources=matches,
    )
