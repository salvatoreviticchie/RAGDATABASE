from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI, RateLimitError, NotFoundError

from .config import settings
from .retriever import retrieve

_openrouter = OpenAI(
    api_key=settings.openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
)

# Primary model from .env, then fallbacks tried in order if unavailable/rate-limited
_FALLBACK_MODELS = [
    settings.llm_model,
    "mistralai/mistral-7b-instruct:free",
    "qwen/qwq-32b:free",
    "microsoft/phi-3-mini-128k-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
]

_SYSTEM_PROMPT = """You are a precise document assistant. Answer ONLY using the context provided below.
If the answer cannot be found in the context, respond with: "I could not find this in the uploaded documents."
When you use information from the context, cite the source inline using the format [Source: <filename>, Page <n>]."""


@dataclass
class GeneratorResponse:
    answer: str
    sources: list[dict]  # each: {score, metadata: {source_file, page, chunk_index, text}}
    model_used: str = ""


def answer(query: str, top_k: int | None = None) -> GeneratorResponse:
    """Retrieve relevant chunks and generate a grounded answer via OpenRouter.
    Automatically falls back through free models if one is unavailable or rate-limited.
    """
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

    # Deduplicate while preserving order
    seen: set[str] = set()
    models_to_try = [m for m in _FALLBACK_MODELS if not (m in seen or seen.add(m))]

    last_error: Exception | None = None
    for model in models_to_try:
        try:
            completion = _openrouter.chat.completions.create(
                model=model,
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
                model_used=model,
            )
        except (RateLimitError, NotFoundError) as e:
            last_error = e
            continue  # try next model

    raise RuntimeError(
        f"All free models are currently unavailable. Last error: {last_error}\n"
        "Try again in a few minutes or add credit at https://openrouter.ai/settings/integrations"
    )
