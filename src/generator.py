from __future__ import annotations

import time
from dataclasses import dataclass

from openai import OpenAI, RateLimitError, NotFoundError, BadRequestError

from .config import settings
from .retriever import retrieve

_openrouter = OpenAI(
    api_key=settings.openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
)

# Primary model from .env, then fallbacks tried in order.
# Models are spread across different upstream providers so they don't all
# rate-limit at the same time (Venice, Fireworks, Together, Google, etc.)
_FALLBACK_MODELS = [
    settings.llm_model,
    "mistralai/mistral-7b-instruct:free",        # Fireworks / Together
    "google/gemma-2-9b-it:free",                 # Google
    "google/gemma-3-1b-it:free",                 # Google (small, fast)
    "qwen/qwen-2-7b-instruct:free",              # Together
    "microsoft/phi-3-mini-128k-instruct:free",   # Azure / Together
    "meta-llama/llama-3.2-3b-instruct:free",     # Fireworks
    "openchat/openchat-7b:free",                 # Lepton
    "huggingfaceh4/zephyr-7b-beta:free",         # HuggingFace
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
    Automatically falls back through free models across different providers
    if one is unavailable or rate-limited.
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
    for i, model in enumerate(models_to_try):
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
        except (RateLimitError, NotFoundError, BadRequestError) as e:
            last_error = e
            # Brief pause before trying the next model
            if i < len(models_to_try) - 1:
                time.sleep(1)
            continue

    raise RuntimeError(
        "⚠️ All free models are currently overloaded.\n\n"
        "Options:\n"
        "1. Wait 1–2 minutes and try again\n"
        "2. Add $1 credit at https://openrouter.ai/settings/integrations "
        "to unlock rate-limit-free access\n\n"
        f"Last error: {last_error}"
    )
