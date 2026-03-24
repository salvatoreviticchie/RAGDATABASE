from __future__ import annotations

import time
from dataclasses import dataclass, field

from openai import OpenAI, RateLimitError, NotFoundError, BadRequestError

from .config import settings
from .retriever import retrieve

_openrouter = OpenAI(
    api_key=settings.openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
)

# Primary model from .env, then paid fallbacks in order.
_FALLBACK_MODELS = [
    settings.llm_model,
    "meta-llama/llama-3.3-70b-instruct",   # strong, cheap (~$0.0003/1k tokens)
    "mistralai/mistral-7b-instruct",        # very fast, very cheap
    "google/gemma-2-9b-it",                # reliable Google model
    "qwen/qwen-2-7b-instruct",             # good fallback
    "microsoft/phi-3-mini-128k-instruct",  # lightweight fallback
]

# How many previous chat turns to include (1 turn = 1 user + 1 assistant message)
_MAX_HISTORY_TURNS = 5

_SYSTEM_PROMPT = """You are a precise document assistant. Answer ONLY using the context provided below.
If the answer cannot be found in the context, respond with: "I could not find this in the uploaded documents."
When you use information from the context, cite the source inline using the format [Source: <filename>, Page <n>].
You also have access to the recent conversation history — use it to understand follow-up questions and references like "it", "that", "the previous answer"."""


@dataclass
class GeneratorResponse:
    answer: str
    sources: list[dict]  # each: {score, metadata: {source_file, page, chunk_index, text}}
    model_used: str = ""


def answer(
    query: str,
    chat_history: list[dict] | None = None,
    top_k: int | None = None,
) -> GeneratorResponse:
    """Retrieve relevant chunks and generate a grounded answer via OpenRouter.

    Args:
        query:        The user's current question.
        chat_history: Previous turns as [{"role": "user"|"assistant", "content": str}, ...]
                      Pass st.session_state.chat_history directly from app.py.
        top_k:        Number of chunks to retrieve (defaults to settings.top_k).

    Automatically falls back through models if one is rate-limited or unavailable.
    """
    # Retrieve relevant chunks for the current query
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

    # ── Build the message list ────────────────────────────────────────────────
    # 1. System prompt
    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # 2. Recent conversation history (last N turns, oldest first)
    if chat_history:
        # Keep only the last _MAX_HISTORY_TURNS complete turns
        history_msgs = [
            {"role": m["role"], "content": m["content"]}
            for m in chat_history
            if m["role"] in ("user", "assistant")
        ]
        messages.extend(history_msgs[-(  _MAX_HISTORY_TURNS * 2):])

    # 3. Current question with retrieved context injected
    user_message = (
        f"Context from the documents (use ONLY this to answer):\n"
        f"---\n{context}\n---\n\n"
        f"Question: {query}"
    )
    messages.append({"role": "user", "content": user_message})

    # ── Try each model in order ───────────────────────────────────────────────
    seen: set[str] = set()
    models_to_try = [m for m in _FALLBACK_MODELS if not (m in seen or seen.add(m))]

    last_error: Exception | None = None
    for i, model in enumerate(models_to_try):
        try:
            completion = _openrouter.chat.completions.create(
                model=model,
                messages=messages,
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
            if i < len(models_to_try) - 1:
                time.sleep(1)
            continue

    raise RuntimeError(
        "⚠️ All models are currently unavailable.\n\n"
        "Options:\n"
        "1. Wait 1–2 minutes and try again\n"
        "2. Add credit at https://openrouter.ai/settings/integrations\n\n"
        f"Last error: {last_error}"
    )
