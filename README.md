# RAG Document Q&A

A Retrieval-Augmented Generation (RAG) app that lets you upload PDF or TXT documents and ask questions about them.
**100% free to run** — powered by Pinecone Inference (embeddings), Pinecone serverless (vector DB), and any free OpenRouter LLM.

---

## Architecture

```
User uploads PDF / TXT
         │
         ▼
   PyMuPDF / open()  ──►  RecursiveCharacterTextSplitter
         │
         ▼
  Pinecone Inference API          (llama-text-embed-v2, 1024d)
  input_type = "passage"  ──►  Pinecone serverless index (cosine)

User types question
         │
         ▼
  Pinecone Inference API          (llama-text-embed-v2, 1024d)
  input_type = "query"   ──►  Pinecone top-k search
         │
         ▼
  OpenRouter free LLM  (google/gemma-3-27b-it:free by default)
  + retrieved context
         │
         ▼
  Answer with inline citations + expandable source chunks
```

---

## Stack

| Component      | Technology                                      | Cost  |
|----------------|-------------------------------------------------|-------|
| Vector DB      | Pinecone serverless                             | Free  |
| Embeddings     | Pinecone Inference (`llama-text-embed-v2`)      | Free  |
| LLM            | OpenRouter (`google/gemma-3-27b-it:free`)       | Free  |
| UI             | Streamlit                                       | Free  |
| PDF parsing    | PyMuPDF                                         | Free  |
| Chunking       | LangChain RecursiveCharacterTextSplitter        | Free  |

---

## Prerequisites

- Python 3.10+
- A [Pinecone](https://app.pinecone.io) account (free tier)
- An [OpenRouter](https://openrouter.ai) account (free tier)

---

## Setup

### 1. Clone the repo

```bash
git clone git@github.com:salvatoreviticchie/RAGDATABASE.git
cd RAGDATABASE
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get your API keys

**Pinecone**
1. Sign up at https://app.pinecone.io (no credit card needed)
2. In the left sidebar click **API Keys**
3. Copy your default key (starts with `pcsk-...`)

**OpenRouter**
1. Sign up at https://openrouter.ai (no credit card needed for free models)
2. Go to **Keys** → **Create key**
3. Copy your key (starts with `sk-or-...`)

### 5. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your two keys — everything else can stay as-is:

```env
PINECONE_API_KEY=pcsk-...        # your Pinecone key
OPENROUTER_API_KEY=sk-or-...     # your OpenRouter key
```

> The Pinecone index is created **automatically** the first time you run the app.

### 6. Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Usage

1. **Sidebar** → click **Browse files** → select one or more PDF or TXT files
2. Click **Index Documents** and wait for the "Indexed N chunks" confirmation
3. Type a question in the chat box at the bottom
4. Expand **Sources** under any answer to see which chunks were retrieved and their relevance scores

---

## Tuning retrieval quality

The three most impactful settings are in `.env`:

### Chunk size (`CHUNK_SIZE`)

Controls how many tokens each document chunk contains before being embedded.

| Value | Best for |
|-------|----------|
| `512` | Short documents, FAQs — fast and precise |
| `1024` | **Default** — balanced, works well for most PDFs |
| `2048` | Long reports, books — maximum context per chunk |

### Chunk overlap (`CHUNK_OVERLAP`)

How many tokens two adjacent chunks share. Prevents important sentences from being cut in half at a boundary. Aim for ~10–15% of `CHUNK_SIZE`.

| Value | Effect |
|-------|--------|
| `64` | Minimal overlap |
| `128` | **Default** — recommended |
| `256` | High overlap — better for dense technical documents |

### Top-K (`TOP_K`)

How many chunks are retrieved per question and sent to the LLM as context.

| Value | Effect |
|-------|--------|
| `5` | Focused, fast |
| `8` | **Default** — better for multi-part questions |
| `12` | Maximum — use for very long documents |

> ⚠️ **Changing `CHUNK_SIZE` or `CHUNK_OVERLAP` requires re-indexing.** Clear the index from the sidebar and re-upload your documents after changing these values.

---

## Choosing a different model

### Embedding models (Pinecone Inference — both free)

| Model                    | Dimensions | Notes                           |
|--------------------------|------------|---------------------------------|
| `llama-text-embed-v2`    | 1024       | Default — best quality          |
| `multilingual-e5-large`  | 384        | Faster, supports 100+ languages |

To switch, update `.env`:

```env
EMBEDDING_MODEL=multilingual-e5-large
EMBEDDING_DIMENSIONS=384
```

> **Important:** Changing the embedding model requires deleting and recreating the Pinecone index (dimensions change).

### LLM models (OpenRouter)

```env
# Paid (recommended — no rate limits with credit)
LLM_MODEL=meta-llama/llama-3.3-70b-instruct

# Free (may be rate-limited during peak hours)
LLM_MODEL=meta-llama/llama-3.3-70b-instruct:free
LLM_MODEL=mistralai/mistral-7b-instruct:free
LLM_MODEL=qwen/qwq-32b:free
```

The app automatically falls back through multiple models if one is unavailable.
Browse the full list at https://openrouter.ai/models

---

## Resetting the Pinecone index

Run this once in a Python shell whenever you need to recreate the index
(e.g. after changing the embedding model or dimensions):

```python
from dotenv import load_dotenv
load_dotenv()

from src.config import get_pc, settings
from pinecone import ServerlessSpec

pc = get_pc()

if settings.pinecone_index_name in [i.name for i in pc.list_indexes()]:
    pc.delete_index(settings.pinecone_index_name)
    print("Old index deleted.")

pc.create_index(
    name=settings.pinecone_index_name,
    dimension=settings.embedding_dimensions,
    metric="cosine",
    spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
)
print(f"Index '{settings.pinecone_index_name}' created with {settings.embedding_dimensions} dimensions.")
```

---

## Project structure

```
RAGDATABASE/
├── app.py                  # Streamlit UI — upload, index, chat
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .gitignore
├── README.md
└── src/
    ├── __init__.py
    ├── config.py           # Settings dataclass + Pinecone client factory
    ├── ingestor.py         # parse → chunk → embed (Pinecone) → upsert pipeline
    ├── retriever.py        # query embed (Pinecone) + vector search
    ├── generator.py        # LLM answer generation via OpenRouter
    └── utils.py            # sha256 vector ID + whitespace cleaner
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `KeyError: 'OPENROUTER_API_KEY'` | Make sure `.env` exists (not just `.env.example`) and the key is set |
| `KeyError: 'PINECONE_API_KEY'` | Same as above — check `.env` |
| `pinecone.exceptions.UnauthorizedException` | Double-check your Pinecone key for extra spaces or truncation |
| Dimension mismatch error on upsert | Your index was created with a different model — run the reset script above |
| Slow first question | Normal — Pinecone cold-starts a serverless index after idle time (~2–3 s) |
| `ModuleNotFoundError: fitz` | Run `pip install PyMuPDF` (import name differs from package name) |
| OpenRouter 429 rate limit | Free models allow ~20 req/min — wait a moment and retry |
