# RAG Document Q&A

A Retrieval-Augmented Generation (RAG) app that lets you upload PDF or TXT documents and ask questions about them. Built with Pinecone, OpenAI, and Streamlit.

## Architecture

```
User uploads PDF/TXT
        │
        ▼
  PyMuPDF / open()   ──► RecursiveCharacterTextSplitter
        │
        ▼
  OpenAI text-embedding-3-small   ──► Pinecone (cosine, 1536d)

User types question
        │
        ▼
  Embed query  ──►  Pinecone top-k search
        │
        ▼
  GPT-4o (temperature=0) + retrieved context
        │
        ▼
  Answer with inline citations + expandable source chunks
```

## Stack

| Component     | Technology                        |
|---------------|-----------------------------------|
| Vector DB     | Pinecone (serverless)             |
| Embeddings    | OpenAI text-embedding-3-small     |
| LLM           | OpenAI GPT-4o                     |
| UI            | Streamlit                         |
| PDF parsing   | PyMuPDF                           |
| Chunking      | LangChain RecursiveCharacterTextSplitter |

## Setup

### 1. Clone and install

```bash
git clone git@github.com:salvatoreviticchie/RAGDATABASE.git
cd RAGDATABASE
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk-...
PINECONE_INDEX_NAME=rag-docs
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

- Get an OpenAI key at https://platform.openai.com/api-keys
- Get a Pinecone key at https://app.pinecone.io — create a free serverless index

### 3. Run

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Usage

1. **Upload documents** — use the sidebar to upload one or more PDF or TXT files
2. **Click "Index Documents"** — chunks are embedded and stored in Pinecone
3. **Ask a question** — type in the chat input at the bottom
4. The app returns a grounded answer with source citations and expandable chunk previews

## Project Structure

```
├── app.py                  # Streamlit entry point
├── src/
│   ├── config.py           # Settings + Pinecone index factory
│   ├── ingestor.py         # Parse → chunk → embed → upsert
│   ├── retriever.py        # Query embedding + Pinecone search
│   ├── generator.py        # GPT-4o answer generation
│   └── utils.py            # ID hashing, text cleaning
├── requirements.txt
├── .env.example
└── .gitignore
```
