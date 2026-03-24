import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


@dataclass
class Settings:
    openai_api_key: str = field(default_factory=lambda: os.environ["OPENAI_API_KEY"])
    pinecone_api_key: str = field(default_factory=lambda: os.environ["PINECONE_API_KEY"])
    pinecone_index_name: str = field(default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "rag-docs"))
    pinecone_cloud: str = field(default_factory=lambda: os.getenv("PINECONE_CLOUD", "aws"))
    pinecone_region: str = field(default_factory=lambda: os.getenv("PINECONE_REGION", "us-east-1"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o"))
    embedding_dimensions: int = 1536
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "512")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "64")))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))


settings = Settings()


def get_index():
    """Return a Pinecone Index object, creating the index if it does not exist."""
    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing = [idx.name for idx in pc.list_indexes()]
    if settings.pinecone_index_name not in existing:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dimensions,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )
    return pc.Index(settings.pinecone_index_name)
