import hashlib


def make_vector_id(filename: str, chunk_index: int) -> str:
    """Generate a deterministic vector ID from filename and chunk index."""
    raw = f"{filename}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()


def clean_text(text: str) -> str:
    """Remove excessive whitespace from extracted text."""
    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)
