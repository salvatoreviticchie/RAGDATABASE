from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.ingestor import ingest, delete_document, clear_index
from src.config import list_indexes, delete_index, get_index, settings
from src.generator import answer

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")

# ── Session state initialisation ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files: set[str] = set()
if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False

# ── Sidebar — document upload ─────────────────────────────────────────────────
with st.sidebar:
    st.title("📂 Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Index Documents", type="primary"):
        for uf in uploaded_files:
            if uf.name in st.session_state.indexed_files:
                st.info(f"{uf.name} already indexed.")
                continue
            save_path = UPLOAD_DIR / uf.name
            save_path.write_bytes(uf.read())
            with st.spinner(f"Processing {uf.name}…"):
                chunks = ingest(str(save_path))
            st.success(f"Indexed {len(chunks)} chunks from **{uf.name}**")
            st.session_state.indexed_files.add(uf.name)

    if st.session_state.indexed_files:
        st.markdown("---")
        st.markdown("**Indexed files**")
        for fname in sorted(st.session_state.indexed_files):
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"📄 {fname}")
            if col2.button("🗑️", key=f"del_{fname}", help=f"Delete {fname}"):
                with st.spinner(f"Deleting {fname}…"):
                    deleted = delete_document(fname)
                    # Also remove local upload file if it exists
                    local = UPLOAD_DIR / fname
                    if local.exists():
                        local.unlink()
                st.session_state.indexed_files.discard(fname)
                st.success(f"Deleted **{fname}** ({deleted} vectors removed)")
                st.rerun()

    # ── Index management ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Index management**")
    st.caption(f"Active index: `{settings.pinecone_index_name}`")

    # ── Clear current index ───────────────────────────────────────────────────
    if not st.session_state.confirm_clear:
        if st.button("🧹 Clear entire index", use_container_width=True):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        st.warning("This will delete **all vectors** in the current index. Are you sure?")
        col1, col2 = st.columns(2)
        if col1.button("✅ Yes, clear it", use_container_width=True):
            with st.spinner("Clearing index…"):
                clear_index()
                # Remove all local upload files
                for f in UPLOAD_DIR.iterdir():
                    f.unlink()
            st.session_state.indexed_files.clear()
            st.session_state.chat_history.clear()
            st.session_state.confirm_clear = False
            st.success("Index cleared.")
            st.rerun()
        if col2.button("❌ Cancel", use_container_width=True):
            st.session_state.confirm_clear = False
            st.rerun()

    # ── Create a new named index ──────────────────────────────────────────────
    with st.expander("➕ Create new index"):
        new_index_name = st.text_input(
            "Index name", placeholder="my-new-index", key="new_index_name"
        )
        if st.button("Create index", use_container_width=True):
            if not new_index_name.strip():
                st.error("Please enter a name for the new index.")
            else:
                with st.spinner(f"Creating index '{new_index_name}'…"):
                    get_index(new_index_name.strip())
                st.success(
                    f"Index **{new_index_name}** created! "
                    f"To use it, set `PINECONE_INDEX_NAME={new_index_name}` in your `.env` and restart."
                )

    # ── List & delete existing indexes ───────────────────────────────────────
    with st.expander("🗂️ All indexes"):
        all_indexes = list_indexes()
        if not all_indexes:
            st.info("No indexes found.")
        for idx_name in all_indexes:
            col1, col2 = st.columns([4, 1])
            label = f"**{idx_name}**" + (" ← active" if idx_name == settings.pinecone_index_name else "")
            col1.markdown(label)
            if idx_name != settings.pinecone_index_name:
                if col2.button("🗑️", key=f"delidx_{idx_name}", help=f"Delete index {idx_name}"):
                    with st.spinner(f"Deleting index '{idx_name}'…"):
                        delete_index(idx_name)
                    st.success(f"Index **{idx_name}** deleted.")
                    st.rerun()
            else:
                col2.markdown("🔒")

# ── Main area — chat interface ────────────────────────────────────────────────
st.title("📄 RAG Document Q&A")
st.caption("Upload documents in the sidebar, then ask questions below.")

# Render existing chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for src in msg["sources"]:
                    meta = src["metadata"]
                    score = src["score"]
                    st.markdown(
                        f"**{meta['source_file']}** — Page {meta['page']} | "
                        f"Score: `{score:.3f}`"
                    )
                    st.text(meta["text"][:400] + ("…" if len(meta["text"]) > 400 else ""))
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a question about your documents…"):
    if not st.session_state.indexed_files:
        st.warning("Please upload and index at least one document first.")
        st.stop()

    # Display user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating… (trying available free models)"):
            try:
                result = answer(prompt)
            except RuntimeError as e:
                st.warning(str(e))
                st.stop()
        st.markdown(result.answer)
        if result.model_used:
            st.caption(f"🤖 Answered by `{result.model_used}`")
        if result.sources:
            with st.expander("Sources", expanded=False):
                for src in result.sources:
                    meta = src["metadata"]
                    score = src["score"]
                    st.markdown(
                        f"**{meta['source_file']}** — Page {meta['page']} | "
                        f"Score: `{score:.3f}`"
                    )
                    st.text(meta["text"][:400] + ("…" if len(meta["text"]) > 400 else ""))
                    st.divider()

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result.answer,
        "sources": result.sources,
    })
