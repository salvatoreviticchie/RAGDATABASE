from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.ingestor import ingest, delete_document
from src.generator import answer

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")

# ── Session state initialisation ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files: set[str] = set()

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
