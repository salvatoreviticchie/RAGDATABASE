from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from src.ingestor import ingest, delete_document, clear_index
from src.config import list_indexes, delete_index, get_index, settings
from src.generator import answer
from src.persistence import load_files, add_file, remove_file, clear_files, remove_index

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")

# ── Session state initialisation ──────────────────────────────────────────────
if "active_index" not in st.session_state:
    st.session_state.active_index: str = settings.pinecone_index_name

if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []

if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False

# Load persisted file list for the active index on every startup
# (uses the JSON file so it survives restarts)
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files: set[str] = load_files(st.session_state.active_index)

active_index = st.session_state.active_index

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── Index switcher ────────────────────────────────────────────────────────
    st.markdown("**🗂️ Active index**")
    all_indexes = list_indexes()
    if all_indexes:
        selected = st.selectbox(
            "Switch index",
            options=all_indexes,
            index=all_indexes.index(active_index) if active_index in all_indexes else 0,
            label_visibility="collapsed",
        )
        if selected != active_index:
            # Switch to the new index — reload its file list and clear chat
            st.session_state.active_index = selected
            st.session_state.indexed_files = load_files(selected)
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.caption(f"`{active_index}`")

    st.markdown("---")

    # ── Document upload ───────────────────────────────────────────────────────
    st.title("📂 Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT or images",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "webp", "gif"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Index Documents", type="primary"):
        for uf in uploaded_files:
            if uf.name in st.session_state.indexed_files:
                st.info(f"{uf.name} already indexed.")
                continue
            save_path = UPLOAD_DIR / uf.name
            save_path.write_bytes(uf.read())
            is_image = uf.name.lower().rsplit(".", 1)[-1] in {"png", "jpg", "jpeg", "webp", "gif"}
            spin_msg = f"Analysing image with vision model… {uf.name}" if is_image else f"Processing {uf.name}…"
            with st.spinner(spin_msg):
                chunks = ingest(str(save_path), index_name=active_index)
            st.success(f"Indexed {len(chunks)} chunks from **{uf.name}**")
            st.session_state.indexed_files.add(uf.name)
            add_file(active_index, uf.name)  # persist to JSON

    if st.session_state.indexed_files:
        st.markdown("---")
        st.markdown("**Indexed files**")
        for fname in sorted(st.session_state.indexed_files):
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"📄 {fname}")
            if col2.button("🗑️", key=f"del_{fname}", help=f"Delete {fname}"):
                with st.spinner(f"Deleting {fname}…"):
                    deleted = delete_document(fname, index_name=active_index)
                    local = UPLOAD_DIR / fname
                    if local.exists():
                        local.unlink()
                st.session_state.indexed_files.discard(fname)
                remove_file(active_index, fname)  # remove from JSON
                st.success(f"Deleted **{fname}** ({deleted} vectors removed)")
                st.rerun()

    # ── Index management ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Index management**")

    # Clear current index
    if not st.session_state.confirm_clear:
        if st.button("🧹 Clear entire index", use_container_width=True):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        st.warning("This will delete **all vectors** in the current index. Are you sure?")
        col1, col2 = st.columns(2)
        if col1.button("✅ Yes, clear it", use_container_width=True):
            with st.spinner("Clearing index…"):
                clear_index(index_name=active_index)
                for f in UPLOAD_DIR.iterdir():
                    f.unlink()
            st.session_state.indexed_files.clear()
            st.session_state.chat_history.clear()
            st.session_state.confirm_clear = False
            clear_files(active_index)  # clear JSON
            st.success("Index cleared.")
            st.rerun()
        if col2.button("❌ Cancel", use_container_width=True):
            st.session_state.confirm_clear = False
            st.rerun()

    # Create a new named index
    with st.expander("➕ Create new index"):
        new_index_name = st.text_input(
            "Index name", placeholder="my-new-index", key="new_index_name"
        )
        sanitized = re.sub(r"[^a-z0-9-]", "-", new_index_name.strip().lower()).strip("-")
        sanitized = re.sub(r"-+", "-", sanitized)
        if sanitized and sanitized != new_index_name.strip():
            st.caption(f"ℹ️ Will be created as: `{sanitized}`")
        if st.button("Create index", use_container_width=True):
            if not sanitized:
                st.error("Please enter a valid name (letters, numbers, hyphens only).")
            else:
                with st.spinner(f"Creating index '{sanitized}'…"):
                    get_index(sanitized)
                # Auto-switch to the newly created index
                st.session_state.active_index = sanitized
                st.session_state.indexed_files = set()
                st.session_state.chat_history = []
                st.success(f"Index **{sanitized}** created and switched!")
                st.rerun()

    # List & delete existing indexes
    with st.expander("🗂️ All indexes"):
        if not all_indexes:
            st.info("No indexes found.")
        for idx_name in all_indexes:
            col1, col2 = st.columns([4, 1])
            label = f"**{idx_name}**" + (" ← active" if idx_name == active_index else "")
            col1.markdown(label)
            if idx_name != active_index:
                if col2.button("🗑️", key=f"delidx_{idx_name}", help=f"Delete index {idx_name}"):
                    with st.spinner(f"Deleting index '{idx_name}'…"):
                        delete_index(idx_name)
                    remove_index(idx_name)  # remove from JSON
                    st.success(f"Index **{idx_name}** deleted.")
                    st.rerun()
            else:
                col2.markdown("🔒")

# ── Main area — chat interface ────────────────────────────────────────────────
st.title("📄 RAG Document Q&A")
st.caption(f"Index: `{active_index}` · Upload documents in the sidebar, then ask questions below.")

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

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            try:
                result = answer(
                    prompt,
                    chat_history=st.session_state.chat_history,
                    index_name=active_index,
                )
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
