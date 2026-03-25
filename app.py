from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

import pandas as pd

from src.ingestor import ingest, delete_document, clear_index
from src.config import list_indexes, delete_index, get_index, settings
from src.generator import answer
from src.evaluator import evaluate, EvalScores
from src.persistence import load_files, add_file, remove_file, clear_files, remove_index
from src.audit import init_db, log as audit_log, recent as audit_recent, summary_stats, scores_over_time
from src.anonymizer import scan as pii_scan, presidio_available

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
init_db()  # ensure audit table exists


def _render_eval(scores: EvalScores) -> None:
    """Render the three RAG-Triad scores as progress bars inside an expander."""

    def _badge(score: float) -> str:
        if score >= 0.75:
            return "✅"
        if score >= 0.50:
            return "⚠️"
        return "❌"

    def _color(score: float) -> str:
        if score >= 0.75:
            return "green"
        if score >= 0.50:
            return "orange"
        return "red"

    with st.expander(
        f"📊 Answer quality  —  avg score: **{scores.average:.0%}**  {_badge(scores.average)}",
        expanded=False,
    ):
        metrics = [
            ("🔒 Groundedness",      scores.groundedness,      scores.groundedness_reason),
            ("🎯 Answer Relevance",  scores.answer_relevance,  scores.answer_relevance_reason),
            ("📚 Context Relevance", scores.context_relevance, scores.context_relevance_reason),
        ]
        for label, score, reason in metrics:
            col_label, col_bar, col_score = st.columns([2, 5, 1])
            col_label.markdown(f"**{label}**")
            col_bar.progress(score)
            col_score.markdown(
                f"<span style='color:{_color(score)};font-weight:bold'>{score:.0%}</span>",
                unsafe_allow_html=True,
            )
            if reason:
                st.caption(f"_{reason}_")
            st.markdown("")  # spacing

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")

# ── Session state initialisation ──────────────────────────────────────────────
if "active_index" not in st.session_state:
    st.session_state.active_index: str = settings.pinecone_index_name

if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []

if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False

# PII anonymization — pending query flow
if "pending_query" not in st.session_state:
    st.session_state.pending_query: str | None = None
if "pending_pii" not in st.session_state:
    st.session_state.pending_pii = None   # AnonymizeResult | None
if "anonymize_enabled" not in st.session_state:
    st.session_state.anonymize_enabled: bool = True

if "preview_file" not in st.session_state:
    st.session_state.preview_file: str | None = None

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
        file_count = len(st.session_state.indexed_files)
        with st.expander(f"📁 Indexed files ({file_count})", expanded=True):
            for fname in sorted(st.session_state.indexed_files):
                ext = fname.lower().rsplit(".", 1)[-1]
                icon = {"pdf": "📕", "docx": "📘", "doc": "📘", "txt": "📃"}.get(
                    ext, "🖼️" if ext in {"png","jpg","jpeg","webp","gif"} else "📄"
                )
                col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
                col1.markdown(f"{icon} {fname}")

                local_path = UPLOAD_DIR / fname

                # ── Preview button ───────────────────────────────────────────
                is_previewing = st.session_state.preview_file == fname
                if col2.button(
                    "👁️" if not is_previewing else "✖️",
                    key=f"prev_{fname}",
                    help="Preview file" if not is_previewing else "Close preview",
                ):
                    st.session_state.preview_file = None if is_previewing else fname
                    st.rerun()

                # ── Download button ──────────────────────────────────────────
                if local_path.exists():
                    col3.download_button(
                        label="⬇️",
                        data=local_path.read_bytes(),
                        file_name=fname,
                        mime="application/octet-stream",
                        key=f"dl_{fname}",
                        help=f"Download {fname}",
                    )
                else:
                    col3.markdown("&nbsp;", unsafe_allow_html=True)

                # ── Delete button ────────────────────────────────────────────
                if col4.button("🗑️", key=f"del_{fname}", help=f"Delete {fname}"):
                    with st.spinner(f"Deleting {fname}…"):
                        deleted = delete_document(fname, index_name=active_index)
                        if local_path.exists():
                            local_path.unlink()
                    if st.session_state.preview_file == fname:
                        st.session_state.preview_file = None
                    st.session_state.indexed_files.discard(fname)
                    remove_file(active_index, fname)
                    st.success(f"Deleted **{fname}** ({deleted} vectors removed)")
                    st.rerun()

    # ── Privacy settings ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**🔒 Privacy**")
    st.session_state.anonymize_enabled = st.toggle(
        "Scan queries for PII",
        value=st.session_state.anonymize_enabled,
        help="Detects emails, phone numbers, names etc. before sending to the LLM",
    )
    if st.session_state.anonymize_enabled:
        mode = "Presidio (NLP)" if presidio_available() else "Regex (install spaCy for full NLP)"
        st.caption(f"Detection engine: `{mode}`")

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

    # ── Analytics ─────────────────────────────────────────────────────────────
    with st.expander("📈 Analytics & Audit"):
        stats = summary_stats()
        if not stats or not stats.get("total_queries"):
            st.info("No queries logged yet.")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total queries", stats["total_queries"])
            col2.metric("Avg quality", f"{(stats['avg_quality'] or 0):.0%}")
            col3.metric("Avg groundedness", f"{(stats['avg_groundedness'] or 0):.0%}")

            # Score trend chart
            trend = scores_over_time(50)
            if len(trend) >= 2:
                df = pd.DataFrame(trend)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")
                st.caption("Quality scores over last 50 queries")
                st.line_chart(df[["avg_score", "groundedness", "answer_relevance", "context_relevance"]])

            # Recent queries table
            st.caption("Recent queries")
            rows = audit_recent(20)
            if rows:
                table_data = [
                    {
                        "Time": r["timestamp"],
                        "Query": r["query"][:60] + ("…" if len(r["query"]) > 60 else ""),
                        "Model": r["model_used"].split("/")[-1] if r["model_used"] else "",
                        "Score": f"{r['avg_score']:.0%}" if r["avg_score"] else "—",
                        "G": f"{r['groundedness']:.0%}" if r["groundedness"] else "—",
                        "AR": f"{r['answer_relevance']:.0%}" if r["answer_relevance"] else "—",
                        "CR": f"{r['context_relevance']:.0%}" if r["context_relevance"] else "—",
                    }
                    for r in rows
                ]
                st.dataframe(table_data, use_container_width=True, hide_index=True)

            # CSV export
            all_rows = audit_recent(10000)
            if all_rows:
                csv = pd.DataFrame([dict(r) for r in all_rows]).to_csv(index=False)
                st.download_button(
                    "⬇️ Export full audit log (CSV)",
                    data=csv,
                    file_name="rag_audit_log.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

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

# ── Main area — file preview ──────────────────────────────────────────────────
st.title("📄 RAG Document Q&A")
st.caption(f"Index: `{active_index}` · Upload documents in the sidebar, then ask questions below.")

if st.session_state.preview_file:
    fname = st.session_state.preview_file
    local_path = UPLOAD_DIR / fname
    ext = fname.lower().rsplit(".", 1)[-1]

    with st.expander(f"👁️ Preview — {fname}", expanded=True):
        if not local_path.exists():
            st.warning("File no longer on disk — it was indexed but the original was removed.")
        elif ext == "txt":
            st.code(local_path.read_text(errors="replace"), language="text")
        elif ext in {"png", "jpg", "jpeg", "webp", "gif"}:
            st.image(str(local_path), use_container_width=True)
        elif ext == "pdf":
            # Embed PDF in an iframe
            import base64
            b64 = base64.b64encode(local_path.read_bytes()).decode()
            st.markdown(
                f'<iframe src="data:application/pdf;base64,{b64}" '
                f'width="100%" height="600px" style="border:none;"></iframe>',
                unsafe_allow_html=True,
            )
        elif ext in {"docx", "doc"}:
            # Extract plain text for preview
            try:
                from docx import Document as DocxDocument
                doc = DocxDocument(str(local_path))
                text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
                st.text_area("Document text", text, height=400, disabled=True)
            except Exception as e:
                st.info(f"Cannot preview this file type inline. Download it to view. ({e})")
        else:
            st.info("Preview not available for this file type — use the ⬇️ button to download.")

    st.divider()

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

# ── PII warning banner (shown when a pending query has PII) ──────────────────
if st.session_state.pending_pii and st.session_state.pending_pii.has_pii:
    pii_result = st.session_state.pending_pii
    with st.container(border=True):
        st.warning("⚠️ **PII detected in your query** — review before sending:")
        for entity in pii_result.entities:
            st.markdown(f"- `{entity.original}` → **{entity.entity_type}** → will become `{entity.placeholder}`")
        st.markdown(f"**Anonymized query:** _{pii_result.anonymized}_")
        col1, col2, col3 = st.columns([2, 2, 1])
        if col1.button("🔒 Send anonymized", type="primary", use_container_width=True):
            st.session_state._send_query = pii_result.anonymized
            st.session_state._original_query = pii_result.original
            st.session_state.pending_pii = None
            st.rerun()
        if col2.button("➡️ Send as-is", use_container_width=True):
            st.session_state._send_query = pii_result.original
            st.session_state._original_query = pii_result.original
            st.session_state.pending_pii = None
            st.rerun()
        if col3.button("✏️ Edit", use_container_width=True):
            st.session_state.pending_query = None
            st.session_state.pending_pii = None
            st.rerun()
    st.stop()

# ── Resolve query: either from PII flow or fresh chat input ──────────────────
prompt = None
original_prompt = None

if hasattr(st.session_state, "_send_query") and st.session_state._send_query:
    prompt = st.session_state._send_query
    original_prompt = st.session_state._original_query
    st.session_state._send_query = None
    st.session_state._original_query = None
elif raw := st.chat_input("Ask a question about your documents…"):
    if not st.session_state.indexed_files:
        st.warning("Please upload and index at least one document first.")
        st.stop()
    # PII scan
    if st.session_state.anonymize_enabled:
        pii_result = pii_scan(raw)
        if pii_result.has_pii:
            st.session_state.pending_query = raw
            st.session_state.pending_pii = pii_result
            st.rerun()
    prompt = raw
    original_prompt = raw

if prompt:
    if not st.session_state.indexed_files:
        st.warning("Please upload and index at least one document first.")
        st.stop()

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        if original_prompt and original_prompt != prompt:
            st.caption("🔒 Query was anonymized before sending")

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

        # ── RAG Triad evaluation (non-blocking) ───────────────────────────────
        scores = None
        if result.sources:
            with st.spinner("Evaluating answer quality…"):
                scores = evaluate(prompt, result.answer, result.sources)
            if scores:
                _render_eval(scores)

        # ── Audit log ─────────────────────────────────────────────────────────
        # Log the original (pre-anonymization) query for full audit trail
        audit_log(
            index_name=active_index,
            query=original_prompt or prompt,
            answer=result.answer,
            model_used=result.model_used,
            groundedness=scores.groundedness if scores else None,
            answer_relevance=scores.answer_relevance if scores else None,
            context_relevance=scores.context_relevance if scores else None,
            avg_score=scores.average if scores else None,
            sources=result.sources,
        )

        # ── Retrieved sources ─────────────────────────────────────────────────
        if result.sources:
            with st.expander("📎 Sources", expanded=False):
                for src in result.sources:
                    meta = src["metadata"]
                    score = src["score"]
                    st.markdown(
                        f"**{meta['source_file']}** — Page {meta['page']} | "
                        f"Similarity: `{score:.3f}`"
                    )
                    st.text(meta["text"][:400] + ("…" if len(meta["text"]) > 400 else ""))
                    st.divider()

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result.answer,
        "sources": result.sources,
    })
