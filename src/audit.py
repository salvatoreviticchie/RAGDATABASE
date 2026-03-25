from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

_DB_PATH = Path("data/audit.db")


def _conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(_DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    """Create the audit table if it doesn't exist yet."""
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp         TEXT    NOT NULL,
                index_name        TEXT    NOT NULL,
                query             TEXT    NOT NULL,
                answer            TEXT    NOT NULL,
                model_used        TEXT,
                groundedness      REAL,
                answer_relevance  REAL,
                context_relevance REAL,
                avg_score         REAL,
                sources           TEXT,
                pii_detected      INTEGER DEFAULT 0
            )
        """)


def log(
    *,
    index_name: str,
    query: str,
    answer: str,
    model_used: str = "",
    groundedness: float | None = None,
    answer_relevance: float | None = None,
    context_relevance: float | None = None,
    avg_score: float | None = None,
    sources: list[dict] | None = None,
) -> None:
    """Insert one audit record."""
    sources_json = json.dumps(
        [
            {
                "source_file": s["metadata"].get("source_file", ""),
                "page": s["metadata"].get("page", ""),
                "score": s.get("score", 0),
            }
            for s in (sources or [])
        ]
    )
    with _conn() as con:
        con.execute(
            """
            INSERT INTO audit_log
              (timestamp, index_name, query, answer, model_used,
               groundedness, answer_relevance, context_relevance, avg_score, sources)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(timespec="seconds"),
                index_name,
                query,
                answer,
                model_used,
                groundedness,
                answer_relevance,
                context_relevance,
                avg_score,
                sources_json,
            ),
        )


def recent(limit: int = 200) -> list[sqlite3.Row]:
    """Return the most recent *limit* audit rows, newest first."""
    with _conn() as con:
        return con.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()


def summary_stats() -> dict:
    """Return aggregate stats across all logged requests."""
    with _conn() as con:
        row = con.execute("""
            SELECT
                COUNT(*)                        AS total_queries,
                ROUND(AVG(avg_score), 3)        AS avg_quality,
                ROUND(AVG(groundedness), 3)     AS avg_groundedness,
                ROUND(AVG(answer_relevance), 3) AS avg_answer_relevance,
                ROUND(AVG(context_relevance), 3)AS avg_context_relevance,
                MIN(timestamp)                  AS first_query,
                MAX(timestamp)                  AS last_query
            FROM audit_log
        """).fetchone()
    return dict(row) if row else {}


def scores_over_time(limit: int = 50) -> list[dict]:
    """Return the last *limit* rows with just timestamp + scores for charting."""
    with _conn() as con:
        rows = con.execute("""
            SELECT timestamp, avg_score, groundedness, answer_relevance, context_relevance
            FROM audit_log
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in reversed(rows)]  # oldest-first for charts
