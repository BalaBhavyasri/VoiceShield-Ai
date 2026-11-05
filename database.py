import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "voiceshield.db")


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables for audit logging and feedback without storing raw audio."""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                transcript TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                risk_score INTEGER NOT NULL,
                ml_score INTEGER DEFAULT 0,
                hybrid_score INTEGER DEFAULT 0,
                detected_tactics TEXT,
                language TEXT DEFAULT 'en-IN',
                audio_duration REAL DEFAULT 0.0,
                is_uncertain INTEGER DEFAULT 0,
                findings_json TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id TEXT NOT NULL,
                user_label TEXT NOT NULL,
                comments TEXT,
                submitted_at TEXT NOT NULL,
                FOREIGN KEY (analysis_id) REFERENCES analyses(id)
            )
        """)
        conn.commit()


def save_analysis_record(record: Dict[str, Any]) -> str:
    """Save an analysis audit record to SQLite."""
    init_db()
    analysis = record.get("analysis", {})
    findings = analysis.get("findings", [])
    tactics = ", ".join(f.get("category", "") for f in findings if isinstance(f, dict))

    with get_db_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO analyses (
                id, timestamp, source, transcript, risk_level, risk_score,
                ml_score, hybrid_score, detected_tactics, language,
                audio_duration, is_uncertain, findings_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record["id"],
            record.get("timestamp", datetime.now().isoformat()),
            record.get("source", "manual"),
            record.get("transcript", ""),
            analysis.get("risk_level", "Unknown"),
            int(analysis.get("risk_score", 0)),
            int(analysis.get("ml_score", 0)),
            int(analysis.get("hybrid_score", analysis.get("risk_score", 0))),
            tactics,
            record.get("language", "en-IN"),
            float(record.get("audio_duration", 0.0)),
            1 if record.get("is_uncertain") else 0,
            json.dumps(findings, ensure_ascii=False)
        ))
        conn.commit()
    return record["id"]


def get_all_records(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve recent analyses from the SQLite audit log."""
    init_db()
    records = []
    with get_db_connection() as conn:
        cursor = conn.execute("""
            SELECT id, timestamp, source, transcript, risk_level, risk_score,
                   ml_score, hybrid_score, detected_tactics, language,
                   audio_duration, is_uncertain, findings_json
            FROM analyses
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        for row in cursor.fetchall():
            findings = []
            try:
                findings = json.loads(row["findings_json"]) if row["findings_json"] else []
            except Exception:
                pass
            records.append({
                "id": row["id"],
                "timestamp": row["timestamp"],
                "source": row["source"],
                "transcript": row["transcript"],
                "language": row["language"],
                "audio_duration": row["audio_duration"],
                "is_uncertain": bool(row["is_uncertain"]),
                "analysis": {
                    "risk_level": row["risk_level"],
                    "risk_score": row["risk_score"],
                    "ml_score": row["ml_score"],
                    "hybrid_score": row["hybrid_score"],
                    "detected_tactics": row["detected_tactics"],
                    "findings": findings
                }
            })
    return records


def save_user_feedback(analysis_id: str, user_label: str, comments: Optional[str] = None):
    """Log user feedback on whether a detection was true positive, false positive, or false negative."""
    init_db()
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO feedback (analysis_id, user_label, comments, submitted_at)
            VALUES (?, ?, ?, ?)
        """, (analysis_id, user_label, comments or "", datetime.now().isoformat()))
        conn.commit()


def purge_database():
    """Purge all stored records in compliance with privacy-by-design principles."""
    init_db()
    with get_db_connection() as conn:
        conn.execute("DELETE FROM feedback")
        conn.execute("DELETE FROM analyses")
        conn.commit()
