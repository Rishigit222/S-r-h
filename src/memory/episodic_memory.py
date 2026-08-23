"""Option B: Persistent Memory Agent for s@r@h.

Features:
- Multi-turn conversation persistence in SQLite
- Automatic entity and preference extraction (e.g., user name, favorite tech, project context)
- Dynamic contextual memory injection into RAG prompts
"""

import sqlite3
import re
import time
from pathlib import Path
from dataclasses import dataclass
from rich.console import Console

console = Console()

DB_PATH = Path("./data/memory.sqlite")


class EpisodicMemoryStore:
    """Manages persistent conversational memory and user preferences."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    fact_key TEXT NOT NULL,
                    fact_value TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(session_id, fact_key)
                )
            """)
            conn.commit()

    def add_message(self, session_id: str, role: str, content: str):
        """Append a message turn to conversation history and extract facts."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, role, content, time.time())
            )
            conn.commit()

        # Automatic preference extraction on user messages
        if role == "user":
            self._auto_extract_facts(session_id, content)

    def _auto_extract_facts(self, session_id: str, text: str):
        """Automatically extract user facts / entities from message."""
        name_match = re.search(r"\bmy name is ([A-Z][a-z]+)\b", text, re.IGNORECASE)
        if name_match:
            self.set_user_fact(session_id, "user_name", name_match.group(1).capitalize())

        pref_match = re.search(r"\bi (?:prefer|like|use|work with) ([A-Za-z0-9_]+)\b", text, re.IGNORECASE)
        if pref_match:
            self.set_user_fact(session_id, "favorite_tech", pref_match.group(1))

        role_match = re.search(r"\bi am an? ([A-Za-z0-9_\s]+(?:developer|engineer|scientist|student|analyst))\b", text, re.IGNORECASE)
        if role_match:
            self.set_user_fact(session_id, "user_role", role_match.group(1).strip())

    def get_recent_history(self, session_id: str, limit: int = 6) -> list[dict]:
        """Fetch the latest N conversation turns."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit)
            )
            rows = cursor.fetchall()
            return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in reversed(rows)]

    def set_user_fact(self, session_id: str, key: str, value: str):
        """Store a long-term user fact or preference."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO user_facts (session_id, fact_key, fact_value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id, fact_key) DO UPDATE SET
                    fact_value = excluded.fact_value,
                    updated_at = excluded.updated_at
                """,
                (session_id, key, value, time.time())
            )
            conn.commit()
        console.print(f"[cyan]🧠 Memory Agent: Remembered fact '{key}' = '{value}' for {session_id}[/cyan]")

    def get_user_facts(self, session_id: str) -> dict[str, str]:
        """Retrieve all stored user facts for a session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT fact_key, fact_value FROM user_facts WHERE session_id = ?", (session_id,))
            rows = cursor.fetchall()
            return {r[0]: r[1] for r in rows}

    def clear_session(self, session_id: str):
        """Reset history and facts for a session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM user_facts WHERE session_id = ?", (session_id,))
            conn.commit()

    def clear_history(self, session_id: str):
        """Alias for clear_session."""
        self.clear_session(session_id)


memory_store = EpisodicMemoryStore()
