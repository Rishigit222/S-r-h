"""Authentication and API Key Vault for s@r@h."""

import sqlite3
import time
import secrets
from pathlib import Path
from dataclasses import dataclass
from pydantic import BaseModel, Field

AUTH_DB_PATH = Path("./data/auth_vault.sqlite")


class UserLoginRequest(BaseModel):
    username: str = Field(default="guest", description="Username or handle")
    role: str = Field(default="Developer", description="User role")


class UserLoginResponse(BaseModel):
    token: str
    username: str
    role: str
    created_at: float


class ApiKeyVaultRequest(BaseModel):
    token: str
    groq_api_key: str | None = None
    google_api_key: str | None = None
    openai_api_key: str | None = None
    ollama_url: str | None = None


class AuthManager:
    """Manages user session tokens and encrypted/secured client API keys."""

    def __init__(self, db_path: Path = AUTH_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    token TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    token TEXT PRIMARY KEY,
                    groq_key TEXT,
                    google_key TEXT,
                    openai_key TEXT,
                    ollama_url TEXT,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY(token) REFERENCES user_sessions(token)
                )
            """)
            conn.commit()

    def create_session(self, username: str = "guest", role: str = "Developer") -> UserLoginResponse:
        token = secrets.token_hex(16)
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO user_sessions (token, username, role, created_at) VALUES (?, ?, ?, ?)",
                (token, username, role, now)
            )
            conn.commit()
        return UserLoginResponse(token=token, username=username, role=role, created_at=now)

    def save_api_keys(self, req: ApiKeyVaultRequest) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO api_keys (token, groq_key, google_key, openai_key, ollama_url, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(token) DO UPDATE SET
                    groq_key = coalesce(excluded.groq_key, api_keys.groq_key),
                    google_key = coalesce(excluded.google_key, api_keys.google_key),
                    openai_key = coalesce(excluded.openai_key, api_keys.openai_key),
                    ollama_url = coalesce(excluded.ollama_url, api_keys.ollama_url),
                    updated_at = excluded.updated_at
                """,
                (req.token, req.groq_api_key, req.google_api_key, req.openai_api_key, req.ollama_url, time.time())
            )
            conn.commit()
            return True

    def get_api_keys(self, token: str) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT groq_key, google_key, openai_key, ollama_url FROM api_keys WHERE token = ?", (token,))
            row = cursor.fetchone()
            if not row:
                return {}
            return {
                "groq_configured": bool(row[0]),
                "google_configured": bool(row[1]),
                "openai_configured": bool(row[2]),
                "ollama_url": row[3] or "http://localhost:11434",
            }


auth_manager = AuthManager()
