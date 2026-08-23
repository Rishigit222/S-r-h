"""Settings, Subscription Tiers, Custom Persona Instructions & Data Controls for s@r@h.

Modeled after ChatGPT Plus/Pro, Claude Pro, and DeepSeek settings architectures.
"""

import time
import sqlite3
from pathlib import Path
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field

from src.optimization.semantic_cache import semantic_cache
from src.memory.episodic_memory import memory_store
from src.graph.knowledge_graph import knowledge_graph
from src.observability.drift_monitor import drift_monitor

SETTINGS_DB_PATH = Path("./data/user_settings.sqlite")


@dataclass
class UserSettings:
    session_id: str = "user_session_01"
    user_name: str = "Rishi"
    email: str = "rishi@openclaw.ai"
    role: str = "Lead AI Architect"
    subscription_tier: str = "free"  # "free" | "pro" | "enterprise"
    auto_refine_enabled: bool = True
    custom_instructions_user: str = "Focus on modern AI/ML architectures, high-performance RAG, and deep reasoning."
    custom_instructions_style: str = "Provide accurate, citation-grounded, production-ready code and formulas."
    reasoning_effort: str = "high"  # "low" | "medium" | "high"
    theme_accent: str = "cyan"  # "cyan" | "violet" | "emerald" | "amber"
    particles_enabled: bool = True
    updated_at: float = 0.0


class UserProfileUpdateRequest(BaseModel):
    user_name: str | None = None
    email: str | None = None
    role: str | None = None
    auto_refine_enabled: bool | None = None
    custom_instructions_user: str | None = None
    custom_instructions_style: str | None = None
    reasoning_effort: str | None = None
    theme_accent: str | None = None
    particles_enabled: bool | None = None


class TierUpgradeRequest(BaseModel):
    target_tier: str = Field(..., description="Target tier: 'free', 'pro', or 'enterprise'")


class SettingsManager:
    """Manages persistent user preferences, subscriptions, and custom instructions in SQLite."""

    def __init__(self, db_path: Path = SETTINGS_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    session_id TEXT PRIMARY KEY,
                    user_name TEXT,
                    email TEXT,
                    role TEXT,
                    subscription_tier TEXT,
                    auto_refine_enabled INTEGER,
                    custom_instructions_user TEXT,
                    custom_instructions_style TEXT,
                    reasoning_effort TEXT,
                    theme_accent TEXT,
                    particles_enabled INTEGER,
                    updated_at REAL
                )
            """)
            conn.commit()

    def get_settings(self, session_id: str = "user_session_01") -> UserSettings:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT user_name, email, role, subscription_tier, auto_refine_enabled,
                       custom_instructions_user, custom_instructions_style, reasoning_effort,
                       theme_accent, particles_enabled, updated_at
                FROM settings WHERE session_id = ?
                """,
                (session_id,)
            )
            row = cursor.fetchone()
            if not row:
                # Default initial record
                default_settings = UserSettings(session_id=session_id, updated_at=time.time())
                self.save_settings(default_settings)
                return default_settings

            return UserSettings(
                session_id=session_id,
                user_name=row[0],
                email=row[1],
                role=row[2],
                subscription_tier=row[3],
                auto_refine_enabled=bool(row[4]),
                custom_instructions_user=row[5],
                custom_instructions_style=row[6],
                reasoning_effort=row[7],
                theme_accent=row[8],
                particles_enabled=bool(row[9]),
                updated_at=row[10],
            )

    def save_settings(self, s: UserSettings):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO settings (
                    session_id, user_name, email, role, subscription_tier,
                    auto_refine_enabled, custom_instructions_user, custom_instructions_style,
                    reasoning_effort, theme_accent, particles_enabled, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_name = excluded.user_name,
                    email = excluded.email,
                    role = excluded.role,
                    subscription_tier = excluded.subscription_tier,
                    auto_refine_enabled = excluded.auto_refine_enabled,
                    custom_instructions_user = excluded.custom_instructions_user,
                    custom_instructions_style = excluded.custom_instructions_style,
                    reasoning_effort = excluded.reasoning_effort,
                    theme_accent = excluded.theme_accent,
                    particles_enabled = excluded.particles_enabled,
                    updated_at = excluded.updated_at
                """,
                (
                    s.session_id, s.user_name, s.email, s.role, s.subscription_tier,
                    1 if s.auto_refine_enabled else 0, s.custom_instructions_user,
                    s.custom_instructions_style, s.reasoning_effort, s.theme_accent,
                    1 if s.particles_enabled else 0, time.time()
                )
            )
            conn.commit()

    def update_profile(self, session_id: str, req: UserProfileUpdateRequest) -> UserSettings:
        curr = self.get_settings(session_id)
        if req.user_name is not None: curr.user_name = req.user_name
        if req.email is not None: curr.email = req.email
        if req.role is not None: curr.role = req.role
        if req.auto_refine_enabled is not None: curr.auto_refine_enabled = req.auto_refine_enabled
        if req.custom_instructions_user is not None: curr.custom_instructions_user = req.custom_instructions_user
        if req.custom_instructions_style is not None: curr.custom_instructions_style = req.custom_instructions_style
        if req.reasoning_effort is not None: curr.reasoning_effort = req.reasoning_effort
        if req.theme_accent is not None: curr.theme_accent = req.theme_accent
        if req.particles_enabled is not None: curr.particles_enabled = req.particles_enabled
        self.save_settings(curr)
        return curr

    def upgrade_tier(self, session_id: str, tier: str) -> dict:
        tier = tier.lower()
        if tier not in ["free", "pro", "enterprise"]:
            tier = "pro"
        curr = self.get_settings(session_id)
        old_tier = curr.subscription_tier
        curr.subscription_tier = tier
        self.save_settings(curr)
        return {
            "session_id": session_id,
            "old_tier": old_tier,
            "current_tier": tier,
            "status": "active",
            "message": f"Successfully activated s@r@h {tier.upper()} tier!",
        }

    def export_full_data_archive(self, session_id: str = "user_session_01") -> dict:
        """Export all user memory, telemetry logs, and knowledge graph triples as a JSON archive."""
        return {
            "export_timestamp": time.time(),
            "export_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "user_settings": asdict(self.get_settings(session_id)),
            "memory_facts": memory_store.get_user_facts(session_id),
            "memory_history": memory_store.get_recent_history(session_id, limit=50),
            "telemetry_summary": drift_monitor.get_summary_metrics(),
            "telemetry_logs": drift_monitor.get_recent_logs(limit=50),
            "knowledge_graph_sample": knowledge_graph.get_all_triples(limit=50),
        }


settings_manager = SettingsManager()
