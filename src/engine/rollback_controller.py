"""Rollback Controller — Checkpoint/restore mechanism for pipeline configurations.

Maintains a stack of pipeline configuration checkpoints, enabling automatic
rollback when a repair degrades system quality.
"""

from __future__ import annotations

import time
import json
import copy
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

ROLLBACK_DB_PATH = Path("./data/rollback_history.sqlite")


@dataclass
class Checkpoint:
    """A saved snapshot of pipeline configuration."""

    checkpoint_id: str
    """Unique identifier for this checkpoint."""

    config_snapshot: dict[str, Any]
    """The pipeline configuration at checkpoint time."""

    reason: str
    """Why this checkpoint was created."""

    created_at: float = field(default_factory=time.time)
    """Timestamp of checkpoint creation."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "reason": self.reason,
            "created_at": self.created_at,
            "created_at_human": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.created_at)),
            "config_keys": list(self.config_snapshot.keys()),
        }


class RollbackController:
    """Manages pipeline configuration checkpoints and rollback operations.

    Provides:
    - save_checkpoint: Capture current pipeline state before a repair
    - restore_checkpoint: Revert to a previous checkpoint
    - list_checkpoints: View checkpoint history
    - auto_rollback: Convenience method that restores the most recent checkpoint
    """

    def __init__(self, db_path: Path = ROLLBACK_DB_PATH, max_checkpoints: int = 50):
        self.db_path = db_path
        self.max_checkpoints = max_checkpoints
        self._checkpoints: list[Checkpoint] = []
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._load_checkpoints()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    checkpoint_id TEXT UNIQUE NOT NULL,
                    config_json TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.commit()

    def _load_checkpoints(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT checkpoint_id, config_json, reason, created_at FROM checkpoints ORDER BY created_at DESC LIMIT ?",
                (self.max_checkpoints,),
            )
            self._checkpoints = [
                Checkpoint(
                    checkpoint_id=row[0],
                    config_snapshot=json.loads(row[1]),
                    reason=row[2],
                    created_at=row[3],
                )
                for row in cursor.fetchall()
            ]

    def save_checkpoint(
        self,
        pipeline_config: dict[str, Any],
        reason: str = "Pre-repair checkpoint",
    ) -> Checkpoint:
        """Save the current pipeline configuration as a checkpoint.

        Args:
            pipeline_config: Current pipeline configuration to snapshot.
            reason: Why this checkpoint is being created.

        Returns:
            The created Checkpoint.
        """
        checkpoint_id = f"cp_{int(time.time() * 1000)}"
        config_copy = copy.deepcopy(pipeline_config)

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            config_snapshot=config_copy,
            reason=reason,
        )

        # Persist to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO checkpoints (checkpoint_id, config_json, reason, created_at) VALUES (?, ?, ?, ?)",
                (checkpoint_id, json.dumps(config_copy), reason, checkpoint.created_at),
            )
            conn.commit()

        self._checkpoints.insert(0, checkpoint)

        # Prune old checkpoints
        if len(self._checkpoints) > self.max_checkpoints:
            old = self._checkpoints.pop()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM checkpoints WHERE checkpoint_id = ?", (old.checkpoint_id,))
                conn.commit()

        console.print(f"[bold blue]💾 CHECKPOINT: Saved '{checkpoint_id}' — {reason}[/bold blue]")
        return checkpoint

    def restore_checkpoint(
        self,
        checkpoint_id: str,
        pipeline_config: dict[str, Any],
    ) -> bool:
        """Restore a specific checkpoint into the pipeline config.

        Args:
            checkpoint_id: ID of the checkpoint to restore.
            pipeline_config: Mutable pipeline config to overwrite.

        Returns:
            True if restoration succeeded, False if checkpoint not found.
        """
        checkpoint = next((c for c in self._checkpoints if c.checkpoint_id == checkpoint_id), None)

        if checkpoint is None:
            # Try loading from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT config_json, reason, created_at FROM checkpoints WHERE checkpoint_id = ?",
                    (checkpoint_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    console.print(f"[red]⚠ ROLLBACK: Checkpoint '{checkpoint_id}' not found[/red]")
                    return False
                checkpoint = Checkpoint(
                    checkpoint_id=checkpoint_id,
                    config_snapshot=json.loads(row[0]),
                    reason=row[1],
                    created_at=row[2],
                )

        # Restore the config
        pipeline_config.clear()
        pipeline_config.update(copy.deepcopy(checkpoint.config_snapshot))

        console.print(
            f"[bold red]⏪ ROLLBACK: Restored checkpoint '{checkpoint_id}' "
            f"(created: {time.strftime('%H:%M:%S', time.localtime(checkpoint.created_at))})[/bold red]"
        )
        return True

    def auto_rollback(self, pipeline_config: dict[str, Any]) -> bool:
        """Rollback to the most recent checkpoint.

        Args:
            pipeline_config: Mutable pipeline config to restore.

        Returns:
            True if rollback succeeded, False if no checkpoints exist.
        """
        if not self._checkpoints:
            console.print("[red]⚠ ROLLBACK: No checkpoints available[/red]")
            return False
        return self.restore_checkpoint(self._checkpoints[0].checkpoint_id, pipeline_config)

    def list_checkpoints(self, limit: int = 10) -> list[dict[str, Any]]:
        """List recent checkpoints.

        Args:
            limit: Maximum number of checkpoints to return.

        Returns:
            List of checkpoint summaries.
        """
        return [cp.to_dict() for cp in self._checkpoints[:limit]]

    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        """Retrieve a specific checkpoint by ID."""
        return next((c for c in self._checkpoints if c.checkpoint_id == checkpoint_id), None)


# Module-level singleton
rollback_controller = RollbackController()
