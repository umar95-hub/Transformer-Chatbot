"""SQLite-backed memory store for PocketSidekick."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class MemoryEntry:
    role: str
    content: str
    created_at: str


class SqliteMemory:
    """Persist and retrieve short conversational memory using SQLite."""

    def __init__(self, db_path: str = "pocketsidekick.db") -> None:
        self.db_path = Path(db_path)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def add(self, role: str, content: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO memory (role, content, created_at) VALUES (?, ?, ?)",
                (role, content, timestamp),
            )

    def recent(self, limit: int = 10) -> List[MemoryEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM memory
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [MemoryEntry(*row) for row in reversed(rows)]

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM memory")
