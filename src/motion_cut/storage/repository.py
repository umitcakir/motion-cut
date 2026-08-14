from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from motion_cut.storage.models import GestureMapping, GestureTemplate


class Repository:
    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = os.environ.get("MOTION_CUT_DB", "motion_cut.db")
        self._db_path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def ensure_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS gesture_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    sample_path TEXT NOT NULL,
                    threshold REAL NOT NULL DEFAULT 0.75,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS gesture_mappings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gesture_name TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    action_payload TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE UNIQUE INDEX IF NOT EXISTS ux_gesture_mappings_name_type
                    ON gesture_mappings (gesture_name, action_type);
                """
            )

    def upsert_gesture_template(
        self,
        name: str,
        sample_path: str,
        threshold: float = 0.75,
    ) -> GestureTemplate:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO gesture_templates (name, sample_path, threshold)
                VALUES (?, ?, ?)
                ON CONFLICT(name)
                DO UPDATE SET
                    sample_path = excluded.sample_path,
                    threshold = excluded.threshold
                """,
                (name, sample_path, threshold),
            )

            row = conn.execute(
                """
                SELECT id, name, sample_path, threshold
                FROM gesture_templates
                WHERE name = ?
                """,
                (name,),
            ).fetchone()

        if row is None:
            raise RuntimeError("Failed to persist gesture template")

        return GestureTemplate(
            id=int(row["id"]),
            name=str(row["name"]),
            sample_path=str(row["sample_path"]),
            threshold=float(row["threshold"]),
        )

    def list_gesture_templates(self) -> list[GestureTemplate]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, name, sample_path, threshold
                FROM gesture_templates
                ORDER BY name COLLATE NOCASE ASC
                """
            ).fetchall()

        templates: list[GestureTemplate] = []
        for row in rows:
            templates.append(
                GestureTemplate(
                    id=int(row["id"]),
                    name=str(row["name"]),
                    sample_path=str(row["sample_path"]),
                    threshold=float(row["threshold"]),
                )
            )
        return templates

    def upsert_gesture_mapping(
        self,
        gesture_name: str,
        action_type: str,
        action_payload: str,
    ) -> GestureMapping:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO gesture_mappings (gesture_name, action_type, action_payload)
                VALUES (?, ?, ?)
                ON CONFLICT(gesture_name, action_type)
                DO UPDATE SET action_payload = excluded.action_payload
                """,
                (gesture_name, action_type, action_payload),
            )

            row = conn.execute(
                """
                SELECT id, gesture_name, action_type, action_payload
                FROM gesture_mappings
                WHERE gesture_name = ? AND action_type = ?
                """,
                (gesture_name, action_type),
            ).fetchone()

        if row is None:
            raise RuntimeError("Failed to persist gesture mapping")

        return GestureMapping(
            id=int(row["id"]),
            gesture_name=str(row["gesture_name"]),
            action_type=str(row["action_type"]),
            action_payload=str(row["action_payload"]),
        )

    def list_gesture_mappings(self) -> list[GestureMapping]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, gesture_name, action_type, action_payload
                FROM gesture_mappings
                ORDER BY gesture_name COLLATE NOCASE ASC
                """
            ).fetchall()

        mappings: list[GestureMapping] = []
        for row in rows:
            mappings.append(
                GestureMapping(
                    id=int(row["id"]),
                    gesture_name=str(row["gesture_name"]),
                    action_type=str(row["action_type"]),
                    action_payload=str(row["action_payload"]),
                )
            )
        return mappings

    def delete_gesture_template(self, gesture_name: str) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM gesture_templates
                WHERE name = ?
                """,
                (gesture_name,),
            )
        return int(cursor.rowcount or 0)

    def delete_gesture_mappings(self, gesture_name: str) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM gesture_mappings
                WHERE gesture_name = ?
                """,
                (gesture_name,),
            )
        return int(cursor.rowcount or 0)
