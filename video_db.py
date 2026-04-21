"""
Lightweight SQLite tracker so we never re-process the same video twice.
Thread-safe: uses per-thread connections.
Distinguishes permanent statuses (success, no_intro, rejected) from
transient failures (error_download, error_api, etc.) which can be retried.
"""
import sqlite3
import threading
from datetime import datetime, timezone
from logger import logger

# Statuses that are PERMANENT — video won't be retried
PERMANENT_STATUSES = frozenset({
    "success",
    "no_intro",
    "rejected",
    "skipped_existing",
    "dry_run",
    "error_permanent",  # age-restricted, private, removed — will never work
})


class VideoDB:
    def __init__(self, db_path: str = "intro_skipper.db"):
        self.db_path = db_path
        self._local = threading.local()
        self._lock = threading.Lock()
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Connection management — one connection per thread
    # ------------------------------------------------------------------
    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    def _ensure_schema(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_videos (
                video_id     TEXT PRIMARY KEY,
                status       TEXT NOT NULL,
                start_time   REAL,
                end_time     REAL,
                processed_at TEXT NOT NULL
            )
        """)
        conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_processed(self, video_id: str) -> bool:
        """
        Returns True only for PERMANENT statuses.
        Transient errors (error_*) return False so the video is retried.
        """
        try:
            row = self._get_conn().execute(
                "SELECT status FROM processed_videos WHERE video_id = ?",
                (video_id,),
            ).fetchone()
            if row is None:
                return False
            return row["status"] in PERMANENT_STATUSES
        except Exception as e:
            logger.error(f"DB is_processed error: {e}")
            return False

    def record(
        self,
        video_id: str,
        status: str,
        start_time: float = None,
        end_time: float = None,
    ):
        now = datetime.now(timezone.utc).isoformat()
        try:
            self._get_conn().execute(
                """
                INSERT INTO processed_videos (video_id, status, start_time, end_time, processed_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    status = excluded.status,
                    start_time = excluded.start_time,
                    end_time = excluded.end_time,
                    processed_at = excluded.processed_at
                """,
                (video_id, status, start_time, end_time, now),
            )
            self._get_conn().commit()
        except Exception as e:
            logger.error(f"DB record error: {e}")

    def get_stats(self) -> dict:
        try:
            rows = self._get_conn().execute(
                "SELECT status, COUNT(*) as cnt FROM processed_videos GROUP BY status"
            ).fetchall()
            return {r["status"]: r["cnt"] for r in rows}
        except Exception as e:
            logger.error(f"DB stats error: {e}")
            return {}

    def reset(self):
        """Clear all records (useful for re-runs)."""
        self._get_conn().execute("DELETE FROM processed_videos")
        self._get_conn().commit()
        logger.info("Database cleared.")

    def reset_errors(self):
        """Clear only transient error records so they can be retried."""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM processed_videos WHERE status NOT IN ({})".format(
                ",".join("?" for _ in PERMANENT_STATUSES)
            ),
            tuple(PERMANENT_STATUSES),
        )
        conn.commit()
        logger.info(f"Cleared {cursor.rowcount} error/transient records for retry.")

    def close(self):
        conn = getattr(self._local, "conn", None)
        if conn:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None
