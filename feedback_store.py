"""
SQLite-backed storage for user feedback and per-channel learning profiles.

Feedback entries record what the detector predicted vs what the user corrected,
along with structured reasons. Channel profiles aggregate this into statistical
corrections that the AdaptiveEngine applies to future detections.

Thread-safe: uses per-thread connections (same pattern as VideoDB).
"""
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional, List, Dict
from logger import logger


REASON_CATEGORIES = {
    "1": "wrong_start",
    "2": "wrong_end",
    "3": "no_intro",
    "4": "too_short",
    "5": "too_long",
    "6": "other",
}

REASON_LABELS = {
    "1": "Wrong start",
    "2": "Wrong end",
    "3": "No intro",
    "4": "Too short",
    "5": "Too long",
    "6": "Other",
}


class FeedbackStore:
    MIN_SAMPLES_FOR_PROFILE = 5

    def __init__(self, db_path: str = "intro_skipper.db"):
        self.db_path = db_path
        self._local = threading.local()
        self._lock = threading.Lock()
        self._ensure_schema()

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
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS feedback (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id        TEXT NOT NULL,
                channel_id      TEXT,
                detected_start  REAL,
                detected_end    REAL,
                correct_start   REAL,
                correct_end     REAL,
                reason_categories TEXT,
                reason_text     TEXT,
                action          TEXT NOT NULL,
                raw_confidence  REAL,
                weighted_score  REAL,
                created_at      TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_feedback_channel
                ON feedback(channel_id);
            CREATE INDEX IF NOT EXISTS idx_feedback_action
                ON feedback(action);

            CREATE TABLE IF NOT EXISTS channel_profiles (
                channel_id          TEXT PRIMARY KEY,
                avg_intro_duration  REAL,
                avg_start_offset    REAL,
                duration_stddev     REAL,
                start_stddev        REAL,
                avg_end_error       REAL,
                avg_start_error     REAL,
                confidence_bias     REAL DEFAULT 0.0,
                false_positive_rate REAL DEFAULT 0.0,
                sample_count        INTEGER DEFAULT 0,
                last_recalculated   TEXT
            );
        """)
        conn.commit()

    # ------------------------------------------------------------------
    # Record feedback
    # ------------------------------------------------------------------
    def record_feedback(
        self,
        video_id: str,
        action: str,
        channel_id: str = None,
        detected_start: float = None,
        detected_end: float = None,
        correct_start: float = None,
        correct_end: float = None,
        reason_categories: str = None,
        reason_text: str = None,
        raw_confidence: float = None,
        weighted_score: float = None,
    ):
        now = datetime.now(timezone.utc).isoformat()
        try:
            self._get_conn().execute(
                """
                INSERT INTO feedback (
                    video_id, channel_id, detected_start, detected_end,
                    correct_start, correct_end, reason_categories, reason_text,
                    action, raw_confidence, weighted_score, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    video_id, channel_id, detected_start, detected_end,
                    correct_start, correct_end, reason_categories, reason_text,
                    action, raw_confidence, weighted_score, now,
                ),
            )
            self._get_conn().commit()
            logger.debug(f"Feedback recorded: {action} for {video_id}")
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")

    # ------------------------------------------------------------------
    # Query feedback
    # ------------------------------------------------------------------
    def get_channel_feedback(self, channel_id: str) -> List[Dict]:
        try:
            rows = self._get_conn().execute(
                "SELECT * FROM feedback WHERE channel_id = ? ORDER BY created_at",
                (channel_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Failed to query feedback: {e}")
            return []

    def get_approved_for_channel(self, channel_id: str) -> List[Dict]:
        try:
            rows = self._get_conn().execute(
                """SELECT * FROM feedback
                   WHERE channel_id = ? AND action = 'approved'
                   ORDER BY created_at""",
                (channel_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Failed to query approved feedback: {e}")
            return []

    def get_denied_for_channel(self, channel_id: str) -> List[Dict]:
        try:
            rows = self._get_conn().execute(
                """SELECT * FROM feedback
                   WHERE channel_id = ? AND action = 'denied'
                   ORDER BY created_at""",
                (channel_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Failed to query denied feedback: {e}")
            return []

    def get_corrected_for_channel(self, channel_id: str) -> List[Dict]:
        """Feedback where user provided corrected times (approved or denied)."""
        try:
            rows = self._get_conn().execute(
                """SELECT * FROM feedback
                   WHERE channel_id = ?
                     AND (correct_start IS NOT NULL OR correct_end IS NOT NULL)
                   ORDER BY created_at""",
                (channel_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Failed to query corrected feedback: {e}")
            return []

    def count_feedback(self, channel_id: str) -> Dict[str, int]:
        try:
            rows = self._get_conn().execute(
                """SELECT action, COUNT(*) as cnt
                   FROM feedback WHERE channel_id = ?
                   GROUP BY action""",
                (channel_id,),
            ).fetchall()
            return {r["action"]: r["cnt"] for r in rows}
        except Exception as e:
            logger.error(f"Failed to count feedback: {e}")
            return {}

    # ------------------------------------------------------------------
    # Channel profile CRUD
    # ------------------------------------------------------------------
    def get_channel_profile(self, channel_id: str) -> Optional[Dict]:
        try:
            row = self._get_conn().execute(
                "SELECT * FROM channel_profiles WHERE channel_id = ?",
                (channel_id,),
            ).fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Failed to get channel profile: {e}")
            return None

    def save_channel_profile(self, profile: Dict):
        now = datetime.now(timezone.utc).isoformat()
        try:
            self._get_conn().execute(
                """
                INSERT INTO channel_profiles (
                    channel_id, avg_intro_duration, avg_start_offset,
                    duration_stddev, start_stddev, avg_end_error, avg_start_error,
                    confidence_bias, false_positive_rate,
                    sample_count, last_recalculated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(channel_id) DO UPDATE SET
                    avg_intro_duration = excluded.avg_intro_duration,
                    avg_start_offset = excluded.avg_start_offset,
                    duration_stddev = excluded.duration_stddev,
                    start_stddev = excluded.start_stddev,
                    avg_end_error = excluded.avg_end_error,
                    avg_start_error = excluded.avg_start_error,
                    confidence_bias = excluded.confidence_bias,
                    false_positive_rate = excluded.false_positive_rate,
                    sample_count = excluded.sample_count,
                    last_recalculated = excluded.last_recalculated
                """,
                (
                    profile["channel_id"],
                    profile.get("avg_intro_duration"),
                    profile.get("avg_start_offset"),
                    profile.get("duration_stddev"),
                    profile.get("start_stddev"),
                    profile.get("avg_end_error"),
                    profile.get("avg_start_error"),
                    profile.get("confidence_bias", 0.0),
                    profile.get("false_positive_rate", 0.0),
                    profile.get("sample_count", 0),
                    now,
                ),
            )
            self._get_conn().commit()
        except Exception as e:
            logger.error(f"Failed to save channel profile: {e}")

    def get_all_profiles(self) -> List[Dict]:
        try:
            rows = self._get_conn().execute(
                "SELECT * FROM channel_profiles ORDER BY channel_id"
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Failed to get all profiles: {e}")
            return []

    # ------------------------------------------------------------------
    # Denial pattern queries
    # ------------------------------------------------------------------
    def get_denial_patterns(self, channel_id: str) -> Dict[str, int]:
        """Count how many times each reason category appears in denials."""
        try:
            rows = self._get_conn().execute(
                """SELECT reason_categories FROM feedback
                   WHERE channel_id = ? AND action = 'denied'
                     AND reason_categories IS NOT NULL""",
                (channel_id,),
            ).fetchall()
            counts: Dict[str, int] = {}
            for row in rows:
                for cat in row["reason_categories"].split(","):
                    cat = cat.strip()
                    if cat:
                        counts[cat] = counts.get(cat, 0) + 1
            return counts
        except Exception as e:
            logger.error(f"Failed to get denial patterns: {e}")
            return {}

    def close(self):
        conn = getattr(self._local, "conn", None)
        if conn:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None
