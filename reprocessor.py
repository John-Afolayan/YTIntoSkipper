"""
Re-processing (audit) mode: re-analyze videos that already have intro
submissions on SponsorBlock, compare the live submission against what the
current detector suggests, and interactively fix bad ones.

Kept separate from the main pipeline because the workflow is inverted:
the main pipeline SKIPS videos that already have an intro segment, while
this module ONLY looks at those videos.

Cache: uses its own table (`reprocessed_videos`) so audit runs never
interfere with the main `processed_videos` dedup table, and can be re-run
from scratch with --ignore-cache without touching pipeline state.
"""
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Iterable, Optional

from automator import _copy_to_clipboard, _url_with_timestamp
from logger import logger

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# Statuses that are PERMANENT — the video won't be re-audited unless
# --ignore-cache is passed. error_* statuses are transient and retried.
PERMANENT_REPROCESS_STATUSES = frozenset({
    "ok",                 # existing submission within tolerance
    "corrected",          # our own segment replaced with the suggestion
    "corrected_unowned",  # suggestion submitted alongside a foreign segment
    "denied",             # user rejected the suggested correction
    "no_submission",      # video has no intro segment on SponsorBlock
    "no_detection",       # detector found no confident intro to compare
    "locked",             # segment locked by a VIP — cannot be overridden
})


class ReprocessCache:
    """
    SQLite cache of audited videos — separate table from the main pipeline's
    `processed_videos` so the two never interfere. Thread-safe via per-thread
    connections (same pattern as VideoDB).
    """

    def __init__(self, db_path: str = "intro_skipper.db"):
        self.db_path = db_path
        self._local = threading.local()
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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reprocessed_videos (
                video_id        TEXT PRIMARY KEY,
                status          TEXT NOT NULL,
                old_start       REAL,
                old_end         REAL,
                suggested_start REAL,
                suggested_end   REAL,
                checked_at      TEXT NOT NULL
            )
        """)
        conn.commit()

    def is_checked(self, video_id: str) -> bool:
        """True only for PERMANENT statuses; transient errors are retried."""
        try:
            row = self._get_conn().execute(
                "SELECT status FROM reprocessed_videos WHERE video_id = ?",
                (video_id,),
            ).fetchone()
            if row is None:
                return False
            return row["status"] in PERMANENT_REPROCESS_STATUSES
        except Exception as e:
            logger.error(f"ReprocessCache is_checked error: {e}")
            return False

    def record(self, video_id: str, status: str,
               old_start: float = None, old_end: float = None,
               suggested_start: float = None, suggested_end: float = None):
        now = datetime.now(timezone.utc).isoformat()
        try:
            self._get_conn().execute(
                """
                INSERT INTO reprocessed_videos
                    (video_id, status, old_start, old_end,
                     suggested_start, suggested_end, checked_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    status = excluded.status,
                    old_start = excluded.old_start,
                    old_end = excluded.old_end,
                    suggested_start = excluded.suggested_start,
                    suggested_end = excluded.suggested_end,
                    checked_at = excluded.checked_at
                """,
                (video_id, status, old_start, old_end,
                 suggested_start, suggested_end, now),
            )
            self._get_conn().commit()
        except Exception as e:
            logger.error(f"ReprocessCache record error: {e}")

    def get_stats(self) -> dict:
        try:
            rows = self._get_conn().execute(
                "SELECT status, COUNT(*) as cnt FROM reprocessed_videos GROUP BY status"
            ).fetchall()
            return {r["status"]: r["cnt"] for r in rows}
        except Exception as e:
            logger.error(f"ReprocessCache stats error: {e}")
            return {}

    def close(self):
        conn = getattr(self._local, "conn", None)
        if conn:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None


class IntroReprocessor:
    """
    Audits existing SponsorBlock intro submissions against fresh detections.

    Per video:
      1. Skip if in the reprocess cache (unless ignore_cache).
      2. Fetch the video's live intro segments from SponsorBlock.
      3. Re-run detection with the current reference(s)/algorithm.
      4. If |Δstart| and |Δend| are both under diff_threshold -> OK, cache.
      5. Otherwise show current vs suggested and ask for y/n approval.
         On approval: remove-and-resubmit if we own the old segment,
         downvote-and-submit-alongside if we don't.
    """

    MAX_CONSECUTIVE_ERRORS = 10

    def __init__(
        self,
        automator,                     # IntroSkipperAutomator (detection reused)
        diff_threshold: float = 0.5,   # seconds; propose a fix at/above this
        ignore_cache: bool = False,
        dry_run: bool = False,
        clipboard: bool = False,
        db_path: str = "intro_skipper.db",
    ):
        self.automator = automator
        self.sponsorblock = automator.sponsorblock
        self.downloader = automator.downloader
        self.diff_threshold = diff_threshold
        self.ignore_cache = ignore_cache
        self.dry_run = dry_run
        self.clipboard = clipboard
        self.cache = ReprocessCache(db_path)
        # Videos that need manual attention (unowned/locked), shown at the end
        self.manual_attention: list = []

    # ------------------------------------------------------------------
    # Comparison helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _closest_intro_segment(existing: list, start: float, end: float) -> Optional[dict]:
        """Pick the existing intro segment closest to the suggestion."""
        intros = [s for s in existing if s.get("category") == "intro"]
        if not intros:
            return None
        return min(
            intros,
            key=lambda s: abs(s.get("segment", [0, 0])[0] - start)
            + abs(s.get("segment", [0, 0])[1] - end),
        )

    def _segment_diff(self, old_seg: list, new_start: float, new_end: float) -> float:
        return max(abs(old_seg[0] - new_start), abs(old_seg[1] - new_end))

    # ------------------------------------------------------------------
    # Single video
    # ------------------------------------------------------------------
    def reprocess_video(self, url: str) -> str:
        """Returns one of: ok, corrected, corrected_unowned, denied,
        no_submission, no_detection, locked, skipped_cache, dry_run, failed."""
        audio_path = None
        video_id = "unknown"
        try:
            video_id = self.downloader.extract_video_id(url)

            if not self.ignore_cache and self.cache.is_checked(video_id):
                logger.info(f"Skipping {video_id} (already audited, in reprocess cache)")
                return "skipped_cache"

            # 1. What does SponsorBlock currently have?
            existing = self.sponsorblock.get_segments(video_id)
            intros = [s for s in existing if s.get("category") == "intro"]
            if not intros:
                logger.info(f"{video_id}: no intro submission on SponsorBlock — nothing to audit")
                self.cache.record(video_id, "no_submission")
                return "no_submission"

            # 2. Re-analyze with the current detector
            audio_path, video_id = self.downloader.download_audio(url)
            segment = self.automator.find_intro_in_video(audio_path)
            if segment is None:
                logger.info(f"{video_id}: detector found no confident intro — cannot audit")
                self.cache.record(
                    video_id, "no_detection",
                    old_start=intros[0]["segment"][0], old_end=intros[0]["segment"][1],
                )
                return "no_detection"

            old = self._closest_intro_segment(existing, segment.start_time, segment.end_time)
            old_seg = old.get("segment", [0.0, 0.0])
            diff = self._segment_diff(old_seg, segment.start_time, segment.end_time)

            # 3. Within tolerance — submission is fine
            if diff < self.diff_threshold:
                logger.info(
                    f"{video_id}: OK — submission {old_seg[0]:.2f}-{old_seg[1]:.2f}s vs "
                    f"suggested {segment.start_time:.2f}-{segment.end_time:.2f}s "
                    f"(max Δ {diff:.2f}s < {self.diff_threshold}s)"
                )
                self.cache.record(
                    video_id, "ok",
                    old_start=old_seg[0], old_end=old_seg[1],
                    suggested_start=segment.start_time, suggested_end=segment.end_time,
                )
                return "ok"

            # 4. Divergent — check ownership, then ask the user
            uuid = old.get("UUID")
            info_list = self.sponsorblock.get_segment_info(uuid) if uuid else []
            info = info_list[0] if info_list else {}
            owned = self.sponsorblock.is_own_segment(info) if info else False
            locked = bool(info.get("locked", 0))
            votes = info.get("votes", "?")

            self._print_comparison(url, video_id, old_seg, segment, diff, owned, locked, votes)

            if self.dry_run:
                print("  [DRY RUN] No changes made, result not cached.")
                return "dry_run"

            if locked:
                # A VIP locked this segment: votes and new submissions in the
                # same category will be rejected. Nothing we can do via API.
                print("  *** Segment is LOCKED by a SponsorBlock VIP — cannot override via API. ***")
                self.manual_attention.append((video_id, url, "locked"))
                self.cache.record(
                    video_id, "locked",
                    old_start=old_seg[0], old_end=old_seg[1],
                    suggested_start=segment.start_time, suggested_end=segment.end_time,
                )
                return "locked"

            if self.clipboard:
                clip_url = _url_with_timestamp(url, segment.start_time)
                if _copy_to_clipboard(clip_url):
                    print(f"  (Copied to clipboard: {clip_url})")

            if not self._confirm("  Apply this correction? (y/n): "):
                self.cache.record(
                    video_id, "denied",
                    old_start=old_seg[0], old_end=old_seg[1],
                    suggested_start=segment.start_time, suggested_end=segment.end_time,
                )
                return "denied"

            # 5. Apply the fix
            return self._apply_correction(video_id, uuid, owned, old_seg, segment, url)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f"Reprocess error for {url}: {e}")
            if video_id != "unknown":
                self.cache.record(video_id, "error_process")
            return "failed"
        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

    def _apply_correction(self, video_id, uuid, owned, old_seg, segment, url) -> str:
        if owned:
            # Our segment: downvoting our own submission removes it, then we
            # submit the corrected one. force=True bypasses the similar-
            # segment guard (corrections can differ by less than 2.5s).
            if uuid and not self.sponsorblock.vote_segment(uuid, self.sponsorblock.VOTE_DOWN):
                print("  Failed to remove the old segment — aborting (nothing submitted).")
                self.cache.record(video_id, "error_api",
                                  old_start=old_seg[0], old_end=old_seg[1],
                                  suggested_start=segment.start_time,
                                  suggested_end=segment.end_time)
                return "failed"
            status_if_ok = "corrected"
        else:
            # Foreign segment: we cannot remove it. Best effort — downvote it
            # and submit the correction alongside so voting can promote ours.
            print("  NOTE: old segment was submitted by another userID — it cannot be")
            print("  removed, only downvoted. Submitting the correction alongside it.")
            if uuid:
                self.sponsorblock.vote_segment(uuid, self.sponsorblock.VOTE_DOWN)
            self.manual_attention.append((video_id, url, "unowned — old segment downvoted, correction submitted alongside"))
            status_if_ok = "corrected_unowned"

        ok = self.sponsorblock.submit_segment(
            video_id, segment.start_time, segment.end_time,
            video_duration=segment.video_duration, force=True,
        )
        if not ok:
            print("  Submission of the corrected segment FAILED — see logs.")
            self.cache.record(video_id, "error_api",
                              old_start=old_seg[0], old_end=old_seg[1],
                              suggested_start=segment.start_time,
                              suggested_end=segment.end_time)
            return "failed"

        print(f"  Corrected: {old_seg[0]:.2f}-{old_seg[1]:.2f}s -> "
              f"{segment.start_time:.2f}-{segment.end_time:.2f}s")
        self.cache.record(video_id, status_if_ok,
                          old_start=old_seg[0], old_end=old_seg[1],
                          suggested_start=segment.start_time,
                          suggested_end=segment.end_time)
        return status_if_ok

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _print_comparison(self, url, video_id, old_seg, segment, diff, owned, locked, votes):
        tier = self.automator._classify_confidence(segment.confidence).upper()
        owned_str = "yes" if owned else "NO (foreign userID)"
        print(f"\n{'=' * 64}")
        print(f"Video: {url}")
        print(f"  Current submission:  {old_seg[0]:.2f}s - {old_seg[1]:.2f}s  "
              f"(votes: {votes}, ours: {owned_str}{', LOCKED' if locked else ''})")
        print(f"  Suggested segment:   {segment.start_time:.2f}s - {segment.end_time:.2f}s  "
              f"(confidence {segment.confidence:.2f} {tier})")
        print(f"  Difference:          start Δ {abs(old_seg[0] - segment.start_time):.2f}s, "
              f"end Δ {abs(old_seg[1] - segment.end_time):.2f}s  "
              f"(threshold {self.diff_threshold:.2f}s)")
        if getattr(segment, "talkover_warning", False):
            print("  *** WARNING: speech detected over the start of the suggested intro — verify carefully ***")
        print(f"{'=' * 64}")

    @staticmethod
    def _confirm(prompt: str) -> bool:
        while True:
            r = input(prompt).strip().lower()
            if r in ("y", "yes"):
                return True
            if r in ("n", "no"):
                return False

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------
    def reprocess_from_source(self, url_source: Iterable[str]):
        urls = [u.strip() for u in url_source if u.strip() and not u.strip().startswith("#")]
        total = len(urls)
        results: dict = {}
        consecutive_errors = 0

        iterator = enumerate(urls, 1)
        pbar = tqdm(iterator, total=total, desc="Auditing", unit="video") if tqdm else iterator

        try:
            for i, url in pbar:
                logger.info(f"\n--- Audit {i}/{total} ---")
                status = self.reprocess_video(url)
                results[status] = results.get(status, 0) + 1

                if status == "failed":
                    consecutive_errors += 1
                    if consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                        logger.error(
                            f"Circuit breaker: {consecutive_errors} consecutive failures. Stopping."
                        )
                        print(f"\n  *** STOPPED: {consecutive_errors} consecutive failures. Check logs. ***")
                        break
                else:
                    consecutive_errors = 0

                if tqdm and hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(
                        OK=results.get("ok", 0),
                        Fixed=results.get("corrected", 0) + results.get("corrected_unowned", 0),
                    )
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")

        # Summary
        print(f"\n{'=' * 64}\nAudit summary:")
        for status, count in sorted(results.items()):
            print(f"  {status}: {count}")
        if self.manual_attention:
            print("\nVideos needing manual attention:")
            for vid, url, reason in self.manual_attention:
                print(f"  {vid}  {reason}\n    {url}")
        print(f"{'=' * 64}")
        logger.info(f"Audit done: {results}")

    def cleanup(self):
        self.cache.close()
