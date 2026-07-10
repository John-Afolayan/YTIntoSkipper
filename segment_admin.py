"""
Admin helper: remove wrongly submitted intro segments from videos.

Used via `python main.py --remove-intros <url|file> [...]`. Only needs
requests/python-dotenv — no audio stack.

For each video:
  1. Fetch its live intro segments from SponsorBlock.
  2. Segments WE submitted (public-userID match) are removed via
     self-downvote — the only removal mechanism SponsorBlock offers.
  3. Foreign segments can't be removed: they are downvoted (best effort)
     and reported for manual escalation.
  4. The video is marked `no_intro` in the local DB so the main pipeline
     never resubmits it, and `removed` in the audit cache so --reprocess
     doesn't flag it either.
"""
import os

from youtube_downloader import extract_video_id
from logger import logger


def collect_urls(items: list) -> list:
    """Each item is either a URL or a path to a text file of URLs."""
    urls = []
    for item in items:
        if os.path.isfile(item):
            with open(item, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        urls.append(line)
        else:
            urls.append(item.strip())
    # de-dup, preserve order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def remove_intro_segments(api, urls: list, db_path: str = "intro_skipper.db",
                          dry_run: bool = False) -> dict:
    """
    Remove our intro segments from the given video URLs.
    Returns a summary dict; prints per-video results.
    """
    from video_db import VideoDB
    from reprocessor import ReprocessCache

    db = VideoDB(db_path)
    audit_cache = ReprocessCache(db_path)
    summary = {"removed": 0, "foreign": 0, "none": 0, "failed": 0}
    foreign_report = []

    print(f"Our public userID: {api.get_public_user_id()[:16]}...")

    for url in urls:
        try:
            video_id = extract_video_id(url)
        except ValueError as e:
            print(f"  SKIP (bad URL): {url} — {e}")
            summary["failed"] += 1
            continue

        segments = api.get_segments(video_id)
        intros = [s for s in segments if s.get("category") == "intro"]
        if not intros:
            print(f"  {video_id}: no intro segments found — nothing to remove")
            summary["none"] += 1
            # Still mark as no-intro so the pipeline doesn't (re)submit one
            if not dry_run:
                db.record(video_id, "no_intro")
            continue

        for seg in intros:
            uuid = seg.get("UUID")
            times = seg.get("segment", [0, 0])
            info_list = api.get_segment_info(uuid) if uuid else []
            info = info_list[0] if info_list else {}
            owned = api.is_own_segment(info) if info else False

            label = f"{video_id}: [{times[0]:.2f}s - {times[1]:.2f}s] (UUID {uuid})"
            if dry_run:
                print(f"  [DRY RUN] {label} — would {'remove (ours)' if owned else 'downvote (foreign)'}")
                continue

            if owned:
                if api.vote_segment(uuid, api.VOTE_DOWN):
                    print(f"  REMOVED  {label}")
                    summary["removed"] += 1
                else:
                    print(f"  FAILED   {label} — vote request failed, see logs")
                    summary["failed"] += 1
            else:
                api.vote_segment(uuid, api.VOTE_DOWN)  # best effort
                submitter = str(info.get("userID", "unknown"))[:16]
                print(f"  FOREIGN  {label} — submitted by {submitter}..., "
                      f"downvoted but cannot remove")
                foreign_report.append((video_id, url, uuid))
                summary["foreign"] += 1

        if not dry_run:
            # Prevent the main pipeline from resubmitting and the audit
            # from re-flagging this video.
            db.record(video_id, "no_intro")
            audit_cache.record(video_id, "removed",
                               old_start=intros[0]["segment"][0],
                               old_end=intros[0]["segment"][1])

    print(f"\nSummary: removed={summary['removed']}  foreign={summary['foreign']}  "
          f"no-segments={summary['none']}  failed={summary['failed']}")
    if foreign_report:
        print("\nForeign segments that could NOT be removed (escalate to a "
              "SponsorBlock VIP in Discord #segment-review):")
        for vid, url, uuid in foreign_report:
            print(f"  {vid}  UUID={uuid}\n    {url}")

    db.close()
    audit_cache.close()
    return summary
