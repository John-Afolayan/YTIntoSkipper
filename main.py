import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the same directory as this script, not cwd.
# load_dotenv() defaults to searching from cwd which may differ.
_script_dir = Path(__file__).resolve().parent
_env_path = _script_dir / ".env"
_env_loaded = load_dotenv(dotenv_path=_env_path)
if not _env_loaded:
    # Also try cwd as fallback
    _env_loaded = load_dotenv()
if not _env_loaded:
    print(f"Note: No .env file found (searched {_env_path} and cwd={os.getcwd()})",
          file=sys.stderr)

from automator import IntroSkipperAutomator
from sponsorblock_api import SponsorBlockAPI
from logger import logger

from populate_urls import yt_dlp_stream_list, build_watch_url_from_entry


def url_generator_from_file(filepath):
    """Yields URLs from a text file."""
    with open(filepath, "r") as f:
        for line in f:
            yield line


def url_generator_from_channel(channel_url):
    """Yields URLs dynamically from a YouTube channel."""
    logger.info(f"Fetching video list from channel: {channel_url}")
    stream = yt_dlp_stream_list(channel_url, fast=True)
    for entry in stream:
        url = build_watch_url_from_entry(entry)
        if url:
            yield url


def main():
    parser = argparse.ArgumentParser(
        description="Automated YouTube intro detection and SponsorBlock submission",
    )

    # --- Input ---
    parser.add_argument(
        "--reference-intro",
        action="append",
        dest="reference_intros",
        help="Path to reference intro audio file (can be specified multiple times)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--urls-file", help="Text file containing YouTube URLs (one per line)")
    group.add_argument("--channel", help="YouTube Channel URL (processes all videos)")

    # --- Behaviour ---
    parser.add_argument("--intro-duration", type=float, default=None,
                        help="Override intro duration in seconds (default: auto-detect from reference)")
    parser.add_argument("--manual-approval", action="store_true",
                        help="Require manual approval before submitting")
    parser.add_argument("--dry-run", action="store_true",
                        help="Detect intros but don't submit to SponsorBlock")
    parser.add_argument("--clipboard", action="store_true",
                        help="Copy video URL (with timestamp) to clipboard during manual approval")
    parser.add_argument("--trim-speech", action="store_true",
                        help="Detect when the YouTuber talks over the end of the intro and trim the skip point earlier")

    # --- Tuning ---
    parser.add_argument("--peak-height", type=float, default=0.25,
                        help="Minimum raw correlation peak height (default: 0.25)")
    parser.add_argument("--weighted-threshold", type=float, default=0.60,
                        help="Minimum weighted score to accept a match (default: 0.60)")
    parser.add_argument("--search-limit", type=float, default=120.0,
                        help="How many seconds of each video to scan (default: 120)")
    parser.add_argument("--early-exit", type=float, default=0.90,
                        help="Score above which to stop scanning early (default: 0.90)")

    # --- Performance ---
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel download+analyse workers (default: 1, use 2-4 for speed)")

    # --- SponsorBlock / Admin ---
    parser.add_argument("--user-id", help="SponsorBlock user ID")
    parser.add_argument("--delete-video", help="Delete intro segments for a specific video ID")
    parser.add_argument("--list-segments", help="List all segments for a video ID")

    # --- Database ---
    parser.add_argument("--db", default="intro_skipper.db", help="SQLite database path")
    parser.add_argument("--db-stats", action="store_true", help="Print database statistics and exit")
    parser.add_argument("--db-reset", action="store_true", help="Clear the database and exit")
    parser.add_argument("--db-reset-errors", action="store_true",
                        help="Clear only transient error records (so failed videos are retried) and exit")

    args = parser.parse_args()

    # --- DB commands ---
    if args.db_stats:
        from video_db import VideoDB
        db = VideoDB(args.db)
        stats = db.get_stats()
        if stats:
            print("Video processing stats:")
            for status, count in sorted(stats.items()):
                print(f"  {status}: {count}")
            print(f"  TOTAL: {sum(stats.values())}")
        else:
            print("No records yet.")
        db.close()
        return

    if args.db_reset:
        from video_db import VideoDB
        db = VideoDB(args.db)
        db.reset()
        print("Database cleared.")
        db.close()
        return

    if args.db_reset_errors:
        from video_db import VideoDB
        db = VideoDB(args.db)
        db.reset_errors()
        db.close()
        return

    # --- List / Delete commands ---
    if args.list_segments:
        user_id = args.user_id or os.getenv("SPONSORBLOCK_USER_ID")
        api = SponsorBlockAPI(user_id)
        segments = api.get_segments(args.list_segments)
        if segments:
            for s in segments:
                seg = s.get("segment", [])
                print(f"  [{s.get('category')}] {seg[0]:.2f}s - {seg[1]:.2f}s  (UUID: {s.get('UUID', 'N/A')})")
        else:
            print("No segments found.")
        return

    if args.delete_video:
        # TODO: implement deletion via UUID
        print("Deletion not yet implemented via this CLI.")
        return

    # --- Processing mode ---
    if not args.reference_intros:
        parser.error("--reference-intro is required for processing mode")

    if not args.urls_file and not args.channel:
        parser.error("You must provide either --urls-file OR --channel")

    automator = IntroSkipperAutomator(
        manual_approval=args.manual_approval,
        dry_run=args.dry_run,
        clipboard=args.clipboard,
        trim_speech=args.trim_speech,
        peak_height=args.peak_height,
        weighted_threshold=args.weighted_threshold,
        search_limit=args.search_limit,
        early_exit_threshold=args.early_exit,
        workers=args.workers,
        db_path=args.db,
    )

    # Load reference intro(s)
    for i, ref_path in enumerate(args.reference_intros):
        if i == 0:
            automator.set_reference_intro(ref_path, args.intro_duration)
        else:
            automator.add_reference_intro(ref_path, args.intro_duration)

    if args.user_id:
        automator.sponsorblock.user_id = args.user_id

    # Log the active SponsorBlock user ID at the start of each run
    sb_uid = automator.sponsorblock.user_id
    logger.info(f"SPONSORBLOCK_USER_ID = {sb_uid}")
    print(f"SPONSORBLOCK_USER_ID = {sb_uid}")

    try:
        if args.channel:
            source = url_generator_from_channel(args.channel)
            automator.process_from_source(source)
        else:
            source = url_generator_from_file(args.urls_file)
            automator.process_from_source(source)
    finally:
        automator.cleanup()


if __name__ == "__main__":
    main()
