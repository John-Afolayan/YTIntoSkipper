# populate_urls.py
from dotenv import load_dotenv
import os
import subprocess
import json
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List, Set, Iterator, Tuple

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

LOG_LEVEL = os.getenv("POPULATE_LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("populate_urls")
logger.setLevel(getattr(logging, LOG_LEVEL))


def parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.date()
        except Exception:
            continue
    raise ValueError(f"Unsupported date format: {s}. Use YYYY or YYYY-MM-DD or YYYYMMDD")


def _yt_dlp_cmd_for_channel(channel_url: str, fast: bool) -> List[str]:
    """
    Build the yt-dlp command for listing a channel/playlist.
    fast=True => --flat-playlist (minimal, very fast)
    fast=False => full --dump-json (includes upload_date and other metadata)
    """
    if fast:
        return ["yt-dlp", "--yes-playlist", "--flat-playlist", "--skip-download", "--dump-json", channel_url]
    else:
        return ["yt-dlp", "--yes-playlist", "--skip-download", "--dump-json", channel_url]


def yt_dlp_stream_list(channel_url: str, fast: bool = True, timeout: int = 900) -> Iterator[dict]:
    """
    Stream JSON lines from yt-dlp for the given channel. Yields parsed JSON objects as they arrive.
    Uses subprocess.Popen so the caller receives items progressively.
    """
    cmd = _yt_dlp_cmd_for_channel(channel_url, fast=fast)
    logger.info("Launching yt-dlp (%s mode): %s", "fast" if fast else "full", " ".join(cmd))
    # Use Popen to stream stdout, decode lines as they arrive
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    # Stream stdout lines (each should be a JSON object for dump-json)
    assert proc.stdout is not None
    try:
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                yield obj
            except json.JSONDecodeError:
                logger.debug("Non-JSON line from yt-dlp: %s", line[:400])
                continue
    finally:
        # Ensure process termination if it's still running
        try:
            proc.stdout.close()
        except Exception:
            pass
        # read and log stderr (non-blocking attempt)
        try:
            stderr = proc.stderr.read() if proc.stderr else ""
            if stderr:
                logger.debug("yt-dlp stderr (tail): %s", stderr[-2000:])
        except Exception:
            pass
        # Wait for yt-dlp to exit
        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def build_watch_url_from_entry(entry: dict) -> Optional[str]:
    if entry.get("webpage_url"):
        return entry.get("webpage_url")
    vid = entry.get("id") or entry.get("url")
    if not vid:
        return None
    if isinstance(vid, str) and vid.startswith("http"):
        return vid
    return f"https://www.youtube.com/watch?v={vid}"


def filter_entry_by_date(entry: dict, after: Optional[date], before: Optional[date]) -> bool:
    """
    Return True if entry passes date filter (or if date not available).
    entry may contain upload_date (YYYYMMDD) or timestamp.
    """
    if not (after or before):
        return True

    upload_date = entry.get("upload_date") or entry.get("timestamp") or entry.get("upload_date_iso") or entry.get("upload_date_utc")
    if not upload_date:
        # No date info -> conservative choice: include (alternatively exclude; choose include)
        return True

    try:
        if isinstance(upload_date, int):
            dt = datetime.utcfromtimestamp(upload_date).date()
        else:
            s = str(upload_date)
            if len(s) == 8 and s.isdigit():
                dt = datetime.strptime(s, "%Y%m%d").date()
            else:
                # try iso parse
                dt = datetime.fromisoformat(s).date()
    except Exception:
        logger.debug("Could not parse upload date '%s' for id=%s", upload_date, entry.get("id"))
        return True

    if after and dt < after:
        return False
    if before and dt > before:
        return False
    return True


def populate_urls(
    output_file: str = "urls.txt",
    channel_url: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    force_full: bool = False,
    progress_every: int = 50,
    timeout: int = 900
) -> int:
    """
    Write YouTube video watch URLs to output_file. Streams results from yt-dlp and logs progress.
    - If date_from/date_to provided, uses full metadata mode (force_full=True will also force full mode).
    - If no date filters and force_full False, uses fast flat-playlist mode.
    Returns number of URLs written.
    """
    if channel_url is None:
        channel_url = os.getenv("CHANNEL_URL")
    if not channel_url:
        raise ValueError("CHANNEL_URL must be provided either as argument or in .env")

    after = parse_date(date_from or os.getenv("DATE_FROM"))
    before = parse_date(date_to or os.getenv("DATE_TO"))
    use_full = force_full or (after is not None or before is not None)

    logger.info("Populating URLs for channel=%s (after=%s before=%s) use_full=%s", channel_url, after, before, use_full)

    # Stream entries and write incrementally
    urls_written = 0
    seen: Set[str] = set()

    # ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_file)) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Open file for writing (overwrite) and flush as we go so you can watch file grow
    with open(output_file, "w", encoding="utf-8") as fh:
        for i, entry in enumerate(yt_dlp_stream_list(channel_url, fast=not use_full, timeout=timeout), start=1):
            try:
                # apply date filter if needed
                if not filter_entry_by_date(entry, after, before):
                    if i % progress_every == 0:
                        logger.debug("Skipping id=%s due to date filter", entry.get("id"))
                    continue

                url = build_watch_url_from_entry(entry)
                if not url:
                    logger.debug("Skipping entry with no url/id: %s", entry.get("title"))
                    continue

                if url in seen:
                    continue
                seen.add(url)

                fh.write(url + "\n")
                fh.flush()
                urls_written += 1

                # Progress logging
                if urls_written % progress_every == 0:
                    logger.info("Wrote %d URLs so far (processed %d items). Latest: %s", urls_written, i, url)

                # small heartbeat every 200 processed items even if not writing
                if i % 200 == 0:
                    logger.info("Processed %d items from yt-dlp; written %d URLs", i, urls_written)
            except KeyboardInterrupt:
                logger.warning("Interrupted by user; stopping early. Written %d urls", urls_written)
                break
            except Exception as e:
                logger.exception("Error processing entry: %s", e)
                # continue processing other entries

    logger.info("Finished. Total URLs written: %d", urls_written)
    return urls_written


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Populate urls.txt from a YouTube channel using yt-dlp (reads .env by default)")
    parser.add_argument("--output", default=os.getenv("OUTPUT_FILE", "urls.txt"), help="Output file to write URLs")
    parser.add_argument("--channel", help="Channel or playlist URL (overrides CHANNEL_URL in .env)")
    parser.add_argument("--from", dest="date_from", help="Start date (inclusive), format YYYY or YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", help="End date (inclusive), format YYYY or YYYY-MM-DD")
    parser.add_argument("--full", action="store_true", help="Force full metadata mode (slower) even if no date filters provided")
    parser.add_argument("--progress-every", type=int, default=int(os.getenv("POPULATE_PROGRESS_EVERY", "50")), help="Log progress every N writes")
    parser.add_argument("--timeout", type=int, default=int(os.getenv("POPULATE_TIMEOUT", "900")), help="Max seconds to wait for yt-dlp to run")
    args = parser.parse_args()

    populate_urls(output_file=args.output, channel_url=args.channel, date_from=args.date_from, date_to=args.date_to, force_full=args.full, progress_every=args.progress_every, timeout=args.timeout)
