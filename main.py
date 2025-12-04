import os
import argparse
from automator import IntroSkipperAutomator
from sponsorblock_api import SponsorBlockAPI
from logger import logger

# Import the helper functions we wrote in populate_urls.py
from populate_urls import yt_dlp_stream_list, build_watch_url_from_entry

def url_generator_from_file(filepath):
    """Yields URLs from a text file"""
    with open(filepath, 'r') as f:
        for line in f:
            yield line

def url_generator_from_channel(channel_url):
    """Yields URLs dynamically from a YouTube channel"""
    logger.info(f"Fetching video list from channel: {channel_url}")
    # fast=True uses flat-playlist (very fast, no full metadata)
    stream = yt_dlp_stream_list(channel_url, fast=True)
    for entry in stream:
        url = build_watch_url_from_entry(entry)
        if url:
            yield url

def main():
    parser = argparse.ArgumentParser(description="Automated YouTube intro detection and SponsorBlock submission")
    parser.add_argument("--reference-intro", help="Path to reference intro audio file")
    
    # New options for input source
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--urls-file", help="Text file containing YouTube URLs (one per line)")
    group.add_argument("--channel", help="YouTube Channel URL (processes all videos in channel)")
    
    parser.add_argument("--intro-duration", type=float, default=10.4, help="Intro duration in seconds")
    parser.add_argument("--user-id", help="SponsorBlock user ID")
    parser.add_argument("--delete-video", help="Delete intro segments for a specific video ID")
    parser.add_argument("--list-segments", help="List all segments for a video ID")
    parser.add_argument("--manual-approval", action="store_true", help="Require manual approval before submitting")

    args = parser.parse_args()

    # --- Handling List/Delete commands (No change here) ---
    if args.list_segments:
        # ... (Same code as before)
        user_id=args.user_id or os.getenv("SPONSORBLOCK_USER_ID")
        api = SponsorBlockAPI(user_id)
        segments = api.get_segments(args.list_segments)
        # ... (Print logic from previous version)
        return

    if args.delete_video:
        # ... (Same code as before)
        return
    # ------------------------------------------------------

    # Processing Mode
    if not args.reference_intro:
        parser.error("--reference-intro is required for processing mode")

    if not args.urls_file and not args.channel:
        parser.error("You must provide either --urls-file OR --channel")

    automator = IntroSkipperAutomator(manual_approval=args.manual_approval)
    automator.set_reference_intro(args.reference_intro, args.intro_duration)

    if args.user_id:
        automator.sponsorblock.user_id = args.user_id
    
    try:
        if args.channel:
            # Stream directly from channel
            source = url_generator_from_channel(args.channel)
            automator.process_from_source(source)
        else:
            # Read from file
            source = url_generator_from_file(args.urls_file)
            automator.process_from_source(source)
            
    finally:
        automator.cleanup()

if __name__ == '__main__':
    main()