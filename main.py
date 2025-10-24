# main.py
import os
import argparse
# import logging

from automator import IntroSkipperAutomator
from sponsorblock_api import SponsorBlockAPI

from logger import logger


def main():
    parser = argparse.ArgumentParser(description="Automated YouTube intro detection and SponsorBlock submission")
    parser.add_argument("--reference-intro", help="Path to reference intro audio file")
    parser.add_argument("--urls-file", help="Text file containing YouTube URLs (one per line)")
    parser.add_argument("--intro-duration", type=float, default=10.4, help="Intro duration in seconds")
    parser.add_argument("--user-id", help="SponsorBlock user ID (generates random if not provided)")
    parser.add_argument("--delete-video", help="Delete intro segments for a specific video ID")
    parser.add_argument("--list-segments", help="List all segments for a video ID")
    parser.add_argument("--manual-approval", action="store_true",
                        help="Require manual approval before submitting each segment")

    args = parser.parse_args()

    if args.list_segments:
        user_id=args.user_id or os.getenv("SPONSORBLOCK_USER_ID")
        api = SponsorBlockAPI(user_id)
        segments = api.get_segments(args.list_segments)

        if not segments:
            print(f"No segments found for video: {args.list_segments}")
        else:
            print(f"\nSegments for video {args.list_segments}:")
            for seg in segments:
                segment_data = seg.get('segment', [])
                start = segment_data[0] if len(segment_data) > 0 else 'N/A'
                end = segment_data[1] if len(segment_data) > 1 else 'N/A'
                print(f"  Category: {seg.get('category', 'N/A')}")
                print(f"  Time: {start:.1f}s - {end:.1f}s")
                print(f"  UUID: {seg.get('UUID', 'N/A')}")
                print(f"  Votes: {seg.get('votes', 'N/A')}")
                print()
        return

    if args.delete_video:
        if not args.user_id:
            print("ERROR: --user-id is required to delete segments")
            return

        api = SponsorBlockAPI(user_id=args.user_id)
        segments = api.get_segments(args.delete_video)

        intro_segments = [s for s in segments if s.get('category') == 'intro']

        if not intro_segments:
            print(f"No intro segments found for video: {args.delete_video}")
            return

        print(f"\nFound {len(intro_segments)} intro segment(s) for video {args.delete_video}:")
        for i, seg in enumerate(intro_segments, 1):
            segment_data = seg.get('segment', [])
            start = segment_data[0] if len(segment_data) > 0 else 'N/A'
            end = segment_data[1] if len(segment_data) > 1 else 'N/A'
            print(f"{i}. Time: {start:.1f}s - {end:.1f}s (UUID: {seg.get('UUID', 'N/A')})")

        confirm = input("\nDelete these segments? (yes/no): ")
        if confirm.lower() == 'yes':
            for seg in intro_segments:
                uuid = seg.get('UUID')
                if uuid:
                    api.delete_segment(uuid)
        else:
            print("Deletion cancelled.")
        return

    if not args.reference_intro or not args.urls_file:
        parser.error("--reference-intro and --urls-file are required for processing mode")

    automator = IntroSkipperAutomator(manual_approval=args.manual_approval)
    automator.set_reference_intro(args.reference_intro, args.intro_duration)

    if args.user_id:
        automator.sponsorblock.user_id = args.user_id
        logger.info(f"Using user ID: {args.user_id}")
    else:
        logger.info(f"Generated user ID: {automator.sponsorblock.user_id}")
        logger.info("Save this ID if you want to manage your submissions later!")

    try:
        automator.process_url_list(args.urls_file)
    finally:
        automator.cleanup()


if __name__ == '__main__':
    main()
