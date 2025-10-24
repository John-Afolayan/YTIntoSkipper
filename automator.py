# automator.py
from pathlib import Path
import numpy as np
import os
import shutil
from typing import Optional, Tuple

from models import IntroSegment
from audio_fingerprint import AudioFingerprinter
from speech_detector import SpeechDetector
from youtube_downloader import YouTubeDownloader
from sponsorblock_api import SponsorBlockAPI
from logger import logger

class IntroSkipperAutomator:
    """Main automation class"""

    def __init__(self, reference_intro_path: str = None, manual_approval: bool = False):
        self.fingerprinter = AudioFingerprinter()
        self.speech_detector = SpeechDetector()
        self.downloader = YouTubeDownloader()
        self.sponsorblock = SponsorBlockAPI()
        self.manual_approval = manual_approval

        self.reference_fingerprint = None
        self.intro_duration = 10.4

        if reference_intro_path:
            self.set_reference_intro(reference_intro_path)

    def set_reference_intro(self, audio_path: str, duration: float = 10.4):
        logger.info(f"Setting reference intro from: {audio_path}")
        self.reference_fingerprint = self.fingerprinter.generate_fingerprint(audio_path, duration=duration)
        self.intro_duration = duration

        if not self.reference_fingerprint:
            raise ValueError("Failed to generate reference fingerprint")

    def find_intro_in_video(self, audio_path: str, search_window: float = 60.0) -> Optional[IntroSegment]:
        if not self.reference_fingerprint:
            raise ValueError("No reference intro set")

        logger.info("Searching for intro in video...")
        best_match = None
        best_score = 0.0

        window_step = 1.0
        search_positions = list(np.arange(0, search_window, window_step))
        if 0.0 not in search_positions:
            search_positions.insert(0, 0.0)

        for i, start_time in enumerate(search_positions):
            if i % 10 == 0:
                logger.info(f"Searching... {i}/{len(search_positions)} positions checked (best score so far: {best_score:.2f})")

            fingerprint = self.fingerprinter.generate_fingerprint(
                audio_path,
                start_time=start_time,
                duration=self.intro_duration + 2
            )

            if fingerprint:
                score = self.fingerprinter.compare_fingerprints(self.reference_fingerprint, fingerprint)

                if score > best_score:
                    best_score = score
                    best_match = start_time
                    logger.info(f"New best match at {start_time:.1f}s with score: {score:.3f}")

                if score > 0.90:
                    logger.info(f"High confidence match found at {start_time:.1f}s (score: {score:.2f})")
                    break

        if best_match is None:
            logger.warning("No match positions produced any fingerprinting results")
            return None

        logger.info(f"Search complete. Best match at {best_match:.1f}s with score {best_score:.3f}")

        if best_score > 0.5:
            end_time = best_match + self.intro_duration
            logger.info(f"Using intro duration: {best_match:.1f}s to {end_time:.1f}s (duration: {end_time - best_match:.1f}s)")

            return IntroSegment(
                start_time=best_match,
                end_time=end_time,
                confidence=best_score
            )

        logger.warning(f"No intro found (best score: {best_score:.2f} is below threshold of 0.5)")
        return None

    def process_video(self, url: str, skip_if_exists: bool = True) -> Tuple[bool, str, str]:
        try:
            video_id = self.downloader.extract_video_id(url)
            logger.info(f"Processing video ID: {video_id}")

            if skip_if_exists:
                logger.info(f"Checking SponsorBlock for existing intro segments...")
                existing_segments = self.sponsorblock.get_segments(video_id)

                has_intro = False
                if existing_segments:
                    for seg in existing_segments:
                        category = seg.get('category', '')
                        if category == 'intro':
                            segment_time = seg.get('segment', [0, 0])
                            logger.info(f"Found existing intro segment: {segment_time[0]:.1f}s - {segment_time[1]:.1f}s")
                            has_intro = True
                            break

                if has_intro:
                    logger.info(f"Video {video_id} already has intro segment(s), skipping entirely...")
                    return True, 'skipped', video_id
                else:
                    logger.info(f"No intro segments found for {video_id}, proceeding with processing...")

            audio_path, video_id = self.downloader.download_audio(url)

            intro_segment = self.find_intro_in_video(audio_path)

            if intro_segment:
                intro_segment.video_id = video_id
                logger.info(f"Found intro: {intro_segment.start_time:.1f}s - {intro_segment.end_time:.1f}s (confidence: {intro_segment.confidence:.2f})")

                if self.manual_approval:
                    print(f"\n{'='*60}")
                    print(f"Video: {url}")
                    print(f"Video ID: {video_id}")
                    print(f"Detected intro segment:")
                    print(f"  Start: {intro_segment.start_time:.2f}s")
                    print(f"  End: {intro_segment.end_time:.2f}s")
                    print(f"  Duration: {intro_segment.end_time - intro_segment.start_time:.2f}s")
                    print(f"  Confidence: {intro_segment.confidence:.2f}")
                    print(f"{'='*60}")

                    if intro_segment.confidence < 0.8:
                        logger.warning(f"Low confidence for intro for {url}, aborting")
                        return False, 'skipped', video_id

                    while True:
                        response = input("Submit this segment to SponsorBlock? (y/n or yes/no): ").strip().lower()
                        if response in ['y', 'yes']:
                            print("Submitting segment...")
                            break
                        elif response in ['n', 'no']:
                            print("Segment submission cancelled.")
                            return False, 'skipped', video_id
                        else:
                            print("Invalid response. Please enter 'y', 'n', 'yes', or 'no'.")

                success = self.sponsorblock.submit_segment(
                    video_id,
                    intro_segment.start_time,
                    intro_segment.end_time
                )

                return success, 'success' if success else 'failed', video_id
            else:
                logger.warning(f"No intro found in video {video_id}")
                return False, 'failed', video_id

        except Exception as e:
            logger.error(f"Failed to process video {url}: {e}")
            return False, 'failed'

    def process_url_list(self, urls_file: str):
        with open(urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        logger.info(f"Processing {len(urls)} videos...")

        results = {'success': 0, 'failed': 0, 'skipped': 0}

        failed_file = Path("failed.txt")
        # Load existing failed IDs to avoid duplicates
        existing_failed = set()
        if failed_file.exists():
            try:
                with failed_file.open("r", encoding="utf-8") as fh:
                    existing_failed = {line.strip() for line in fh if line.strip()}
            except Exception as e:
                logger.warning("Could not read existing failed file %s: %s", failed_file, e)
                existing_failed = set()

        new_failed = []  # collect new failed ids (or URLs) to append later

        for i, url in enumerate(urls, 1):
            logger.info(f"\n[{i}/{len(urls)}] Processing: {url}")
            success, status, video_id = self.process_video(url)
            results[status] += 1
            if not success and url not in existing_failed:
                new_failed.append(url)
                existing_failed.add(url)  # keep in-memory set up-to-date

        logger.info(f"\n=== Processing Complete ===")
        logger.info(f"Success: {results['success']}")
        logger.info(f"Skipped: {results['skipped']}")
        logger.info(f"Failed: {results['failed']}")
        logger.info(f"Total: {len(urls)}")

    def cleanup(self):
        self.fingerprinter.cleanup()
        if hasattr(self.downloader, 'output_dir'):
            if self.downloader.output_dir.startswith('/tmp/'):
                shutil.rmtree(self.downloader.output_dir, ignore_errors=True)
                logger.info("Cleaned up temporary files")
                