import os
from typing import Tuple, Optional, Iterable, Iterator # Updated imports
# REMOVED: from automator import IntroSkipperAutomator <-- This line caused the crash
from sponsorblock_api import SponsorBlockAPI
from youtube_downloader import YouTubeDownloader
from audio_fingerprint import AudioFingerprinter
from speech_detector import SpeechDetector
from models import IntroSegment
from logger import logger

class IntroSkipperAutomator:
    """Main automation class"""

    def __init__(self, manual_approval: bool = False):
        self.fingerprinter = AudioFingerprinter()
        self.speech_detector = SpeechDetector()
        self.downloader = YouTubeDownloader()
        self.sponsorblock = SponsorBlockAPI()
        self.manual_approval = manual_approval

        self.reference_data = None
        self.intro_duration = 0.0

    def set_reference_intro(self, audio_path: str, duration: float = None):
        logger.info(f"Analyzing reference intro: {audio_path}")
        self.reference_data = self.fingerprinter.generate_fingerprint(audio_path, duration=duration)
        
        if self.reference_data:
            self.intro_duration = self.reference_data["duration"]
            logger.info(f"Reference intro set. Duration: {self.intro_duration:.2f}s")
        else:
            raise ValueError("Failed to analyze reference intro audio")

    def find_intro_in_video(self, audio_path: str) -> Optional[IntroSegment]:
        if not self.reference_data:
            raise ValueError("No reference intro set")

        # Search first 240 seconds (4 minutes)
        match_result = self.fingerprinter.find_match(self.reference_data, audio_path, search_limit_seconds=240.0)

        if match_result:
            start_time, end_time, score = match_result
            logger.info(f"Match candidate found: {start_time:.2f}s - {end_time:.2f}s (Score: {score:.3f})")

            if score > 0.45:
                return IntroSegment(start_time=start_time, end_time=end_time, confidence=score)
        
        logger.warning("No intro match found above threshold.")
        return None

    def process_video(self, url: str, skip_if_exists: bool = True) -> Tuple[bool, str, str]:
        audio_path = None
        video_id = "unknown"

        try:
            # Extract ID first to check SponsorBlock before downloading
            try:
                video_id = self.downloader.extract_video_id(url)
            except Exception:
                # If extraction fails, we might be dealing with a raw ID or complex URL
                # Let the downloader handle it later, but we can't skip_if_exists easily
                pass

            logger.info(f"Processing: {url} (ID: {video_id})")

            if video_id != "unknown" and skip_if_exists:
                existing_segments = self.sponsorblock.get_segments(video_id)
                if any(s.get('category') == 'intro' for s in existing_segments):
                    logger.info(f"Video {video_id} already has intro segment(s), skipping...")
                    return True, 'skipped', video_id

            try:
                audio_path, video_id = self.downloader.download_audio(url)
            except Exception as e:
                logger.error(f"Download failed: {e}")
                return False, 'failed', video_id

            intro_segment = self.find_intro_in_video(audio_path)

            if intro_segment:
                intro_segment.video_id = video_id
                
                if self.manual_approval:
                    self._print_manual_prompt(url, video_id, intro_segment)
                    if not self._get_user_confirmation():
                        return False, 'skipped', video_id

                success = self.sponsorblock.submit_segment(
                    video_id, intro_segment.start_time, intro_segment.end_time
                )
                return success, 'success' if success else 'failed', video_id
            
            else:
                return False, 'failed', video_id

        except Exception as e:
            logger.error(f"Failed to process video {url}: {e}")
            return False, 'failed', video_id
            
        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

    def _print_manual_prompt(self, url, video_id, segment):
        print(f"\n{'='*60}")
        print(f"Video: {url}")
        print(f"Detected intro: {segment.start_time:.2f}s - {segment.end_time:.2f}s")
        print(f"Confidence: {segment.confidence:.2f}")
        print(f"{'='*60}")

    def _get_user_confirmation(self):
        while True:
            r = input("Submit? (y/n): ").strip().lower()
            if r in ['y', 'yes']: return True
            if r in ['n', 'no']: return False

    def process_from_source(self, url_source: Iterable[str]):
        """
        Accepts any iterable (list, generator, etc) of URLs.
        """
        results = {'success': 0, 'failed': 0, 'skipped': 0}
        
        try:
            for i, url in enumerate(url_source, 1):
                url = url.strip()
                if not url or url.startswith('#'): continue

                logger.info(f"\n--- Item {i} ---")
                success, status, vid = self.process_video(url)
                results[status] += 1
        except KeyboardInterrupt:
            logger.warning("\nProcess interrupted by user.")
        
        logger.info(f"\n=== Session Complete ===")
        logger.info(f"Success: {results['success']}, Failed: {results['failed']}, Skipped: {results['skipped']}")

    def cleanup(self):
        self.fingerprinter.cleanup()