import os
import numpy as np
from scipy.signal import find_peaks
from typing import Tuple, Optional, Iterable

from sponsorblock_api import SponsorBlockAPI
from youtube_downloader import YouTubeDownloader
from audio_fingerprint import AudioFingerprinter
from speech_detector import SpeechDetector
from models import IntroSegment
from logger import logger

class IntroSkipperAutomator:
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
            logger.info(f"Reference loaded. Duration: {self.intro_duration:.2f}s")
        else:
            raise ValueError("Failed to analyze reference intro")

    def find_intro_in_video(self, audio_path: str) -> Optional[IntroSegment]:
        if not self.reference_data: raise ValueError("No reference intro set")

        # 1. Get scores
        scores, sr, hop_length = self.fingerprinter.scan_video(self.reference_data, audio_path)
        if scores is None or len(scores) == 0: return None

        # 2. Find Peaks
        # Threshold lowered to 0.3. Linear matching is tougher, so decent matches might sit at 0.4-0.5.
        # We rely on the "Early Bias" logic to filter out noise.
        peaks, properties = find_peaks(scores, height=0.3, distance=sr*2)
        
        if len(peaks) == 0:
            logger.info("No intro detected (No correlation peaks above 0.3)")
            return None

        time_per_frame = hop_length / sr
        candidates = []
        for p in peaks:
            time_sec = p * time_per_frame
            score = scores[p]
            candidates.append((time_sec, score))
        
        candidates.sort(key=lambda x: x[0])

        selected_candidate = None
        
        # 3. Selection Logic (Refined)
        for time_sec, score in candidates:
            logger.debug(f"Candidate: {time_sec:.2f}s, Score: {score:.3f}")
            
            # Snap-to-Start Check
            # If a candidate is within the first 2 seconds, treat it as 0.0s
            if time_sec < 2.0:
                logger.debug(f"Snapping candidate at {time_sec:.2f}s to 0.0s")
                time_sec = 0.0

            # Priority Zone: 0s - 45s
            if time_sec < 45.0:
                # Acceptance Threshold: 0.5
                if score > 0.5:
                    logger.info(f"Accepted Priority Match at {time_sec:.2f}s (Score: {score:.3f})")
                    selected_candidate = (time_sec, score)
                    break
            else:
                # Late Zone: Only accept if very strong (0.8+) AND we haven't found an early one
                if score > 0.8:
                    if selected_candidate is None:
                        selected_candidate = (time_sec, score)
                    elif score > selected_candidate[1] + 0.25:
                         # Very strict override
                         selected_candidate = (time_sec, score)

        if not selected_candidate:
            return None

        start_time, confidence = selected_candidate
        end_time = start_time + self.intro_duration
        
        logger.info(f"Base Segment: {start_time:.2f}s - {end_time:.2f}s (Conf: {confidence:.3f})")

        # 4. Trimming Logic
        # We only check for speech in the LAST 25% of the intro.
        # Checking too early (e.g. 50%) might trigger on the YouTuber saying "Hey guys" BEFORE the intro ends.
        check_ratio = 0.25
        check_duration = self.intro_duration * check_ratio
        check_start = max(start_time, end_time - check_duration)
        
        speech_start = self.speech_detector.detect_speech_entry(audio_path, check_start, check_duration)
        
        if speech_start:
            # If speech is detected, pad back 0.1s
            new_end = max(start_time, speech_start - 0.1)
            logger.info(f"Speech detected at {speech_start:.2f}s. Trimming: {end_time:.2f}s -> {new_end:.2f}s")
            end_time = new_end

        return IntroSegment(start_time=start_time, end_time=end_time, confidence=confidence)

    def process_video(self, url: str, skip_if_exists: bool = True) -> Tuple[bool, str, str]:
        # (Same standard process_video implementation as previous steps)
        # Included for completeness of the file structure
        audio_path = None
        video_id = "unknown"

        try:
            try:
                video_id = self.downloader.extract_video_id(url)
            except Exception: pass

            logger.info(f"Processing: {url} (ID: {video_id})")

            if video_id != "unknown" and skip_if_exists:
                existing_segments = self.sponsorblock.get_segments(video_id)
                if any(s.get('category') == 'intro' for s in existing_segments):
                    logger.info(f"Skipping {video_id} (already has intro)")
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
                logger.info(f"No intro found for {video_id}")
                return False, 'failed', video_id

        except Exception as e:
            logger.error(f"Failed to process video {url}: {e}")
            return False, 'failed', video_id
        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except OSError: pass

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