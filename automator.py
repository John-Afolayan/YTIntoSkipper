import os
import numpy as np
from scipy.signal import find_peaks
from typing import Tuple, Optional, Iterable

from sponsorblock_api import SponsorBlockAPI
from youtube_downloader import YouTubeDownloader
from audio_fingerprint import AudioFingerprinter
from models import IntroSegment
from logger import logger

class IntroSkipperAutomator:
    # CONSTRAINT: Never submit an intro smaller than this
    MIN_INTRO_DURATION = 2.0 

    def __init__(self, manual_approval: bool = False):
        self.fingerprinter = AudioFingerprinter()
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

        # 1. Scan Video
        scores, sr, hop_length = self.fingerprinter.scan_video(self.reference_data, audio_path)
        if scores is None or len(scores) == 0: return None

        # 2. Peaks
        peaks, _ = find_peaks(scores, height=0.25, distance=sr*2)
        if len(peaks) == 0: return None

        time_per_frame = hop_length / sr
        candidates = []
        
        # 3. Weighted Selection (Kept from previous working version)
        for p in peaks:
            time_sec = p * time_per_frame
            raw_score = scores[p]
            if time_sec < 1.5: time_sec = 0.0
            
            weighted_score = raw_score
            if time_sec < 20.0: weighted_score += 0.35 
            elif time_sec < 45.0: weighted_score += 0.1
            
            candidates.append({"time": time_sec, "raw": raw_score, "weighted": weighted_score})
        
        candidates.sort(key=lambda x: x["weighted"], reverse=True)
        best = candidates[0]
        
        logger.info(f"Top Candidate: {best['time']:.2f}s (Raw: {best['raw']:.3f}, Weighted: {best['weighted']:.3f})")

        if best["weighted"] < 0.60:
            logger.info("Best candidate failed weighted threshold.")
            return None

        start_time = best["time"]
        raw_confidence = best["raw"]
        end_time = start_time + self.intro_duration

        # 4. Adaptive Divergence Check
        actual_end_timestamp = self.fingerprinter.detect_audio_divergence(
            self.reference_data, 
            audio_path, 
            start_time
        )

        if actual_end_timestamp:
            proposed_duration = actual_end_timestamp - start_time
            
            # === SAFETY CONSTRAINT ===
            if proposed_duration < self.MIN_INTRO_DURATION:
                logger.warning(f"Divergence detected at {actual_end_timestamp:.2f}s, but duration ({proposed_duration:.2f}s) is too short.")
                logger.info(f"Ignoring divergence. Keeping original end time: {end_time:.2f}s")
            else:
                # Valid trim
                new_end = max(start_time, actual_end_timestamp - 0.1)
                logger.info(f"Intro trimmed (Divergence at {actual_end_timestamp:.2f}s): {end_time:.2f}s -> {new_end:.2f}s")
                end_time = new_end

        return IntroSegment(start_time=start_time, end_time=end_time, confidence=raw_confidence)

    def process_video(self, url: str, skip_if_exists: bool = True) -> Tuple[bool, str, str]:
        # (Standard processing logic...)
        audio_path = None
        video_id = "unknown"
        try:
            try:
                video_id = self.downloader.extract_video_id(url)
            except: pass

            logger.info(f"Processing: {url}")
            if video_id != "unknown" and skip_if_exists:
                existing = self.sponsorblock.get_segments(video_id)
                if any(s.get('category') == 'intro' for s in existing):
                    logger.info("Skipping (already has intro)")
                    return True, 'skipped', video_id

            audio_path, video_id = self.downloader.download_audio(url)
            intro_segment = self.find_intro_in_video(audio_path)

            if intro_segment:
                intro_segment.video_id = video_id
                if self.manual_approval:
                    self._print_manual_prompt(url, video_id, intro_segment)
                    if not self._get_user_confirmation():
                        return False, 'skipped', video_id
                
                success = self.sponsorblock.submit_segment(video_id, intro_segment.start_time, intro_segment.end_time)
                return success, 'success', video_id
            else:
                logger.info("No intro found.")
                return False, 'failed', video_id

        except Exception as e:
            logger.error(f"Process error: {e}")
            return False, 'failed', video_id
        finally:
            if audio_path and os.path.exists(audio_path):
                try: os.remove(audio_path)
                except: pass

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
        except KeyboardInterrupt: pass
        logger.info(f"Done. S:{results['success']} F:{results['failed']}")

    def cleanup(self):
        self.fingerprinter.cleanup()