import subprocess
import numpy as np
import os
from logger import logger

class SpeechDetector:
    """Detects speech boundaries using VAD"""

    def __init__(self):
        self.sample_rate = 16000
        self.vad = None
        self._check_dependencies()

    def _check_dependencies(self):
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(2) # Mode 2: Aggressive
        except ImportError:
            logger.warning("webrtcvad not installed. Speech detection disabled.")

    def detect_speech_entry(self, audio_path: str, start_time: float, duration: float) -> float:
        """
        Scans a specific window and returns the timestamp of the FIRST detected speech.
        Returns None if no speech is detected.
        """
        if not self.vad:
            return None

        # Extract just the relevant audio chunk to memory
        cmd = [
            "ffmpeg", "-y", "-v", "quiet",
            "-i", audio_path,
            "-ss", str(start_time),
            "-t", str(duration),
            "-ar", str(self.sample_rate),
            "-ac", "1",
            "-f", "s16le",
            "-"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, check=True)
            audio_data = np.frombuffer(result.stdout, dtype=np.int16)
            
            frame_duration_ms = 30
            frame_size = int(self.sample_rate * frame_duration_ms / 1000) # 480 samples

            # Sliding window to find speech
            # We require 2 consecutive frames of speech to trigger a "start"
            # to avoid popping noises triggering a cut.
            consecutive_speech = 0
            
            for i in range(0, len(audio_data) - frame_size, frame_size):
                frame = audio_data[i:i + frame_size].tobytes()
                
                if self.vad.is_speech(frame, self.sample_rate):
                    consecutive_speech += 1
                else:
                    consecutive_speech = 0
                
                # If we detect 3 consecutive frames (~90ms) of speech, mark it
                if consecutive_speech >= 3:
                    # Calculate time relative to the window start
                    # We subtract the buffer (3 frames) to get the start of the phrase
                    speech_offset = (i - (frame_size * 2)) / self.sample_rate
                    found_time = start_time + speech_offset
                    return max(start_time, found_time)

            return None

        except Exception as e:
            logger.error(f"VAD error: {e}")
            return None 