# speech_detector.py
import subprocess
# import logging
import numpy as np

# logger = logging.getLogger(__name__)
from logger import logger

class SpeechDetector:
    """Detects speech boundaries using VAD"""

    def __init__(self):
        self.sample_rate = 16000
        self._check_dependencies()

    def _check_dependencies(self):
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(2)
        except ImportError:
            logger.warning("webrtcvad not installed. Speech detection will be limited.")
            self.vad = None

    def find_speech_start(self, audio_path: str, start_time: float, window_size: float = 10.0) -> float:
        if not self.vad:
            return start_time + 2.0

        cmd = [
            "ffmpeg", "-i", audio_path,
            "-ss", str(start_time),
            "-t", str(window_size),
            "-ar", str(self.sample_rate),
            "-ac", "1",
            "-f", "s16le",
            "-"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, check=True, stderr=subprocess.DEVNULL)
            audio_data = np.frombuffer(result.stdout, dtype=np.int16)

            frame_duration_ms = 30
            frame_size = int(self.sample_rate * frame_duration_ms / 1000)

            speech_frames = []
            for i in range(0, len(audio_data) - frame_size, frame_size):
                frame = audio_data[i:i + frame_size].tobytes()
                is_speech = self.vad.is_speech(frame, self.sample_rate)
                speech_frames.append(is_speech)

            for i in range(len(speech_frames) - 3):
                if all(speech_frames[i:i+3]):
                    return start_time + (i * frame_duration_ms / 1000)

            return start_time + 2.0

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to process audio for speech detection: {e}")
            return start_time + 2.0