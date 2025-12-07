import numpy as np
import librosa
from scipy import signal
from logger import logger

class AudioFingerprinter:
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def generate_fingerprint(self, audio_path: str, duration: float = None, start_time: float = 0.0):
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate, offset=start_time, duration=duration, mono=True)
            
            # Linear CQT (Constant-Q Transform)
            # We do NOT convert to dB. We want the strong musical peaks to dominate.
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            
            # Normalize the reference itself to unit energy
            # This helps standardize the scores later
            norm_factor = np.linalg.norm(chroma) + 1e-9
            chroma = chroma / norm_factor

            return {
                "type": "chroma_cqt_linear",
                "data": chroma,
                "duration": librosa.get_duration(y=y, sr=sr),
                "sr": sr
            }
        except Exception as e:
            logger.error(f"Fingerprint error {audio_path}: {e}")
            return None

    def scan_video(self, needle_data: dict, haystack_path: str, search_limit_seconds: float = 300.0):
        if not needle_data: return None

        try:
            y_haystack, sr = librosa.load(haystack_path, sr=self.sample_rate, duration=search_limit_seconds, mono=True)
            
            # Compute CQT for haystack
            chroma_haystack = librosa.feature.chroma_cqt(y=y_haystack, sr=sr)
            
            needle = needle_data["data"]
            haystack = chroma_haystack

            n_features, n_frames_needle = needle.shape
            n_features_stack, n_frames_stack = haystack.shape

            if n_frames_stack < n_frames_needle:
                logger.warning("Video is shorter than reference intro")
                return None, None, None

            # === Cross-Correlation ===
            # We use a simple sliding dot product sum.
            # Since we didn't normalize the haystack windows yet, high-volume sections 
            # would naturally score higher. We must normalize the window energy.
            
            # 1. Numerator: Sliding Dot Product
            numerator = np.zeros(n_frames_stack - n_frames_needle + 1)
            for i in range(n_features):
                numerator += signal.correlate(haystack[i], needle[i], mode='valid')

            # 2. Denominator: Sliding Window Norm of Haystack
            # Calculate energy of the haystack windows
            haystack_sq = haystack ** 2
            window_sum_sq = np.zeros(n_frames_stack - n_frames_needle + 1)
            ones_kernel = np.ones(n_frames_needle)
            
            for i in range(n_features):
                window_sum_sq += signal.correlate(haystack_sq[i], ones_kernel, mode='valid')
            
            haystack_window_norms = np.sqrt(window_sum_sq)

            # 3. Final Score
            # Since needle is already unit-norm, we just divide by haystack norm
            # Add small epsilon to prevent silence division errors
            epsilon = 1e-5
            scores = numerator / (haystack_window_norms + epsilon)

            # Heuristic scaling:
            # In linear space, 'perfect' matches usually hover around 0.6-0.8 due to
            # slight mastering differences. We boost this slightly to make 0-1 easier to read.
            # This is optional but helps with the thresholds in the Automator.
            # scores = scores * 1.0 (Keeping raw for now to be safe)

            hop_length = 512
            return scores, sr, hop_length

        except Exception as e:
            logger.error(f"Scan error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None, None

    def cleanup(self):
        pass