import os
import numpy as np
import librosa
from scipy import signal
from logger import logger

class AudioFingerprinter:
    """
    Advanced Audio Matching using Chroma Feature Cross-Correlation.
    Replaces brittle hashing (fpcalc) with signal processing.
    """

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def generate_fingerprint(self, audio_path: str, duration: float = None, start_time: float = 0.0):
        """
        Loads audio and computes Chroma features.
        Returns a dictionary containing the features and metadata.
        """
        try:
            # Load audio (mono, downsampled for speed)
            y, sr = librosa.load(
                audio_path, 
                sr=self.sample_rate, 
                offset=start_time, 
                duration=duration,
                mono=True
            )
            
            # Compute Chroma CQT (Constant-Q Transform)
            # This represents audio as 12 musical pitches over time.
            # It is robust to noise and voice-overs.
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            
            return {
                "type": "chroma_cqt",
                "data": chroma,
                "duration": librosa.get_duration(y=y, sr=sr),
                "sr": sr
            }
        except Exception as e:
            logger.error(f"Failed to generate features for {audio_path}: {e}")
            return None

    def find_match(self, needle_data: dict, haystack_path: str, search_limit_seconds: float = 180.0):
        """
        Finds the 'needle' (intro) inside the 'haystack' (video) using Cross-Correlation.
        
        Args:
            needle_data: The result of generate_fingerprint for the intro.
            haystack_path: Path to the full video audio.
            search_limit_seconds: Only search the first N seconds to save RAM/Time.
        
        Returns:
            (start_time, end_time, confidence_score) or None
        """
        if not needle_data or needle_data["type"] != "chroma_cqt":
            raise ValueError("Invalid reference fingerprint")

        logger.info(f"Loading video audio (first {search_limit_seconds}s) for analysis...")
        
        try:
            # Load target audio (Limit duration to save RAM)
            y_haystack, sr = librosa.load(
                haystack_path, 
                sr=self.sample_rate, 
                duration=search_limit_seconds,
                mono=True
            )
            
            # Compute features for the video
            chroma_haystack = librosa.feature.chroma_cqt(y=y_haystack, sr=sr)
            
            logger.info("Computing signal correlation...")

            # We correlate the chroma matrices.
            # To do this efficiently with 2D arrays (12 pitch bins x Time), 
            # we flatten the 12 bins or sum correlations per bin.
            # A robust heuristic is to sum the correlations of each pitch class.
            
            ref_len = needle_data["data"].shape[1]
            stack_len = chroma_haystack.shape[1]
            
            if stack_len < ref_len:
                logger.warning("Video is shorter than the reference intro.")
                return None

            # Normalized Cross-Correlation
            # We iterate over the 12 chroma bins (C, C#, D...)
            total_correlation = np.zeros(stack_len - ref_len + 1)
            
            for i in range(12):
                # Correlate row i of haystack with row i of needle
                # mode='valid' returns only where they fully overlap
                corr = signal.correlate(chroma_haystack[i], needle_data["data"][i], mode='valid')
                
                # Normalize (optional, but helps with volume differences)
                # Simple normalization by vector magnitudes helps robustness
                norm_factor = np.linalg.norm(needle_data["data"][i]) * np.linalg.norm(chroma_haystack[i])
                if norm_factor > 0:
                    corr = corr / norm_factor
                    
                total_correlation += corr

            # Find peak
            total_correlation /= 12.0 # Average score across 12 pitch bins
            peak_idx = np.argmax(total_correlation)
            peak_score = total_correlation[peak_idx]
            
            # Scale score roughly to 0-1 range (heuristic)
            # Pure signal correlation can be low even for matches, so we scale it up
            # based on empirical testing.
            confidence = float(peak_score) * 10.0 
            confidence = min(confidence, 1.0)

            # Convert frame index back to time
            # Librosa hop_length defaults to 512
            hop_length = 512
            time_per_frame = hop_length / sr
            
            start_time = peak_idx * time_per_frame
            end_time = start_time + needle_data["duration"]
            
            return start_time, end_time, confidence

        except Exception as e:
            logger.error(f"Error during audio matching: {e}")
            return None

    def cleanup(self):
        # Librosa doesn't create temp files, so nothing to do here.
        pass