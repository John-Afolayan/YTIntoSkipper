import numpy as np
import librosa
from scipy import signal
from logger import logger

class AudioFingerprinter:
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def generate_fingerprint(self, audio_path: str, duration: float = None, start_time: float = 0.0):
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, offset=start_time, duration=duration, mono=True)
            # Use CQT for fingerprinting (Musically accurate)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            
            # Normalize for matching
            norm_factor = np.linalg.norm(chroma) + 1e-9
            chroma_norm = chroma / norm_factor

            return {
                "type": "chroma_cqt_linear",
                "data": chroma_norm,      # For finding the intro location
                "raw_chroma": chroma,     # For divergence checking (un-normalized globally)
                "duration": librosa.get_duration(y=y, sr=sr),
                "sr": sr
            }
        except Exception as e:
            logger.error(f"Fingerprint error {audio_path}: {e}")
            return None

    def scan_video(self, needle_data: dict, haystack_path: str, search_limit_seconds: float = 300.0):
        # Keeps the robust Linear logic
        if not needle_data: return None
        try:
            y_haystack, sr = librosa.load(haystack_path, sr=self.sample_rate, duration=search_limit_seconds, mono=True)
            chroma_haystack = librosa.feature.chroma_cqt(y=y_haystack, sr=sr)
            
            needle = needle_data["data"]
            haystack = chroma_haystack

            if haystack.shape[1] < needle.shape[1]: return None, None, None

            # Standard Normalized Cross Correlation approximation
            numerator = np.zeros(haystack.shape[1] - needle.shape[1] + 1)
            for i in range(needle.shape[0]):
                numerator += signal.correlate(haystack[i], needle[i], mode='valid')

            haystack_sq = haystack ** 2
            window_sum_sq = np.zeros(haystack.shape[1] - needle.shape[1] + 1)
            ones_kernel = np.ones(needle.shape[1])
            
            for i in range(needle.shape[0]):
                window_sum_sq += signal.correlate(haystack_sq[i], ones_kernel, mode='valid')
            
            haystack_window_norms = np.sqrt(window_sum_sq)
            epsilon = 1e-5
            scores = numerator / (haystack_window_norms + epsilon)

            return scores, sr, 512

        except Exception as e:
            logger.error(f"Scan error: {e}")
            return None, None, None

    def detect_audio_divergence(self, needle_data: dict, video_path: str, match_start_time: float):
        """
        Adaptive Cosine Similarity Check.
        Learns the quality of the video before judging if the intro ended.
        """
        try:
            duration = needle_data["duration"]
            ref_chroma = needle_data["raw_chroma"] # Use raw CQT, not globally normalized
            sr = needle_data["sr"]

            # Load the video audio at the match point
            y_vid, _ = librosa.load(video_path, sr=sr, offset=match_start_time, duration=duration, mono=True)
            
            # Compute CQT of video
            vid_chroma = librosa.feature.chroma_cqt(y=y_vid, sr=sr)

            # Fix Lengths
            min_cols = min(ref_chroma.shape[1], vid_chroma.shape[1])
            ref = ref_chroma[:, :min_cols]
            vid = vid_chroma[:, :min_cols]

            # 1. Normalize Columns (Frame-by-Frame)
            # This makes the comparison volume-independent for every single frame
            ref_norm = np.linalg.norm(ref, axis=0) + 1e-9
            vid_norm = np.linalg.norm(vid, axis=0) + 1e-9
            
            ref = ref / ref_norm
            vid = vid / vid_norm

            # 2. Compute Cosine Similarity (Dot Product of normalized vectors)
            # Result is 0.0 (Different) to 1.0 (Identical) per frame
            similarity = np.sum(ref * vid, axis=0)

            # 3. Establish Baseline (The "Calibration")
            # We look at the first 1.5 seconds (approx 65 frames)
            # We assume the start of the match is correct.
            hop_length = 512
            calibration_frames = int(1.5 * sr / hop_length)
            calibration_frames = min(calibration_frames, len(similarity))

            if calibration_frames < 10: return None 

            baseline_sim = np.mean(similarity[:calibration_frames])
            
            # 4. Define Thresholds
            # If the video is high quality, baseline might be 0.95. Cut if < 0.8.
            # If the video is compressed, baseline might be 0.7. Cut if < 0.55.
            # We allow a drop of 0.25 from the baseline.
            drop_tolerance = 0.25
            cut_threshold = baseline_sim - drop_tolerance

            # Frames required to confirm a cut (0.4 seconds)
            frames_required = int(0.4 * sr / hop_length)
            bad_frame_counter = 0

            # 5. Scan for Divergence
            # Start slightly after calibration to prevent jitter
            start_scan_idx = int(2.0 * sr / hop_length) 

            for i in range(start_scan_idx, len(similarity)):
                score = similarity[i]

                if score < cut_threshold:
                    bad_frame_counter += 1
                else:
                    # If we see a good frame, reset (unless it's just a momentary glitch in a cut)
                    # We use a slight decay to be sticky
                    bad_frame_counter = max(0, bad_frame_counter - 1)

                if bad_frame_counter > frames_required:
                    # Divergence confirmed. 
                    # Return time of the START of this bad sequence
                    divergence_idx = i - frames_required
                    time_offset = librosa.frames_to_time(divergence_idx, sr=sr, hop_length=hop_length)
                    return match_start_time + time_offset

            return None

        except Exception as e:
            logger.error(f"Divergence check failed: {e}")
            return None

    def cleanup(self):
        pass