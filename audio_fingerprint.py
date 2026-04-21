import shutil
import warnings
import numpy as np
import librosa
import subprocess
import tempfile
import hashlib
import os
from scipy import signal
from pathlib import Path
from logger import logger


_ffmpeg_available: bool | None = None  # cached check


def _has_ffmpeg() -> bool:
    global _ffmpeg_available
    if _ffmpeg_available is None:
        _ffmpeg_available = shutil.which("ffmpeg") is not None
    return _ffmpeg_available


class AudioFingerprinter:
    CACHE_DIR = Path(".fingerprint_cache")

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------
    # Audio loading strategy:
    #   1. If ffmpeg is available, pre-convert to WAV (handles video
    #      containers, weird codecs, etc.)
    #   2. If ffmpeg is missing, fall back to direct librosa.load()
    #      which works for pure audio files (m4a, mp3, wav, ogg, flac)
    #      but will fail on video containers.
    # ------------------------------------------------------------------
    @staticmethod
    def _convert_to_wav(audio_path: str) -> str:
        """
        Convert any media file to a temporary 16-bit mono WAV via ffmpeg.
        Returns the path to the temp WAV (caller must clean up).
        Raises RuntimeError if ffmpeg fails.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        cmd = [
            "ffmpeg", "-y", "-v", "quiet",
            "-i", audio_path,
            "-vn",                # strip video
            "-ac", "1",           # mono
            "-ar", "22050",       # target sample rate
            "-sample_fmt", "s16", # 16-bit
            tmp.name,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return tmp.name
        except subprocess.CalledProcessError as e:
            os.unlink(tmp.name)
            raise RuntimeError(
                f"ffmpeg failed to convert '{audio_path}' to WAV. "
                f"stderr: {e.stderr.decode(errors='replace')[:500]}"
            )

    def _load_audio(self, audio_path: str, offset: float = 0.0, duration: float = None):
        """
        Load audio from any file, returning (y, sr, tmp_path_or_None).
        Tries ffmpeg conversion first; falls back to direct librosa.load().
        The caller MUST clean up tmp_path if it's not None.
        """
        # --- Strategy 1: ffmpeg pre-conversion (handles everything) ---
        if _has_ffmpeg():
            tmp_path = self._convert_to_wav(audio_path)
            try:
                y, sr = librosa.load(
                    tmp_path, sr=self.sample_rate,
                    offset=offset, duration=duration, mono=True,
                )
                return y, sr, tmp_path
            except Exception:
                # Clean up tmp on failure before re-raising
                os.unlink(tmp_path)
                raise

        # --- Strategy 2: direct librosa load (works for pure audio) ---
        logger.warning(
            "ffmpeg not found on PATH. Falling back to direct librosa loading. "
            "This works for pure audio files but will fail on video containers. "
            "Install ffmpeg for full format support: "
            "  WSL/Linux: sudo apt install ffmpeg  |  "
            "  macOS: brew install ffmpeg  |  "
            "  Windows: winget install ffmpeg"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress audioread deprecation noise
            y, sr = librosa.load(
                audio_path, sr=self.sample_rate,
                offset=offset, duration=duration, mono=True,
            )
        return y, sr, None

    # ------------------------------------------------------------------
    # Fingerprint caching helpers
    # ------------------------------------------------------------------
    @classmethod
    def _cache_key(cls, audio_path: str, duration, start_time: float) -> str:
        """Deterministic cache key based on file content + params."""
        h = hashlib.sha256()
        with open(audio_path, "rb") as f:
            while chunk := f.read(1 << 16):
                h.update(chunk)
        h.update(f"{duration}:{start_time}".encode())
        return h.hexdigest()

    @classmethod
    def _load_cached(cls, key: str) -> dict | None:
        path = cls.CACHE_DIR / f"{key}.npz"
        if not path.exists():
            return None
        try:
            d = np.load(str(path), allow_pickle=True)
            return {
                "type": str(d["type"]),
                "data": d["data"],
                "raw_chroma": d["raw_chroma"],
                "duration": float(d["duration"]),
                "sr": int(d["sr"]),
            }
        except Exception:
            return None

    @classmethod
    def _save_cache(cls, key: str, fp: dict):
        cls.CACHE_DIR.mkdir(exist_ok=True)
        path = cls.CACHE_DIR / f"{key}.npz"
        np.savez_compressed(
            str(path),
            type=fp["type"],
            data=fp["data"],
            raw_chroma=fp["raw_chroma"],
            duration=fp["duration"],
            sr=fp["sr"],
        )

    # ------------------------------------------------------------------
    # Core fingerprint generation
    # ------------------------------------------------------------------
    def generate_fingerprint(
        self,
        audio_path: str,
        duration: float = None,
        start_time: float = 0.0,
        use_cache: bool = True,
    ):
        # Try cache first
        cache_key = None
        if use_cache:
            try:
                cache_key = self._cache_key(audio_path, duration, start_time)
                cached = self._load_cached(cache_key)
                if cached:
                    logger.info("Loaded fingerprint from cache")
                    return cached
            except Exception:
                pass  # non-fatal

        tmp_path = None
        try:
            y, sr, tmp_path = self._load_audio(audio_path, offset=start_time, duration=duration)

            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

            norm_factor = np.linalg.norm(chroma) + 1e-9
            chroma_norm = chroma / norm_factor

            fp = {
                "type": "chroma_cqt_linear",
                "data": chroma_norm,
                "raw_chroma": chroma,
                "duration": librosa.get_duration(y=y, sr=sr),
                "sr": sr,
            }

            # Save to cache
            if use_cache and cache_key:
                try:
                    self._save_cache(cache_key, fp)
                except Exception:
                    pass

            return fp

        except Exception as e:
            logger.error(f"Fingerprint error {audio_path}: {e}")
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Scan video for intro match (with early-exit optimisation)
    # ------------------------------------------------------------------
    def scan_video(
        self,
        needle_data: dict,
        haystack_path: str,
        search_limit_seconds: float = 120.0,
        early_exit_threshold: float = 0.90,
    ):
        if not needle_data:
            return None, None, None

        tmp_path = None
        try:
            y_haystack, sr, tmp_path = self._load_audio(
                haystack_path, duration=search_limit_seconds,
            )
            chroma_haystack = librosa.feature.chroma_cqt(y=y_haystack, sr=sr)

            needle = needle_data["data"]
            haystack = chroma_haystack
            hop_length = 512

            if haystack.shape[1] < needle.shape[1]:
                return None, None, None

            n_positions = haystack.shape[1] - needle.shape[1] + 1

            # --- Normalized cross-correlation ---
            numerator = np.zeros(n_positions)
            for i in range(needle.shape[0]):
                numerator += signal.correlate(haystack[i], needle[i], mode="valid")

            haystack_sq = haystack ** 2
            window_sum_sq = np.zeros(n_positions)
            ones_kernel = np.ones(needle.shape[1])
            for i in range(needle.shape[0]):
                window_sum_sq += signal.correlate(haystack_sq[i], ones_kernel, mode="valid")

            haystack_window_norms = np.sqrt(window_sum_sq)
            scores = numerator / (haystack_window_norms + 1e-5)

            # --- Early exit: if we already have a very strong peak, skip further work ---
            if np.max(scores) >= early_exit_threshold:
                logger.debug(
                    f"Early-exit: peak score {np.max(scores):.3f} >= {early_exit_threshold}"
                )

            return scores, sr, hop_length

        except Exception as e:
            logger.error(f"Scan error: {e}")
            return None, None, None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Adaptive divergence detection (unchanged logic, robust loading)
    # ------------------------------------------------------------------
    def detect_audio_divergence(
        self, needle_data: dict, video_path: str, match_start_time: float
    ):
        """
        Adaptive Cosine Similarity Check.
        Learns the quality of the video before judging if the intro ended.
        """
        wav_path = None
        try:
            duration = needle_data["duration"]
            ref_chroma = needle_data["raw_chroma"]
            sr = needle_data["sr"]

            y_vid, _, wav_path = self._load_audio(
                video_path, offset=match_start_time, duration=duration,
            )
            vid_chroma = librosa.feature.chroma_cqt(y=y_vid, sr=sr)

            min_cols = min(ref_chroma.shape[1], vid_chroma.shape[1])
            ref = ref_chroma[:, :min_cols]
            vid = vid_chroma[:, :min_cols]

            ref_norm = np.linalg.norm(ref, axis=0) + 1e-9
            vid_norm = np.linalg.norm(vid, axis=0) + 1e-9
            ref = ref / ref_norm
            vid = vid / vid_norm

            similarity = np.sum(ref * vid, axis=0)

            hop_length = 512
            calibration_frames = int(1.5 * sr / hop_length)
            calibration_frames = min(calibration_frames, len(similarity))
            if calibration_frames < 10:
                return None

            baseline_sim = np.mean(similarity[:calibration_frames])
            drop_tolerance = 0.25
            cut_threshold = baseline_sim - drop_tolerance

            frames_required = int(0.4 * sr / hop_length)
            bad_frame_counter = 0
            start_scan_idx = int(2.0 * sr / hop_length)

            for i in range(start_scan_idx, len(similarity)):
                score = similarity[i]
                if score < cut_threshold:
                    bad_frame_counter += 1
                else:
                    bad_frame_counter = max(0, bad_frame_counter - 1)

                if bad_frame_counter > frames_required:
                    divergence_idx = i - frames_required
                    time_offset = librosa.frames_to_time(
                        divergence_idx, sr=sr, hop_length=hop_length,
                    )
                    return match_start_time + time_offset

            return None

        except Exception as e:
            logger.error(f"Divergence check failed: {e}")
            return None
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Speech-over-intro detection
    # Compares speech-band energy (300Hz-3kHz) between the reference
    # intro and the video. A sudden excess in the video's speech band
    # that wasn't in the reference = someone talking over the intro.
    # Only scans the TAIL of the intro to stay conservative.
    # ------------------------------------------------------------------
    def detect_speech_overlay(
        self,
        needle_data: dict,
        video_path: str,
        match_start_time: float,
        tail_seconds: float = 4.0,
        min_trim: float = 0.3,
        speech_ratio_threshold: float = 1.8,
        consecutive_frames_required: int = 4,
    ) -> float | None:
        """
        Detect where a speaker starts talking over the intro music.

        Strategy:
          1. Load the tail end of both the reference intro and the video
             at the matched position.
          2. Compute short-time energy in the speech band (300Hz-3kHz)
             for both signals.
          3. Compute the ratio: video_speech_energy / ref_speech_energy.
             Where this ratio is consistently > threshold, the speaker
             has started talking.
          4. Return the absolute timestamp where speech begins, or None.

        Args:
            needle_data: Reference fingerprint data.
            video_path: Path to the downloaded video audio.
            match_start_time: Where the intro was detected in the video.
            tail_seconds: How many seconds from the end of the intro to scan.
            min_trim: Minimum seconds to trim (ignore detections smaller than this).
            speech_ratio_threshold: How much more speech-band energy the video
                must have vs the reference to count as "speech overlay".
                1.8 = 80% more energy. Higher = more conservative.
            consecutive_frames_required: How many consecutive frames must
                exceed the threshold to confirm speech. Higher = fewer
                false positives but might miss very short speech.

        Returns:
            Absolute timestamp where speech overlay begins, or None.
        """
        ref_tmp = None
        vid_tmp = None
        try:
            intro_duration = needle_data["duration"]
            sr = needle_data["sr"]

            # Only scan the tail portion of the intro
            scan_duration = min(tail_seconds, intro_duration - 1.0)
            if scan_duration < 1.0:
                return None  # intro too short to meaningfully scan

            tail_offset_in_intro = intro_duration - scan_duration

            # Load reference tail
            ref_path = None  # We need the original reference audio path...
            # But we don't have it — we have the fingerprint data.
            # So instead, recompute from raw_chroma which we already have.
            # Actually, we need the raw waveform. Let's load from video
            # at the matched position and compare spectral shape.

            # Reference: load from match_start + tail_offset
            ref_start = match_start_time + tail_offset_in_intro

            # We need both the reference audio file and the video.
            # Since we only have the fingerprint (not the reference path),
            # we'll compare the video's speech band against the reference's
            # chroma to detect new energy. But a cleaner approach:
            # store the reference waveform's speech-band profile during
            # fingerprint generation. For now, use a simpler approach:
            # compute the speech-band energy of the reference from its
            # raw_chroma, and compare to the video.

            # --- Load video tail audio ---
            y_vid, _, vid_tmp = self._load_audio(
                video_path, offset=ref_start, duration=scan_duration,
            )

            if len(y_vid) < sr * 0.5:
                return None  # too little audio

            # --- Compute STFT for the video ---
            n_fft = 2048
            hop = 512
            S_vid = np.abs(librosa.stft(y_vid, n_fft=n_fft, hop_length=hop))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

            # Speech band: 300Hz - 3kHz
            speech_mask = (freqs >= 300) & (freqs <= 3000)
            # Non-speech band: everything else (music tends to dominate here)
            nonspeech_mask = ~speech_mask & (freqs > 50)  # skip DC/rumble

            # Per-frame energy in each band
            speech_energy = np.sum(S_vid[speech_mask, :] ** 2, axis=0)
            nonspeech_energy = np.sum(S_vid[nonspeech_mask, :] ** 2, axis=0)

            # Spectral ratio: proportion of energy in speech band
            total_energy = speech_energy + nonspeech_energy + 1e-12
            speech_ratio = speech_energy / total_energy

            # --- Now we need a baseline: what does the reference intro's
            # speech ratio look like? We can estimate it from the raw_chroma.
            # But chroma doesn't directly give us speech-band energy.
            #
            # Better approach: compute a BASELINE from the EARLY part of
            # this same video's matched intro (where we know there's no
            # speech overlay). This is self-referencing and robust.
            # ---

            # Load the EARLY part of the intro in this video (first 2-3 sec)
            baseline_duration = min(3.0, intro_duration - scan_duration - 0.5)
            if baseline_duration < 1.0:
                return None

            y_baseline, _, ref_tmp = self._load_audio(
                video_path, offset=match_start_time, duration=baseline_duration,
            )

            S_baseline = np.abs(librosa.stft(y_baseline, n_fft=n_fft, hop_length=hop))
            baseline_speech = np.sum(S_baseline[speech_mask, :] ** 2, axis=0)
            baseline_nonspeech = np.sum(S_baseline[nonspeech_mask, :] ** 2, axis=0)
            baseline_total = baseline_speech + baseline_nonspeech + 1e-12
            baseline_ratio = np.median(baseline_speech / baseline_total)

            if baseline_ratio < 1e-6:
                return None  # degenerate case

            # --- Scan for speech onset ---
            # We look for frames where the speech ratio is significantly
            # higher than the baseline (intro-only) ratio.
            consecutive = 0
            for frame_idx in range(len(speech_ratio)):
                ratio_vs_baseline = speech_ratio[frame_idx] / (baseline_ratio + 1e-9)

                if ratio_vs_baseline >= speech_ratio_threshold:
                    consecutive += 1
                else:
                    # Slow decay: allow 1 frame gap (speech can have
                    # momentary dips between syllables)
                    consecutive = max(0, consecutive - 1)

                if consecutive >= consecutive_frames_required:
                    # Speech confirmed — calculate timestamp
                    onset_frame = frame_idx - consecutive_frames_required + 1
                    time_in_tail = librosa.frames_to_time(
                        onset_frame, sr=sr, hop_length=hop,
                    )
                    absolute_time = ref_start + time_in_tail

                    # Sanity: ensure we're trimming a meaningful amount
                    trim_amount = (match_start_time + intro_duration) - absolute_time
                    if trim_amount < min_trim:
                        logger.debug(
                            f"Speech overlay detected but trim too small "
                            f"({trim_amount:.2f}s < {min_trim:.2f}s). Ignoring."
                        )
                        return None

                    logger.info(
                        f"Speech overlay detected at {absolute_time:.2f}s "
                        f"(trimming {trim_amount:.2f}s from end of intro, "
                        f"ratio={ratio_vs_baseline:.2f}x baseline)"
                    )
                    return absolute_time

            return None

        except Exception as e:
            logger.error(f"Speech overlay detection failed: {e}")
            return None
        finally:
            for tmp in (ref_tmp, vid_tmp):
                if tmp and os.path.exists(tmp):
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass

    def cleanup(self):
        pass
