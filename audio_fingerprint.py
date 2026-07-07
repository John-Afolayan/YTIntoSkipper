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


_ffmpeg_executable: str | None = None  # cached check


def _get_ffmpeg_executable() -> str | None:
    global _ffmpeg_executable
    if _ffmpeg_executable is not None:
        return _ffmpeg_executable

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        _ffmpeg_executable = ffmpeg_path
        return _ffmpeg_executable

    try:
        import imageio_ffmpeg
        _ffmpeg_executable = imageio_ffmpeg.get_ffmpeg_exe()
        return _ffmpeg_executable
    except Exception:
        return None


def _has_ffmpeg() -> bool:
    return _get_ffmpeg_executable() is not None


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
        ffmpeg_cmd = _get_ffmpeg_executable()
        if not ffmpeg_cmd:
            raise RuntimeError("ffmpeg not found")

        cmd = [
            ffmpeg_cmd, "-y", "-v", "quiet",
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
            # KeyError on caches written before "speech_env" existed is
            # caught below -> treated as a cache miss -> regenerated.
            return {
                "type": str(d["type"]),
                "data": d["data"],
                "raw_chroma": d["raw_chroma"],
                "duration": float(d["duration"]),
                "sr": int(d["sr"]),
                "speech_env": d["speech_env"],
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
            speech_env=fp["speech_env"],
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
                # Per-frame speech-band (300Hz-3kHz) energy of the reference
                # itself. Speech-overlay detection compares the video's tail
                # against this to find energy the intro doesn't account for.
                "speech_env": self._speech_band_envelope(y, sr),
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
    # Match verification
    # ------------------------------------------------------------------
    def verify_match_quality(
        self,
        needle_data: dict,
        video_path: str,
        start_time: float,
        sim_threshold: float = 0.70,
    ) -> dict | None:
        """
        Measure how well the reference actually matches the video at a
        candidate start position, frame by frame.

        Cross-correlation peaks can be inflated by harmonically similar
        music, especially at position 0. A true intro match keeps high
        cosine similarity for the FULL reference duration; a false peak
        collapses after a few seconds. Coverage captures that difference.

        Returns:
            {"mean_similarity": float, "coverage": float, "frames": int}
            or None if the audio could not be analysed.
        """
        wav_path = None
        try:
            duration = needle_data["duration"]
            ref_chroma = needle_data["raw_chroma"]
            sr = needle_data["sr"]

            y_vid, _, wav_path = self._load_audio(
                video_path, offset=start_time, duration=duration,
            )
            if len(y_vid) < sr * 1.0:
                return None

            vid_chroma = librosa.feature.chroma_cqt(y=y_vid, sr=sr)

            min_cols = min(ref_chroma.shape[1], vid_chroma.shape[1])
            if min_cols < 20:
                return None

            ref = ref_chroma[:, :min_cols]
            vid = vid_chroma[:, :min_cols]
            ref = ref / (np.linalg.norm(ref, axis=0) + 1e-9)
            vid = vid / (np.linalg.norm(vid, axis=0) + 1e-9)
            similarity = np.sum(ref * vid, axis=0)

            return {
                "mean_similarity": float(np.mean(similarity)),
                "coverage": float(np.mean(similarity >= sim_threshold)),
                "frames": int(min_cols),
            }

        except Exception as e:
            logger.error(f"Match verification failed at {start_time:.2f}s: {e}")
            return None
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
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
    # Speech-band envelope helper
    # ------------------------------------------------------------------
    SPEECH_ENV_N_FFT = 2048
    SPEECH_ENV_HOP = 512

    @classmethod
    def _speech_band_envelope(cls, y: np.ndarray, sr: int) -> np.ndarray:
        """Per-frame energy in the speech band (300Hz-3kHz)."""
        S = np.abs(librosa.stft(y, n_fft=cls.SPEECH_ENV_N_FFT, hop_length=cls.SPEECH_ENV_HOP))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=cls.SPEECH_ENV_N_FFT)
        band = (freqs >= 300) & (freqs <= 3000)
        return np.sum(S[band, :] ** 2, axis=0)

    @staticmethod
    def _syllabic_modulation_ratio(envelope: np.ndarray, fs_env: float) -> float | None:
        """
        Fraction of the envelope's modulation energy in the syllabic band
        (3-9 Hz). Speech scores high; music/sound effects score low.
        Returns None if the envelope is too short to analyse.
        """
        if len(envelope) < 32:
            return None
        env = envelope / (np.max(envelope) + 1e-12)
        env = env - np.mean(env)
        mod_spectrum = np.abs(np.fft.rfft(env)) ** 2
        mod_freqs = np.fft.rfftfreq(len(env), d=1.0 / fs_env)
        syllabic = (mod_freqs >= 3.0) & (mod_freqs <= 9.0)
        broad = (mod_freqs >= 0.5) & (mod_freqs <= 20.0)
        denom = float(np.sum(mod_spectrum[broad])) + 1e-12
        return float(np.sum(mod_spectrum[syllabic])) / denom

    # Modulation ratio above this = speech-like. Calibrated on real intro
    # music (0.07-0.26) vs real speech (mostly 0.31-0.65).
    SPEECH_MOD_THRESHOLD = 0.35

    # ------------------------------------------------------------------
    # VAD confirmation for speech-overlay candidates
    # ------------------------------------------------------------------
    _vad_warned = False  # class-level: warn about missing webrtcvad only once

    def _vad_confirms_speech(
        self,
        seg: np.ndarray,
        sr: int,
        min_speech_frames: int = 6,
    ) -> bool | None:
        """
        Run webrtcvad over a confirmation window (sliced by the caller) and
        report whether it contains actual speech.

        Returns True (speech), False (no speech), or None (VAD unavailable
        or too little audio — caller decides the fallback).
        """
        try:
            import webrtcvad
        except ImportError:
            if not AudioFingerprinter._vad_warned:
                logger.warning(
                    "webrtcvad not installed — speech-overlay confirmation "
                    "falls back to syllabic-modulation analysis. Install it "
                    "for more reliable results: pip install webrtcvad-wheels"
                )
                AudioFingerprinter._vad_warned = True
            return None

        try:
            target_sr = 16000
            if len(seg) < sr * 0.4:
                return None  # not enough audio to judge

            seg16 = librosa.resample(seg, orig_sr=sr, target_sr=target_sr)
            pcm = np.clip(seg16 * 32767.0, -32768, 32767).astype(np.int16)

            # Mode 3 = most aggressive at filtering out non-speech; with
            # music underneath we'd rather miss borderline speech than trim
            # on a sound effect.
            vad = webrtcvad.Vad(3)
            frame_len = int(target_sr * 0.03)  # 30ms frames
            speech_frames = 0
            total_frames = 0
            for i in range(0, len(pcm) - frame_len + 1, frame_len):
                frame = pcm[i:i + frame_len].tobytes()
                total_frames += 1
                if vad.is_speech(frame, target_sr):
                    speech_frames += 1

            if total_frames == 0:
                return None
            return speech_frames >= min_speech_frames

        except Exception as e:
            logger.debug(f"VAD confirmation failed: {e}")
            return None

    def _modulation_confirms_speech(
        self,
        seg: np.ndarray,
        sr: int,
        ratio_threshold: float = 0.35,
    ) -> bool | None:
        """
        Fallback speech confirmation when webrtcvad is unavailable.

        Speech carries strong 3-9 Hz amplitude modulation in the speech band
        (the syllable rate). Sound effects, risers, and music sweeps have
        smooth or slow envelopes and score low. Calibrated on real intro
        music (0.07-0.26) vs real speech (mostly 0.31-0.65); 0.35 rejects
        all music samples — a false "no" here only means no trim, which is
        the safe direction.

        Returns True (speech-like modulation), False (not speech-like),
        or None (too little audio to judge).
        """
        try:
            if len(seg) < sr * 1.0:
                return None  # need >= ~1s for 3 Hz modulation resolution

            envelope = self._speech_band_envelope(seg, sr)
            fs_env = sr / self.SPEECH_ENV_HOP  # envelope sample rate (~43 Hz)
            ratio = self._syllabic_modulation_ratio(envelope, fs_env)
            if ratio is None:
                return None

            logger.debug(f"Syllabic modulation ratio: {ratio:.3f}")
            return ratio >= ratio_threshold

        except Exception as e:
            logger.debug(f"Modulation speech check failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Talk-over-at-start detection
    # ------------------------------------------------------------------
    # Sub-window size matches the calibration of SPEECH_MOD_THRESHOLD —
    # the modulation-spectrum distribution shifts with window length.
    SPEECH_WINDOW_SEC = 1.6
    SPEECH_WINDOW_HOP_SEC = 0.8

    def detect_talkover_at_start(
        self,
        needle_data: dict,
        video_path: str,
        match_start_time: float,
        head_seconds: float = 4.0,
        min_speech_windows: int = 2,
    ) -> bool:
        """
        Detect whether the creator is speaking over the HEAD of the matched
        intro. If the reference intro is instrumental at that point but the
        video has speech there, the match is a talk-over — skipping it
        would cut the creator's talking.

        Scans the head in overlapping 1.6s sub-windows and requires
        `min_speech_windows` speech-like windows to reduce false alarms.
        The check disables itself (returns False) when the reference intro
        contains vocals in its head, since speech detection can't
        distinguish the creator from the intro's own singing.

        Returns True only when talk-over is confidently detected.
        """
        wav_path = None
        try:
            sr = needle_data["sr"]
            intro_duration = needle_data["duration"]
            ref_env = needle_data.get("speech_env")
            if ref_env is None:
                return False

            head = min(head_seconds, intro_duration - 0.5)
            win = self.SPEECH_WINDOW_SEC
            hop_w = self.SPEECH_WINDOW_HOP_SEC
            if head < win:
                return False

            fs_env = sr / self.SPEECH_ENV_HOP
            window_starts = np.arange(0.0, head - win + 0.01, hop_w)

            # Reference vocal check: any speech-like sub-window in the
            # reference head means we can't judge — stay quiet.
            for w in window_starts:
                seg_env = ref_env[int(w * fs_env):int((w + win) * fs_env)]
                r = self._syllabic_modulation_ratio(seg_env, fs_env)
                if r is not None and r >= self.SPEECH_MOD_THRESHOLD:
                    logger.debug(
                        f"Reference intro head is speech-like at +{w:.1f}s "
                        f"(ratio {r:.2f}) — talk-over check disabled."
                    )
                    return False

            y, _, wav_path = self._load_audio(
                video_path, offset=match_start_time, duration=head,
            )
            if len(y) < sr * win:
                return False

            votes = 0
            for w in window_starts:
                seg = y[int(w * sr):int((w + win) * sr)]
                if len(seg) < sr * 1.0:
                    continue
                says_speech = self._vad_confirms_speech(seg, sr)
                if says_speech is None:
                    says_speech = self._modulation_confirms_speech(seg, sr)
                if says_speech:
                    votes += 1
                    if votes >= min_speech_windows:
                        return True

            return False

        except Exception as e:
            logger.debug(f"Talk-over check failed: {e}")
            return False
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Speech-over-intro detection
    # Compares the video's speech-band energy (300Hz-3kHz) against the
    # REFERENCE INTRO's own speech-band envelope, frame-aligned and
    # gain-calibrated. Energy the intro itself doesn't account for =
    # someone talking over it. Only scans the TAIL of the intro to stay
    # conservative, and every energy candidate must be confirmed as
    # actual speech (VAD, or syllabic modulation as fallback).
    # ------------------------------------------------------------------
    def detect_speech_overlay(
        self,
        needle_data: dict,
        video_path: str,
        match_start_time: float,
        tail_seconds: float = 4.0,
        min_trim: float = 0.3,
        excess_energy_threshold: float = 2.0,
        consecutive_frames_required: int = 4,
    ) -> float | None:
        """
        Detect where a speaker starts talking over the intro music.

        Strategy:
          1. The fingerprint stores the reference intro's per-frame
             speech-band energy envelope ("speech_env").
          2. Calibrate the video's gain against the reference using the
             EARLY part of the matched intro (known speech-free).
          3. In the tail, compute per-frame excess:
                 video_energy / (gain * reference_energy)
             Consistent excess above threshold = added sound.
          4. Confirm the added sound is speech (VAD, or syllabic-modulation
             fallback) before reporting it.

        Args:
            needle_data: Reference fingerprint data (must contain speech_env).
            video_path: Path to the downloaded video audio.
            match_start_time: Where the intro was detected in the video.
            tail_seconds: How many seconds from the end of the intro to scan.
            min_trim: Minimum seconds to trim (ignore detections smaller than this).
            excess_energy_threshold: How much more speech-band energy the video
                must have vs the gain-scaled reference to count as an overlay
                candidate. 2.0 = double the energy. Higher = more conservative.
            consecutive_frames_required: How many consecutive frames must
                exceed the threshold to flag a candidate. Higher = fewer
                false positives but might miss very short speech.

        Returns:
            Absolute timestamp where speech overlay begins, or None.
        """
        ref_tmp = None
        vid_tmp = None
        try:
            intro_duration = needle_data["duration"]
            sr = needle_data["sr"]
            ref_env_full = needle_data.get("speech_env")
            if ref_env_full is None:
                logger.warning(
                    "Fingerprint has no speech_env (stale cache?) — "
                    "speech-overlay detection skipped. Delete .fingerprint_cache "
                    "to regenerate."
                )
                return None

            # Only scan the tail portion of the intro
            scan_duration = min(tail_seconds, intro_duration - 1.0)
            if scan_duration < 1.0:
                return None  # intro too short to meaningfully scan

            tail_offset_in_intro = intro_duration - scan_duration
            ref_start = match_start_time + tail_offset_in_intro

            # --- Load video tail audio ---
            y_vid, _, vid_tmp = self._load_audio(
                video_path, offset=ref_start, duration=scan_duration,
            )

            if len(y_vid) < sr * 0.5:
                return None  # too little audio

            hop = self.SPEECH_ENV_HOP
            env_vid = self._speech_band_envelope(y_vid, sr)

            # --- Gain calibration on the EARLY (speech-free) intro part ---
            baseline_duration = min(3.0, intro_duration - scan_duration - 0.5)
            if baseline_duration < 1.0:
                return None

            y_baseline, _, ref_tmp = self._load_audio(
                video_path, offset=match_start_time, duration=baseline_duration,
            )
            env_base_vid = self._speech_band_envelope(y_baseline, sr)

            fs_env = sr / hop
            ref_env_early = ref_env_full[:len(env_base_vid)]
            ref_early_med = float(np.median(ref_env_early))
            if ref_early_med <= 1e-12:
                return None  # degenerate reference
            gain = float(np.median(env_base_vid)) / ref_early_med
            if gain <= 0:
                return None

            # --- Frame-aligned excess energy in the tail ---
            tail_start_frame = int(round(tail_offset_in_intro * fs_env))
            ref_env_tail = ref_env_full[tail_start_frame:tail_start_frame + len(env_vid)]
            n = min(len(ref_env_tail), len(env_vid))
            if n < consecutive_frames_required + 2:
                return None

            from scipy.signal import medfilt
            env_vid_s = medfilt(env_vid[:n], kernel_size=5)
            ref_scaled = medfilt(gain * ref_env_tail[:n], kernel_size=5)
            # Floor keeps quiet reference moments (fade-outs) from exploding
            # the ratio on noise alone.
            floor = 0.05 * (float(np.median(ref_scaled)) + 1e-12)
            excess = env_vid_s / (ref_scaled + floor)

            # --- Scan for speech onset ---
            # Consistent excess energy the intro doesn't account for is an
            # overlay CANDIDATE. It must then be confirmed as actual speech —
            # otherwise a bass drop, riser, or sound effect in the intro
            # tail would trigger a bogus trim.
            consecutive = 0
            for frame_idx in range(n):
                ratio_vs_baseline = excess[frame_idx]

                if ratio_vs_baseline >= excess_energy_threshold:
                    consecutive += 1
                else:
                    # Slow decay: allow 1 frame gap (speech can have
                    # momentary dips between syllables)
                    consecutive = max(0, consecutive - 1)

                if consecutive >= consecutive_frames_required:
                    # Energy candidate — calculate timestamp
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

                    # --- Speech confirmation: VAD first, syllabic
                    # modulation as fallback when VAD is unavailable ---
                    # The window stays INSIDE the intro tail: sliding it
                    # earlier for near-end candidates avoids two problems —
                    # windows too short to analyse, and contamination from
                    # normal post-intro speech just past the intro end.
                    tail_len_sec = len(y_vid) / sr
                    win_start = max(0.0, min(time_in_tail - 0.2, tail_len_sec - 1.0))
                    win_end = min(time_in_tail + 1.4, tail_len_sec)
                    confirm_seg = y_vid[int(win_start * sr):int(win_end * sr)]

                    vad_says_speech = self._vad_confirms_speech(confirm_seg, sr)
                    if vad_says_speech is None:
                        vad_says_speech = self._modulation_confirms_speech(
                            confirm_seg, sr,
                        )
                    if vad_says_speech is False:
                        logger.info(
                            f"Energy spike at {absolute_time:.2f}s "
                            f"(excess={ratio_vs_baseline:.2f}x) NOT confirmed as "
                            f"speech — likely a sound effect. Continuing scan."
                        )
                        consecutive = 0
                        continue

                    confirmation = "speech-confirmed" if vad_says_speech else "energy-only"
                    logger.info(
                        f"Speech overlay detected at {absolute_time:.2f}s "
                        f"({confirmation}, trimming {trim_amount:.2f}s from end "
                        f"of intro, ratio={ratio_vs_baseline:.2f}x baseline)"
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
