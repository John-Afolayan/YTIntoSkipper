import os
import math
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.signal import find_peaks
from typing import Tuple, Optional, Iterable, List

from sponsorblock_api import SponsorBlockAPI
from youtube_downloader import YouTubeDownloader
from audio_fingerprint import AudioFingerprinter
from video_db import VideoDB
from feedback_store import FeedbackStore, REASON_CATEGORIES, REASON_LABELS
from adaptive_engine import AdaptiveEngine
from detection_utils import select_start_candidate, resolve_ambiguity
from models import IntroSegment
from logger import logger

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _copy_to_clipboard(text: str) -> bool:
    """Best-effort clipboard copy (works on macOS, Linux w/ xclip/xsel, WSL)."""
    import subprocess, platform, shutil

    system = platform.system()

    if system == "Darwin":
        try:
            subprocess.run(["pbcopy"], input=text.encode(), check=True)
            return True
        except Exception as e:
            logger.debug(f"pbcopy failed: {e}")

    # Try native Linux clipboard tools first
    for tool, cmd in [
        ("xclip", ["xclip", "-selection", "clipboard"]),
        ("xsel", ["xsel", "--clipboard", "--input"]),
    ]:
        if shutil.which(tool):
            try:
                subprocess.run(cmd, input=text.encode(), check=True)
                return True
            except Exception as e:
                logger.debug(f"{tool} failed: {e}")

    # WSL: try clip.exe (Windows clipboard)
    # shutil.which() often misses it because /mnt/c/Windows/System32
    # isn't always on PATH inside Python subprocesses.
    clip_candidates = ["clip.exe", "/mnt/c/Windows/System32/clip.exe"]
    for clip_path in clip_candidates:
        try:
            result = subprocess.run(
                [clip_path],
                input=text.encode(), check=True,
                capture_output=True,
            )
            return True
        except FileNotFoundError:
            logger.debug(f"Not found: {clip_path}")
        except subprocess.CalledProcessError as e:
            logger.debug(f"{clip_path} failed (rc={e.returncode}): {e.stderr}")
        except Exception as e:
            logger.debug(f"{clip_path} error: {e}")

    # PowerShell fallback (another WSL option)
    ps_candidates = [
        "powershell.exe",
        "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
    ]
    for ps_path in ps_candidates:
        try:
            # Escape single quotes in URL
            safe_text = text.replace("'", "''")
            subprocess.run(
                [ps_path, "-NoProfile", "-Command", f"Set-Clipboard -Value '{safe_text}'"],
                check=True, capture_output=True,
            )
            return True
        except FileNotFoundError:
            logger.debug(f"Not found: {ps_path}")
        except subprocess.CalledProcessError as e:
            logger.debug(f"{ps_path} failed (rc={e.returncode}): {e.stderr}")
        except Exception as e:
            logger.debug(f"{ps_path} error: {e}")

    logger.warning(
        "Clipboard copy failed: no working clipboard tool found. "
        "Tried: xclip, xsel, clip.exe, powershell.exe. "
        "Install xclip (sudo apt install xclip) or ensure Windows interop is enabled."
    )
    return False


def _format_timestamp_param(seconds: float) -> str:
    """
    Format seconds into YouTube's ?t= format:
      < 60s       -> ?t=45s
      < 3600s     -> ?t=1m30s
      >= 3600s    -> ?t=1h2m30s
    """
    total = int(seconds)
    if total <= 0:
        return ""
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"?t={h}h{m}m{s}s"
    elif m > 0:
        return f"?t={m}m{s}s"
    else:
        return f"?t={s}s"


def _url_with_timestamp(url: str, start_time: float) -> str:
    """Append ?t= timestamp to a YouTube URL if start_time > 0."""
    if start_time < 1.0:
        return url
    ts = _format_timestamp_param(start_time)
    # Strip any existing ?t= or &t= param
    import re
    url_clean = re.sub(r'[?&]t=[^&]*', '', url)
    separator = "&" if "?" in url_clean else "?"
    return f"{url_clean}{separator}{ts.lstrip('?')}" if ts else url_clean


class IntroSkipperAutomator:
    # CONSTRAINT: Never submit an intro smaller than this
    MIN_INTRO_DURATION = 2.0

    def __init__(
        self,
        manual_approval: bool = False,
        dry_run: bool = False,
        clipboard: bool = False,
        trim_speech: bool = False,
        # Tunable thresholds
        peak_height: float = 0.25,
        weighted_threshold: float = 0.60,
        search_limit: float = 120.0,
        early_exit_threshold: float = 0.90,
        # Parallelism
        workers: int = 1,
        # DB path
        db_path: str = "intro_skipper.db",
        # Channel tracking
        channel_id: str = None,
        # Cookies for age-restricted videos
        cookies_file: str = None,
        cookies_from_browser: str = None,
    ):
        self.fingerprinter = AudioFingerprinter()
        self.downloader = YouTubeDownloader(
            cookies_file=cookies_file,
            cookies_from_browser=cookies_from_browser,
        )
        self.sponsorblock = SponsorBlockAPI()
        self.db = VideoDB(db_path)
        self.manual_approval = manual_approval
        self.dry_run = dry_run
        self.clipboard = clipboard
        self.trim_speech = trim_speech
        self.channel_id = channel_id

        # Feedback & adaptive learning
        self.feedback_store = FeedbackStore(db_path)
        self.adaptive_engine = AdaptiveEngine(self.feedback_store)

        # Tunable
        self.peak_height = peak_height
        self.weighted_threshold = weighted_threshold
        self.search_limit = search_limit
        self.early_exit_threshold = early_exit_threshold
        self.workers = max(1, workers)

        # Multi-reference support: list of (fingerprint_data, duration)
        self._references: List[dict] = []
        self.intro_duration = 0.0

        # Load channel profile on startup if available
        if self.channel_id:
            self.adaptive_engine.maybe_recalculate(self.channel_id)

    # ------------------------------------------------------------------
    # Reference management (supports multiple reference files)
    # ------------------------------------------------------------------
    def set_reference_intro(self, audio_path: str, duration: float = None):
        """Load a single reference intro (backward-compatible)."""
        self._references.clear()
        self.add_reference_intro(audio_path, duration)

    def add_reference_intro(self, audio_path: str, duration: float = None):
        """Add an additional reference intro for matching."""
        logger.info(f"Analyzing reference intro: {audio_path}")
        data = self.fingerprinter.generate_fingerprint(audio_path, duration=duration)
        if data:
            self._references.append(data)
            # Use the longest reference as the canonical duration
            self.intro_duration = max(self.intro_duration, data["duration"])
            logger.info(
                f"Reference loaded ({len(self._references)} total). "
                f"Duration: {data['duration']:.2f}s"
            )
        else:
            raise ValueError(f"Failed to analyze reference intro: {audio_path}")

    @property
    def reference_data(self):
        """Primary reference (first loaded). For backward compat."""
        return self._references[0] if self._references else None

    # ------------------------------------------------------------------
    # Improved position weighting
    # ------------------------------------------------------------------
    @staticmethod
    def _position_weight(time_sec: float) -> float:
        """
        Continuous decay that strongly favours earlier positions.
        Returns a bonus in [0, 0.40] that decays exponentially.
        At 0s -> +0.40, at 20s -> +0.16, at 60s -> +0.02, at 120s -> ~0.
        """
        return 0.40 * math.exp(-0.05 * time_sec)

    # ------------------------------------------------------------------
    # Confidence classification
    # ------------------------------------------------------------------
    # Raw score thresholds for confidence tiers.
    # HIGH: almost certainly correct — can auto-submit.
    # MEDIUM: probably correct — should review.
    # LOW: not confident — skip entirely.
    CONFIDENCE_HIGH = 0.80
    CONFIDENCE_LOW = 0.45   # below this raw score, reject outright

    @classmethod
    def _classify_confidence(cls, raw_score: float) -> str:
        if raw_score >= cls.CONFIDENCE_HIGH:
            return "high"
        elif raw_score >= cls.CONFIDENCE_LOW:
            return "medium"
        else:
            return "low"

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------
    def find_intro_in_video(self, audio_path: str) -> Optional[IntroSegment]:
        if not self._references:
            raise ValueError("No reference intro set")

        best_segment = None
        best_weighted = -1.0

        # Try each reference and pick the best overall match
        for ref_idx, ref_data in enumerate(self._references):
            segment = self._find_with_reference(audio_path, ref_data, ref_idx)
            if segment and segment.confidence > best_weighted:
                best_segment = segment
                best_weighted = segment.confidence

        return best_segment

    def _find_with_reference(
        self, audio_path: str, ref_data: dict, ref_idx: int
    ) -> Optional[IntroSegment]:
        # 1. Scan Video
        scores, sr, hop_length = self.fingerprinter.scan_video(
            ref_data, audio_path,
            search_limit_seconds=self.search_limit,
            early_exit_threshold=self.early_exit_threshold,
        )
        if scores is None or len(scores) == 0:
            return None

        # 2. Peaks
        #    `distance` is measured in CHROMA FRAMES (hop_length samples
        #    each), not audio samples. Require ~2 seconds between peaks:
        #    2 * sr / hop_length frames. (The previous `sr * 2` value was
        #    ~17 minutes of frames, which silently limited every scan to a
        #    single peak — delayed intros never even became candidates.)
        min_peak_distance = max(1, int(2.0 * sr / hop_length))
        peaks, _ = find_peaks(scores, height=self.peak_height, distance=min_peak_distance)
        if len(peaks) == 0:
            return None

        time_per_frame = hop_length / sr
        candidates = []

        # 3. Build candidate list with weighting
        for p in peaks:
            time_sec = p * time_per_frame
            raw_score = scores[p]
            if time_sec < 1.5:
                time_sec = 0.0

            weighted_score = raw_score + self._position_weight(time_sec)
            candidates.append({
                "time": time_sec,
                "raw": raw_score,
                "weighted": weighted_score,
                "frame": p,
            })

        # 3b. Always consider position 0 as a candidate.
        #     find_peaks() can miss the very first frame if the correlation
        #     doesn't have a local valley before it.  If position 0 has a
        #     strong score and no existing candidate covers it, inject it.
        score_at_zero = float(scores[0]) if len(scores) > 0 else 0.0
        has_zero_candidate = any(c["time"] == 0.0 for c in candidates)
        if not has_zero_candidate and score_at_zero >= self.peak_height:
            weighted_zero = score_at_zero + self._position_weight(0.0)
            candidates.append({
                "time": 0.0,
                "raw": score_at_zero,
                "weighted": weighted_zero,
                "frame": 0,
            })
            logger.debug(
                f"Injected position-0 candidate (Raw: {score_at_zero:.3f}, "
                f"Weighted: {weighted_zero:.3f}) — missed by find_peaks"
            )

        candidates.sort(key=lambda x: x["weighted"], reverse=True)

        # Verification callback: measures how well the reference ACTUALLY
        # matches the video at a given start (frame-wise cosine coverage).
        # Cached per position — arbitration and ambiguity resolution may ask
        # about the same timestamp.
        verify_cache: dict = {}

        def _verify(start_time: float):
            key = round(start_time, 2)
            if key not in verify_cache:
                result = self.fingerprinter.verify_match_quality(
                    ref_data, audio_path, start_time,
                )
                if result:
                    logger.info(
                        f"  Verify @ {start_time:.2f}s: "
                        f"coverage={result['coverage']:.2f}, "
                        f"mean_sim={result['mean_similarity']:.2f}"
                    )
                verify_cache[key] = result
            return verify_cache[key]

        best = select_start_candidate(
            candidates, score_at_zero, verify_fn=_verify, logger=logger,
        )

        # 4. Always log score at position 0 for diagnostics
        #    This helps understand edge cases where music is laid over the intro.
        logger.info(f"Score at position 0s: {score_at_zero:.3f}")

        # Log all candidates
        logger.info(f"Found {len(candidates)} candidate(s):")
        for i, c in enumerate(candidates[:5]):
            marker = " <-- selected" if c is best else ""
            logger.info(
                f"  #{i+1}: {c['time']:.2f}s "
                f"(Raw: {c['raw']:.3f}, Weighted: {c['weighted']:.3f}){marker}"
            )
        if best is None:
            return None

        # 5. Ambiguity detection
        #    If the best match is late (>15s) AND position 0 scores almost as
        #    high in *raw* correlation, the fingerprint matches at both places.
        #    This means the intro music is overlaid at 0s — we can't reliably
        #    determine the correct boundaries, so skip for manual review.
        #
        #    We use a RATIO check: position-0 must be within 85% of the best
        #    raw score AND itself be quite strong (>= 0.60) to be considered
        #    a competing match.  This avoids false-flagging cases like
        #    Ibd05fyzKQc where best=0.986 and zero=0.805 (ratio=0.82 — well
        #    below 0.90, so clearly not ambiguous).
        AMBIGUITY_WINDOW = 15.0
        AMBIGUITY_ZERO_MIN = 0.60       # position 0 must be at least this strong
        AMBIGUITY_RATIO_THRESHOLD = 0.90 # zero must be >= 90% of best raw score

        if best["time"] > AMBIGUITY_WINDOW and score_at_zero >= AMBIGUITY_ZERO_MIN:
            ratio = score_at_zero / best["raw"] if best["raw"] > 0 else 0.0
            if ratio >= AMBIGUITY_RATIO_THRESHOLD:
                logger.warning(
                    f"AMBIGUOUS: best match at {best['time']:.2f}s (Raw: {best['raw']:.3f}) "
                    f"but position 0 also scores {score_at_zero:.3f} "
                    f"(ratio: {ratio:.2f} >= {AMBIGUITY_RATIO_THRESHOLD}). "
                    f"Verifying both positions against the reference..."
                )
                decision = resolve_ambiguity(best["time"], 0.0, _verify, logger=logger)
                if decision == "zero":
                    zero_pool = [c for c in candidates if c["time"] == 0.0]
                    best = max(zero_pool, key=lambda c: c["raw"], default=None) or {
                        "time": 0.0,
                        "raw": score_at_zero,
                        "weighted": score_at_zero + self._position_weight(0.0),
                        "frame": 0,
                    }
                elif decision != "best":
                    logger.warning(
                        "Still ambiguous after verification. Skipping for manual review."
                    )
                    return None
            else:
                logger.info(
                    f"Position 0 scores {score_at_zero:.3f} vs best {best['raw']:.3f} "
                    f"(ratio: {ratio:.2f} < {AMBIGUITY_RATIO_THRESHOLD}). "
                    f"Difference is significant — proceeding with best match."
                )

        if len(self._references) > 1:
            logger.info(
                f"  Ref#{ref_idx} selected: {best['time']:.2f}s "
                f"(Raw: {best['raw']:.3f}, Weighted: {best['weighted']:.3f})"
            )
        else:
            logger.info(
                f"Selected: {best['time']:.2f}s "
                f"(Raw: {best['raw']:.3f}, Weighted: {best['weighted']:.3f})"
            )

        # 6. Confidence classification
        confidence_tier = self._classify_confidence(best["raw"])

        if confidence_tier == "low":
            logger.info(
                f"Low confidence ({best['raw']:.3f} < {self.CONFIDENCE_LOW}). "
                f"Skipping — likely no intro in this video."
            )
            return None

        if best["weighted"] < self.weighted_threshold:
            logger.info("Best candidate failed weighted threshold.")
            return None

        start_time = best["time"]
        raw_confidence = best["raw"]
        intro_dur = ref_data["duration"]
        end_time = start_time + intro_dur

        # 3.5 Talk-over guard: if the creator is speaking over the START of
        #     the matched intro (and the reference is instrumental there),
        #     submitting this skip would cut the talking. Cap confidence
        #     below the auto-submit tier so it always gets manual review.
        talkover = self.fingerprinter.detect_talkover_at_start(
            ref_data, audio_path, start_time,
        )
        force_review = False
        if talkover:
            logger.warning(
                f"Speech detected over the start of the matched intro at "
                f"{start_time:.2f}s — likely talk-over content. Forcing "
                f"manual review."
            )
            raw_confidence = min(raw_confidence, self.CONFIDENCE_HIGH - 0.01)
            # The confidence cap alone is NOT enough: a positive adaptive
            # adjustment could push the tier back to "high" downstream.
            force_review = True

        # 4. Adaptive Divergence Check
        actual_end_timestamp = self.fingerprinter.detect_audio_divergence(
            ref_data, audio_path, start_time,
        )

        if actual_end_timestamp:
            proposed_duration = actual_end_timestamp - start_time
            if proposed_duration < self.MIN_INTRO_DURATION:
                logger.warning(
                    f"Divergence at {actual_end_timestamp:.2f}s too short "
                    f"({proposed_duration:.2f}s). Keeping original end: {end_time:.2f}s"
                )
            else:
                new_end = max(start_time, actual_end_timestamp - 0.1)
                logger.info(
                    f"Intro trimmed (divergence at {actual_end_timestamp:.2f}s): "
                    f"{end_time:.2f}s -> {new_end:.2f}s"
                )
                end_time = new_end

        # 5. Speech-over-intro detection (opt-in via --trim-speech)
        #    Only trims the tail — never rejects a match or extends it.
        if self.trim_speech:
            speech_start = self.fingerprinter.detect_speech_overlay(
                ref_data, audio_path, start_time,
            )
            if speech_start is not None:
                proposed_end = max(start_time + self.MIN_INTRO_DURATION, speech_start - 0.15)
                if proposed_end < end_time:
                    logger.info(
                        f"Speech trim: {end_time:.2f}s -> {proposed_end:.2f}s "
                        f"(speaker starts at {speech_start:.2f}s)"
                    )
                    end_time = proposed_end
                else:
                    logger.debug("Speech detected but would not shorten intro. Ignoring.")

        # Add small pre-buffer for intros not at position 0
        if start_time > 0.5:
            start_time = max(0.0, start_time - 0.2)

        # Get full video duration from audio file (soundfile/ffprobe —
        # librosa.get_duration(path=...) on m4a hits the deprecated
        # audioread fallback and will break on librosa 1.0)
        from audio_fingerprint import get_audio_duration
        video_duration = get_audio_duration(audio_path)

        return IntroSegment(
            start_time=start_time,
            end_time=end_time,
            confidence=raw_confidence,
            video_duration=video_duration,
            talkover_warning=talkover,
            force_review=force_review,
        )

    # ------------------------------------------------------------------
    # Process a single video
    # ------------------------------------------------------------------
    def process_video(
        self, url: str, skip_if_exists: bool = True,
    ) -> Tuple[bool, str, str]:
        import time as _time

        audio_path = None
        video_id = "unknown"
        try:
            try:
                video_id = self.downloader.extract_video_id(url)
            except Exception:
                pass

            logger.info(f"Processing: {url}")

            # Check local DB first
            if video_id != "unknown" and skip_if_exists:
                if self.db.is_processed(video_id):
                    logger.info("Skipping (already in local DB)")
                    return True, "skipped", video_id

                existing = self.sponsorblock.get_segments(video_id)
                if any(s.get("category") == "intro" for s in existing):
                    logger.info("Skipping (already has intro on SponsorBlock)")
                    self.db.record(video_id, "skipped_existing")
                    return True, "skipped", video_id

            # Download with retry + backoff (YouTube rate-limits are transient)
            download_retries = 3
            download_backoff = [3, 10, 30]
            for attempt in range(download_retries):
                try:
                    audio_path, video_id = self.downloader.download_audio(url)
                    break  # success
                except RuntimeError as dl_err:
                    err_str = str(dl_err)

                    # Age-restricted: distinct status, retried automatically
                    # on future runs once cookies are configured.
                    if err_str.startswith("age-restricted"):
                        logger.warning(f"Age-restricted video: {err_str}")
                        print(f"\n  [AGE-RESTRICTED] {url}\n  {err_str}")
                        self.db.record(video_id, "error_age_restricted")
                        return False, "failed", video_id

                    # Don't retry on permanent errors (private, removed, etc)
                    permanent_keywords = [
                        "Sign in to confirm",
                        "Private video",
                        "Video unavailable",
                        "removed by the uploader",
                    ]
                    if any(kw in err_str for kw in permanent_keywords):
                        logger.warning(f"Permanent download failure: {err_str}")
                        self.db.record(video_id, "error_permanent")
                        return False, "failed", video_id

                    if attempt < download_retries - 1:
                        wait = download_backoff[attempt]
                        logger.warning(
                            f"Download failed (attempt {attempt + 1}/{download_retries}): "
                            f"{err_str}. Retrying in {wait}s..."
                        )
                        _time.sleep(wait)
                    else:
                        logger.error(f"Download failed after {download_retries} attempts: {err_str}")
                        self.db.record(video_id, "error_download")
                        return False, "failed", video_id

            intro_segment = self.find_intro_in_video(audio_path)

            if intro_segment:
                intro_segment.video_id = video_id
                confidence_tier = self._classify_confidence(intro_segment.confidence)

                # --- Adaptive corrections ---
                correction = None
                if self.channel_id:
                    correction = self.adaptive_engine.apply_corrections(
                        intro_segment, self.channel_id,
                    )
                    if correction.should_reject:
                        logger.info(
                            f"Adaptive rejection ({correction.reject_reason}): {url}"
                        )
                        print(f"\n  [ADAPTIVE SKIP] {correction.reject_reason}: {url}")
                        self.db.record(video_id, "adaptive_rejected")
                        return False, "failed", video_id

                    if correction.flags:
                        for flag in correction.flags:
                            if flag != "insufficient_data":
                                logger.info(f"Adaptive correction: {flag}")

                    # Apply corrections
                    if correction.corrected_start != intro_segment.start_time:
                        logger.info(
                            f"Start adjusted: {intro_segment.start_time:.2f}s -> "
                            f"{correction.corrected_start:.2f}s"
                        )
                        intro_segment.start_time = correction.corrected_start
                    if correction.corrected_end != intro_segment.end_time:
                        logger.info(
                            f"End adjusted: {intro_segment.end_time:.2f}s -> "
                            f"{correction.corrected_end:.2f}s"
                        )
                        intro_segment.end_time = correction.corrected_end

                    # Adjust effective confidence for threshold decisions.
                    # DOWNGRADE ONLY: a positive adjustment (channel
                    # confidence bias, profile-aligned bonus) must never
                    # promote a medium detection to "high" — that would
                    # auto-submit something already flagged for review.
                    effective_confidence = intro_segment.confidence + correction.confidence_adjustment
                    adjusted_tier = self._classify_confidence(effective_confidence)
                    tier_rank = {"low": 0, "medium": 1, "high": 2}
                    if tier_rank[adjusted_tier] < tier_rank[confidence_tier]:
                        confidence_tier = adjusted_tier

                    # Flag deviations for manual review
                    has_deviation = any("deviation" in f for f in correction.flags)
                    if has_deviation and confidence_tier == "high":
                        confidence_tier = "medium"
                        logger.info("Deviation detected — downgrading to medium confidence for review")

                if self.dry_run:
                    self._print_manual_prompt(url, video_id, intro_segment, dry_run=True)
                    self.db.record(video_id, "dry_run", intro_segment.start_time, intro_segment.end_time)
                    return True, "dry_run", video_id

                # Medium confidence (or a hard force_review flag, e.g. the
                # talk-over guard): always require manual review regardless
                # of --manual-approval flag, with a warning.
                needs_review = (
                    self.manual_approval
                    or confidence_tier == "medium"
                    or intro_segment.force_review
                )

                # Parallel workers can't prompt (input() from worker threads
                # is broken/interleaved) — park the video for a sequential
                # re-run instead of submitting unreviewed.
                if needs_review and self.workers > 1:
                    logger.warning(
                        f"Needs manual review but running with --workers "
                        f"{self.workers} — parking {video_id} (status "
                        f"'needs_review'). Re-run sequentially to review it."
                    )
                    print(f"\n  [NEEDS REVIEW] Parked for sequential re-run: {url}")
                    self.db.record(video_id, "needs_review",
                                   intro_segment.start_time, intro_segment.end_time)
                    return False, "needs_review", video_id

                if needs_review:
                    if confidence_tier == "medium" or intro_segment.force_review:
                        print(f"\n  *** MEDIUM CONFIDENCE ({intro_segment.confidence:.2f}) — please verify ***")
                        logger.warning(
                            f"Medium confidence ({intro_segment.confidence:.2f}) for {url}. "
                            f"Forcing manual review."
                        )
                    self._print_manual_prompt(url, video_id, intro_segment)
                    # Build URL with timestamp so the user can jump right to the intro
                    clipboard_url = _url_with_timestamp(url, intro_segment.start_time)
                    if self.clipboard:
                        if _copy_to_clipboard(clipboard_url):
                            print(f"  (Copied to clipboard: {clipboard_url})")
                        else:
                            print(f"  (Clipboard copy failed. URL: {clipboard_url})")
                    if not self._get_user_approval_with_feedback(url, video_id, intro_segment):
                        return False, "skipped", video_id
                else:
                    # Auto-approved (high confidence) — record positive feedback
                    if self.channel_id:
                        self.feedback_store.record_feedback(
                            video_id=video_id,
                            action="approved",
                            channel_id=self.channel_id,
                            detected_start=intro_segment.start_time,
                            detected_end=intro_segment.end_time,
                            raw_confidence=intro_segment.confidence,
                        )

                success = self.sponsorblock.submit_segment(
                    video_id,
                    intro_segment.start_time,
                    intro_segment.end_time,
                    video_duration=intro_segment.video_duration
                )
                if success:
                    self.db.record(video_id, "success", intro_segment.start_time, intro_segment.end_time)
                    # Periodic recalibration
                    if self.channel_id:
                        self.adaptive_engine.maybe_recalculate(self.channel_id)
                    return True, "success", video_id
                else:
                    # API failure is transient — don't record permanently
                    self.db.record(video_id, "error_api", intro_segment.start_time, intro_segment.end_time)
                    return False, "failed", video_id
            else:
                logger.info(f"No intro found (or ambiguous). Manual review needed: {url}")
                print(f"\n  [SKIPPED] No confident intro detected: {url}")
                self.db.record(video_id, "no_intro")
                return False, "failed", video_id

        except RuntimeError as e:
            # Download failures — transient, will be retried
            err_msg = str(e)
            logger.error(f"Download error: {err_msg}")
            self.db.record(video_id, "error_download")
            return False, "failed", video_id
        except Exception as e:
            err_msg = str(e)
            logger.error(f"Process error: {err_msg}")
            # Use error_* prefix so DB treats it as transient (retryable)
            self.db.record(video_id, f"error_process")
            return False, "failed", video_id
        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _print_manual_prompt(self, url, video_id, segment, dry_run=False):
        prefix = "[DRY RUN] " if dry_run else ""
        tier = self._classify_confidence(segment.confidence).upper()
        print(f"\n{'=' * 60}")
        print(f"{prefix}Video: {url}")
        print(f"Detected intro: {segment.start_time:.2f}s - {segment.end_time:.2f}s")
        print(f"Duration: {segment.end_time - segment.start_time:.2f}s")
        print(f"Confidence: {segment.confidence:.2f} ({tier})")
        if getattr(segment, "talkover_warning", False):
            print("*** WARNING: speech detected over the start of this intro — "
                  "the creator may be talking over it. Verify before submitting! ***")
        print(f"{'=' * 60}")

    def _get_user_confirmation(self):
        while True:
            r = input("Submit? (y/n): ").strip().lower()
            if r in ["y", "yes"]:
                return True
            if r in ["n", "no"]:
                return False

    def _get_user_approval_with_feedback(
        self, url: str, video_id: str, segment: IntroSegment,
    ) -> bool:
        """
        Streamlined approval prompt with quick correction support.
        """
        while True:
            r = input("  [y] Approve  [n] Deny  [c] Correct & Submit  [q] Skip video: ").strip().lower()
            if r == "y":
                # Approved as-is
                if self.channel_id:
                    self.feedback_store.record_feedback(
                        video_id=video_id,
                        action="approved",
                        channel_id=self.channel_id,
                        detected_start=segment.start_time,
                        detected_end=segment.end_time,
                        raw_confidence=segment.confidence,
                        weighted_score=getattr(segment, "weighted_score", None),
                    )
                return True
            
            if r == "q":
                return False

            if r == "c":
                # Partial approval / Correction
                print(f"\n  Current: {segment.start_time:.2f}s - {segment.end_time:.2f}s")
                
                def _parse_time(prompt, default):
                    val = input(prompt).strip().lower()
                    if not val:
                        return default
                    # Remove 's', 'sec', etc.
                    clean_val = "".join(c for c in val if c.isdigit() or c in ".-")
                    try:
                        return float(clean_val)
                    except ValueError:
                        print(f"    Invalid input '{val}', using default.")
                        return default

                new_start = _parse_time(f"  Correct start (Enter for {segment.start_time:.2f}s): ", segment.start_time)
                new_end = _parse_time(f"  Correct end   (Enter for {segment.end_time:.2f}s): ", segment.end_time)
                
                # Update the segment object so the caller submits the new times
                segment.start_time = new_start
                segment.end_time = new_end
                
                if self.channel_id:
                    self.feedback_store.record_feedback(
                        video_id=video_id,
                        action="approved", # It's still an approval, just corrected
                        channel_id=self.channel_id,
                        detected_start=segment.start_time,
                        detected_end=segment.end_time,
                        correct_start=new_start,
                        correct_end=new_end,
                        raw_confidence=segment.confidence,
                    )
                return True

            if r == "n":
                # Denied
                break

        # --- Collect denial reason ---
        self.db.record(video_id, "rejected")
        if not self.channel_id:
            return False

        print("\n  Reason? [1] Wrong start [2] Wrong end [3] No intro [4] Too short [5] Too long [6] Other")
        reason_input = input("  Selection (e.g. 3): ").strip()
        
        reason_cats = []
        for part in reason_input.split(","):
            part = part.strip()
            if part in REASON_CATEGORIES:
                reason_cats.append(REASON_CATEGORIES[part])
        reason_categories = ",".join(reason_cats) if reason_cats else None

        self.feedback_store.record_feedback(
            video_id=video_id,
            action="denied",
            channel_id=self.channel_id,
            detected_start=segment.start_time,
            detected_end=segment.end_time,
            reason_categories=reason_categories,
            raw_confidence=segment.confidence,
            weighted_score=getattr(segment, "weighted_score", None),
        )
        print("  Feedback saved.")

        # Check if we should recalculate
        self.adaptive_engine.maybe_recalculate(self.channel_id)
        
        return False

    # ------------------------------------------------------------------
    # Batch processing (sequential or parallel)
    # ------------------------------------------------------------------
    MAX_CONSECUTIVE_ERRORS = 10
    # Hard cap: even 5 simultaneous yt-dlp downloads can trigger
    # YouTube's rate limiter.  SponsorBlock's server (sponsor.ajay.app)
    # resets connections when hit by >3-4 concurrent POSTs.
    MAX_SAFE_WORKERS = 5

    def process_from_source(self, url_source: Iterable[str]):
        results = {"success": 0, "failed": 0, "skipped": 0, "dry_run": 0}

        urls = list(url_source)  # materialise for tqdm + parallel
        total = len(urls)

        if self.workers > self.MAX_SAFE_WORKERS:
            logger.warning(
                f"Capping workers from {self.workers} to {self.MAX_SAFE_WORKERS}. "
                f"Higher values cause YouTube rate-limiting and SponsorBlock "
                f"connection resets."
            )
            self.workers = self.MAX_SAFE_WORKERS

        if self.workers > 1 and not self.manual_approval:
            self._process_parallel(urls, results, total)
        else:
            self._process_sequential(urls, results, total)

        logger.info(
            f"Done. Success:{results['success']}  Failed:{results['failed']}  "
            f"Skipped:{results['skipped']}  DryRun:{results['dry_run']}"
        )

    def _process_sequential(self, urls, results, total):
        iterator = enumerate(urls, 1)
        if tqdm:
            pbar = tqdm(iterator, total=total, desc="Processing", unit="video")
        else:
            pbar = iterator

        consecutive_errors = 0
        try:
            for i, url in pbar:
                url = url.strip()
                if not url or url.startswith("#"):
                    continue
                logger.info(f"\n--- Item {i}/{total} ---")
                success, status, vid = self.process_video(url)
                results[status] = results.get(status, 0) + 1

                # Circuit breaker
                if status == "failed":
                    consecutive_errors += 1
                    if consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                        logger.error(
                            f"Circuit breaker: {consecutive_errors} consecutive failures. "
                            f"Stopping to avoid wasting resources. Check logs for errors."
                        )
                        print(f"\n  *** STOPPED: {consecutive_errors} consecutive failures. Check logs. ***")
                        break
                else:
                    consecutive_errors = 0

                if tqdm and hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(S=results["success"], F=results["failed"])
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")

    def _process_parallel(self, urls, results, total):
        """
        Rate-limited parallel pipeline.

        Architecture:
        - A bounded queue feeds URLs to a small worker pool.
        - Only `workers` (max 5) downloads happen concurrently.
        - A semaphore rate-limits SponsorBlock API calls to 1 at a time.
        - On download/API failure, workers back off before retrying.
        - A shared stop_event halts all workers after too many errors.
        - URLs are fed incrementally (not all at once) to allow early stop.
        """
        import time
        from queue import Queue, Empty

        clean = [u.strip() for u in urls if u.strip() and not u.strip().startswith("#")]

        if tqdm:
            pbar = tqdm(total=len(clean), desc="Processing", unit="video")
        else:
            pbar = None

        # Shared state (protected by lock)
        lock = threading.Lock()
        consecutive_errors = 0
        stop_event = threading.Event()

        # Serialise SponsorBlock API calls: only 1 submission at a time
        sb_semaphore = threading.Semaphore(1)

        # Rate-limit between downloads: each worker sleeps this many seconds
        # before starting the next download.  Starts low, increases on failure.
        download_delay = [1.0]  # mutable so workers can read updated value

        def _process_one(url):
            nonlocal consecutive_errors

            if stop_event.is_set():
                return "skipped"

            # Rate-limit downloads
            time.sleep(download_delay[0])

            if stop_event.is_set():
                return "skipped"

            success, status, vid = self.process_video(url)

            with lock:
                results[status] = results.get(status, 0) + 1

                if status == "failed":
                    consecutive_errors += 1
                    # Adaptive backoff: increase delay when errors pile up
                    download_delay[0] = min(30.0, download_delay[0] * 1.5)
                    logger.debug(f"Download delay increased to {download_delay[0]:.1f}s")

                    if consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                        logger.error(
                            f"Circuit breaker: {consecutive_errors} consecutive "
                            f"failures. Stopping all workers."
                        )
                        print(
                            f"\n  *** STOPPED: {consecutive_errors} consecutive "
                            f"failures. Check logs. ***"
                        )
                        stop_event.set()
                else:
                    consecutive_errors = 0
                    # Ease off the backoff on success
                    download_delay[0] = max(1.0, download_delay[0] * 0.8)

            return status

        # Override submit_segment to serialise through the semaphore.
        # This prevents concurrent API calls that cause ConnectionResetError.
        original_submit = self.sponsorblock.submit_segment

        def _serialised_submit(*args, **kwargs):
            with sb_semaphore:
                # Small pause between SponsorBlock submissions
                import time as _t
                _t.sleep(0.5)
                return original_submit(*args, **kwargs)

        self.sponsorblock.submit_segment = _serialised_submit

        try:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                # Feed URLs in batches instead of submitting all 3000+ at once.
                # This lets the circuit breaker actually stop work.
                batch_size = self.workers * 3
                futures = {}

                idx = 0
                while idx < len(clean) and not stop_event.is_set():
                    # Submit a small batch
                    batch_end = min(idx + batch_size, len(clean))
                    for url in clean[idx:batch_end]:
                        if stop_event.is_set():
                            break
                        f = pool.submit(_process_one, url)
                        futures[f] = url
                    idx = batch_end

                    # Drain completed futures before feeding more
                    done = []
                    for f in list(futures):
                        if f.done():
                            done.append(f)
                    for f in done:
                        try:
                            f.result()
                        except Exception as e:
                            with lock:
                                results["failed"] += 1
                                consecutive_errors += 1
                                logger.error(f"Worker error: {e}")
                                if consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                                    stop_event.set()
                        del futures[f]
                        if pbar:
                            pbar.update(1)
                            pbar.set_postfix(
                                S=results["success"], F=results["failed"]
                            )

                # Wait for remaining futures
                for f in as_completed(futures):
                    if stop_event.is_set():
                        break
                    try:
                        f.result()
                    except Exception as e:
                        with lock:
                            results["failed"] += 1
                            logger.error(f"Worker error: {e}")
                    if pbar:
                        pbar.update(1)
                        pbar.set_postfix(S=results["success"], F=results["failed"])

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
            stop_event.set()
        finally:
            # Restore original method
            self.sponsorblock.submit_segment = original_submit
            if pbar:
                pbar.close()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def cleanup(self):
        self.fingerprinter.cleanup()
        self.db.close()
        self.feedback_store.close()
