"""
Adaptive engine that learns from user feedback to improve future detections.

Responsibilities:
- Recalculate per-channel profiles from accumulated feedback
- Apply bounded corrections to raw detections (duration, start offset)
- Adjust confidence based on channel history
- Detect and reject patterns that repeatedly fail
- Flag detections that deviate significantly from the channel's learned profile
"""
import math
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from feedback_store import FeedbackStore
from models import IntroSegment
from logger import logger


@dataclass
class CorrectionResult:
    """Result of applying adaptive corrections to a detection."""
    original_start: float
    original_end: float
    corrected_start: float
    corrected_end: float
    confidence_adjustment: float  # added to raw confidence for threshold decisions
    flags: list  # warnings/notes for logging
    should_reject: bool = False
    reject_reason: str = ""


class AdaptiveEngine:
    # Minimum feedback entries before applying any corrections
    MIN_SAMPLES = 5

    # Maximum correction bounds (prevents runaway feedback loops)
    MAX_DURATION_CORRECTION = 2.0  # seconds
    MAX_START_CORRECTION = 1.0    # seconds

    # Pattern rejection: if >N denials match a pattern, auto-reject similar
    PATTERN_REJECTION_THRESHOLD = 3

    # Deviation: flag if detection is >N stddevs from channel mean
    DEVIATION_SIGMA = 2.0

    # How often to recalculate (every N new feedback entries)
    RECALC_INTERVAL = 5

    def __init__(self, feedback_store: FeedbackStore):
        self.store = feedback_store
        self._profile_cache: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Profile calculation
    # ------------------------------------------------------------------
    def recalculate_profile(self, channel_id: str) -> Optional[Dict]:
        """
        Recompute the channel profile from all feedback for that channel.
        Uses exponential weighting to prefer recent feedback.
        """
        all_feedback = self.store.get_channel_feedback(channel_id)
        if not all_feedback:
            return None

        approved = [f for f in all_feedback if f["action"] == "approved"]
        denied = [f for f in all_feedback if f["action"] == "denied"]
        total = len(approved) + len(denied)

        if total < self.MIN_SAMPLES:
            logger.debug(
                f"Channel {channel_id}: only {total} feedback entries, "
                f"need {self.MIN_SAMPLES} for profiling."
            )
            return None

        # --- Compute intro duration stats ---
        # We use user-corrected times as ground truth, or approved detections if no correction.
        # Error = Detected - GroundTruth
        # Correction = GroundTruth - Detected = -Error
        durations = []
        start_offsets = []
        start_errors = []
        end_errors = []
        
        # Ground truth data points (weighted)
        samples = []

        # Process from newest to oldest for potential decay weighting
        # For now, we'll collect all valid ground truth points.
        for f in all_feedback:
            # Determine "Ground Truth" for this sample
            gs = f.get("correct_start") if f.get("correct_start") is not None else (f.get("detected_start") if f["action"] == "approved" else None)
            ge = f.get("correct_end") if f.get("correct_end") is not None else (f.get("detected_end") if f["action"] == "approved" else None)
            
            ds = f.get("detected_start")
            de = f.get("detected_end")

            if gs is not None and ge is not None:
                durations.append(ge - gs)
                start_offsets.append(gs)
                
                if ds is not None:
                    # Error = Predicted - Actual
                    start_errors.append(ds - gs)
                if de is not None:
                    end_errors.append(de - ge)

        if not durations:
            return None

        # --- Outlier Rejection ---
        # Before computing the final mean, we filter out samples that are 
        # significant outliers (e.g. > 2.5 stddevs from the median).
        # This prevents one-off "edge case" videos from ruining the profile.
        def _filter_outliers(data):
            if len(data) < 5:
                return data
            data_sorted = sorted(data)
            mid = len(data_sorted) // 2
            median = data_sorted[mid]
            # Use a simple range check for small sample sizes
            q1 = data_sorted[len(data_sorted)//4]
            q3 = data_sorted[3*len(data_sorted)//4]
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return [x for x in data if lower <= x <= upper]

        clean_durations = _filter_outliers(durations)
        clean_starts = _filter_outliers(start_offsets)
        
        # If filtering removed everything (shouldn't happen), fallback to recent
        if not clean_durations: clean_durations = durations
        if not clean_starts: clean_starts = start_offsets

        # Use recent N clean samples
        N_RECENT = 20
        recent_durations = clean_durations[-N_RECENT:]
        recent_starts = clean_starts[-N_RECENT:]

        avg_duration = sum(recent_durations) / len(recent_durations)
        avg_start = sum(recent_starts) / len(recent_starts)
        duration_stddev = _stddev(recent_durations)
        start_stddev = _stddev(recent_starts)

        # Systematic bias (all time, but capped)
        avg_start_error = sum(start_errors) / len(start_errors) if start_errors else 0.0
        avg_end_error = sum(end_errors) / len(end_errors) if end_errors else 0.0

        # --- False positive rate ---
        false_positive_rate = len(denied) / total if total > 0 else 0.0

        # --- Confidence bias ---
        # If FP rate is high (>20%), penalize confidence. If low (<5%), boost.
        confidence_bias = (0.05 - false_positive_rate) * 0.4
        confidence_bias = max(-0.15, min(0.08, confidence_bias))

        profile = {
            "channel_id": channel_id,
            "avg_intro_duration": avg_duration,
            "avg_start_offset": avg_start,
            "duration_stddev": duration_stddev,
            "start_stddev": start_stddev,
            "avg_end_error": avg_end_error,
            "avg_start_error": avg_start_error,
            "confidence_bias": confidence_bias,
            "false_positive_rate": false_positive_rate,
            "sample_count": total,
        }

        self.store.save_channel_profile(profile)
        self._profile_cache[channel_id] = profile

        logger.info(
            f"Profile updated for {channel_id}: dur={avg_duration:.1f}s, "
            f"start={avg_start:.1f}s, bias[start={avg_start_error:+.2f}s, end={avg_end_error:+.2f}s], "
            f"FP={false_positive_rate:.1%}"
        )
        return profile

    def get_profile(self, channel_id: str) -> Optional[Dict]:
        """Get cached profile or load from DB."""
        if channel_id in self._profile_cache:
            return self._profile_cache[channel_id]

        profile = self.store.get_channel_profile(channel_id)
        if profile:
            self._profile_cache[channel_id] = profile
        return profile

    # ------------------------------------------------------------------
    # Apply corrections to a detection
    # ------------------------------------------------------------------
    def apply_corrections(
        self,
        segment: IntroSegment,
        channel_id: str,
    ) -> CorrectionResult:
        """
        Apply learned corrections to a raw detection. Returns a CorrectionResult
        with corrected times, confidence adjustment, and any flags/rejections.
        """
        result = CorrectionResult(
            original_start=segment.start_time,
            original_end=segment.end_time,
            corrected_start=segment.start_time,
            corrected_end=segment.end_time,
            confidence_adjustment=0.0,
            flags=[],
        )

        profile = self.get_profile(channel_id)
        if not profile or profile.get("sample_count", 0) < self.MIN_SAMPLES:
            result.flags.append("insufficient_data")
            return result

        # --- 1. Pattern rejection ---
        # Don't auto-reject if confidence is extremely high (0.95+), unless
        # the channel is notorious for false positives.
        should_reject, reject_reason = self._should_reject_by_pattern(segment, channel_id, profile)
        if should_reject:
            result.should_reject = True
            result.reject_reason = reject_reason
            return result

        # --- 2. Start offset correction ---
        # Correction = GroundTruth - Predicted = -Error
        avg_start_error = profile.get("avg_start_error", 0.0)
        if abs(avg_start_error) > 0.05:  # 50ms threshold
            correction = -avg_start_error
            correction = _clamp(correction, -self.MAX_START_CORRECTION, self.MAX_START_CORRECTION)
            result.corrected_start = max(0.0, segment.start_time + correction)
            if abs(correction) > 0.1:
                result.flags.append(f"start_corrected:{correction:+.2f}s")

        # --- 3. End / duration correction ---
        avg_end_error = profile.get("avg_end_error", 0.0)
        if abs(avg_end_error) > 0.05:
            correction = -avg_end_error
            correction = _clamp(correction, -self.MAX_DURATION_CORRECTION, self.MAX_DURATION_CORRECTION)
            result.corrected_end = segment.end_time + correction
            if abs(correction) > 0.1:
                result.flags.append(f"end_corrected:{correction:+.2f}s")

        # --- 4. Deviation check ---
        detected_duration = segment.end_time - segment.start_time
        avg_duration = profile.get("avg_intro_duration", 0.0)
        duration_std = profile.get("duration_stddev", 0.0)

        if duration_std > 0.1:
            deviation = abs(detected_duration - avg_duration) / duration_std
            if deviation > self.DEVIATION_SIGMA:
                result.flags.append(f"duration_deviation:{deviation:.1f}σ")
                # Penalize confidence for large deviations
                penalty = 0.05 * (deviation - self.DEVIATION_SIGMA)
                result.confidence_adjustment -= min(0.20, penalty)

        # --- 5. Confidence bias from channel history ---
        confidence_bias = profile.get("confidence_bias", 0.0)
        result.confidence_adjustment += confidence_bias

        # --- 6. Profile alignment bonus ---
        if avg_duration > 0 and duration_std > 0.1:
            deviation = abs(detected_duration - avg_duration) / duration_std
            if deviation < 0.3:
                result.confidence_adjustment += 0.04
                result.flags.append("profile_aligned")

        return result

    # ------------------------------------------------------------------
    # Pattern-based rejection
    # ------------------------------------------------------------------
    def _should_reject_by_pattern(
        self,
        segment: IntroSegment,
        channel_id: str,
        profile: Dict,
    ) -> Tuple[bool, str]:
        """
        Check if this detection matches a pattern that's been repeatedly denied.
        """
        patterns = self.store.get_denial_patterns(channel_id)
        
        # Rule 1: "No Intro" reliability
        no_intro_count = patterns.get("no_intro", 0)
        fp_rate = profile.get("false_positive_rate", 0.0)
        
        if no_intro_count >= self.PATTERN_REJECTION_THRESHOLD:
            # If the channel frequently has no intro, be very strict with medium confidence
            if segment.confidence < 0.75:
                return True, f"high_fp_rate_channel (denials={no_intro_count})"

        # Rule 2: Impossible Start Time
        # If intros for this channel usually start at 0s (std < 1s),
        # but this detection is at 60s, it's likely a false positive.
        avg_start = profile.get("avg_start_offset", 0.0)
        start_std = profile.get("start_stddev", 0.0)
        if start_std > 0 and start_std < 2.0:  # Channel has very consistent start times
            if abs(segment.start_time - avg_start) > max(10.0, 4.0 * start_std):
                # But allow high confidence matches to bypass
                if segment.confidence < 0.85:
                    return True, f"atypical_start_time ({segment.start_time:.1f}s vs avg {avg_start:.1f}s)"

        # Rule 3: Extreme Duration Deviation
        avg_dur = profile.get("avg_intro_duration", 0.0)
        dur_std = profile.get("duration_stddev", 0.0)
        if dur_std > 0 and avg_dur > 0:
            det_dur = segment.end_time - segment.start_time
            if abs(det_dur - avg_dur) > max(15.0, 5.0 * dur_std):
                if segment.confidence < 0.85:
                    return True, f"extreme_duration_deviation ({det_dur:.1f}s vs avg {avg_dur:.1f}s)"

        return False, ""

    # ------------------------------------------------------------------
    # Check if recalculation is needed
    # ------------------------------------------------------------------
    def should_recalculate(self, channel_id: str) -> bool:
        """Check if the profile needs recalculation based on new feedback."""
        profile = self.get_profile(channel_id)
        if not profile:
            # No profile yet — check if we have enough data now
            counts = self.store.count_feedback(channel_id)
            total = sum(counts.values())
            return total >= self.MIN_SAMPLES

        stored_count = profile.get("sample_count", 0)
        current_counts = self.store.count_feedback(channel_id)
        current_total = sum(current_counts.values())

        return (current_total - stored_count) >= self.RECALC_INTERVAL

    def maybe_recalculate(self, channel_id: str):
        """Recalculate profile if enough new data has accumulated."""
        if self.should_recalculate(channel_id):
            self.recalculate_profile(channel_id)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _stddev(values: list) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
