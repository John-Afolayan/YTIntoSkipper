"""Evidence-based candidate selection, shared by normal and audit detection."""
import math


def select_start_candidate(candidates, score_at_zero=0.0, verify_fn=None, logger=None):
    """Verify every plausible peak, including single peaks and endpoints.

    Failure to obtain independent evidence is rejection, not permission to
    auto-submit a chroma match. Similar verified repeats prefer the first
    occurrence. Candidate times always retain their measured alignment.
    """
    if verify_fn is None:
        return None
    verified = []
    for candidate in candidates:
        raw, time = float(candidate["raw"]), float(candidate["time"])
        if not math.isfinite(raw) or not math.isfinite(time) or raw < 0.45 or time < 0:
            continue
        quality = verify_fn(time)
        if not quality or not quality.get("verified", False):
            continue
        evidence = float(quality.get("quality", 0.0))
        if math.isfinite(evidence) and evidence > 0:
            verified.append((candidate, min(raw, evidence)))
    if not verified:
        if logger:
            logger.info("No candidate passed independent audio verification.")
        return None
    strongest = max(evidence for _, evidence in verified)
    cutoff = .80 if strongest >= .80 else strongest - .05
    return min((c for c, evidence in verified if evidence >= cutoff),
               key=lambda c: c["time"])
