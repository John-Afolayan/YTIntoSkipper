"""Candidate selection and arbitration logic for intro detection.

Pure decision logic lives here so it can be unit-tested without audio I/O.
Where audio evidence is needed, a verification callback is injected:

    verify_fn(start_time) -> dict | None

The dict must contain:
    "coverage"         fraction of frames whose cosine similarity to the
                       reference stays above threshold (1.0 = perfect match
                       for the full intro duration)
    "mean_similarity"  mean frame-wise cosine similarity

A true intro match keeps high coverage for the entire reference duration;
a false correlation peak (e.g. harmonically similar music at 0s) collapses
after a few seconds and scores low coverage.
"""

DELAYED_OVERRIDE_MIN_TIME = 3.0
DELAYED_OVERRIDE_STRONG_ZERO = 0.85
DELAYED_OVERRIDE_HIGH_RAW = 0.95
DELAYED_OVERRIDE_HIGH_GAIN = 0.12
DELAYED_OVERRIDE_MEDIUM_ZERO = 0.78
DELAYED_OVERRIDE_MEDIUM_RAW = 0.90
DELAYED_OVERRIDE_MEDIUM_GAIN = 0.18

# --- Verification arbitration thresholds ---
# A delayed candidate only challenges a 0s winner if its raw correlation is
# strong on its own AND not far below the 0s evidence.
VERIFY_MIN_DELAYED_RAW = 0.70
VERIFY_RAW_MARGIN = 0.05
# To flip the decision, the challenger must have decisively better coverage.
VERIFY_MIN_WINNER_COVERAGE = 0.60
VERIFY_COVERAGE_LEAD = 0.20
# A 0s match verified this well is never overridden.
VERIFY_STRONG_COVERAGE = 0.85
# Minimum 0s evidence before we bother re-checking a delayed winner.
VERIFY_ZERO_RECHECK_MIN = 0.55

# --- Ambiguity resolution thresholds (intro music also present at 0s) ---
AMBIGUITY_RESOLVE_MIN_COVERAGE = 0.60
AMBIGUITY_RESOLVE_LEAD = 0.25


def select_best_candidate(candidates, score_at_zero, logger=None):
    """
    Select the intro candidate using correlation heuristics only.

    Position weighting is useful when the intro really starts near 0s and a
    repeat later in the video has a slightly higher raw score. The exception is
    a weak/moderate 0s match losing badly to a very strong delayed match; in
    that case the delayed peak is usually the real intro start.
    """
    if not candidates:
        return None

    ranked = sorted(candidates, key=lambda x: x["weighted"], reverse=True)
    best = ranked[0]

    zero_raws = [float(c["raw"]) for c in ranked if c["time"] == 0.0]
    zero_evidence = max([float(score_at_zero), *zero_raws])

    if best["time"] == 0.0:
        delayed = [
            c for c in ranked
            if c["time"] >= DELAYED_OVERRIDE_MIN_TIME
        ]
        delayed_raw_best = max(delayed, key=lambda x: x["raw"], default=None)

        if delayed_raw_best and _delayed_candidate_clearly_better(
            zero_evidence,
            float(delayed_raw_best["raw"]),
        ):
            if logger:
                logger.info(
                    "Delayed candidate override: "
                    f"{delayed_raw_best['time']:.2f}s raw "
                    f"{delayed_raw_best['raw']:.3f} beats 0s evidence "
                    f"{zero_evidence:.3f}"
                )
            best = delayed_raw_best

    return best


def _delayed_candidate_clearly_better(zero_evidence, delayed_raw):
    if zero_evidence >= DELAYED_OVERRIDE_STRONG_ZERO:
        return False

    raw_gain = delayed_raw - zero_evidence

    if (
        delayed_raw >= DELAYED_OVERRIDE_HIGH_RAW
        and raw_gain >= DELAYED_OVERRIDE_HIGH_GAIN
    ):
        return True

    return (
        zero_evidence < DELAYED_OVERRIDE_MEDIUM_ZERO
        and delayed_raw >= DELAYED_OVERRIDE_MEDIUM_RAW
        and raw_gain >= DELAYED_OVERRIDE_MEDIUM_GAIN
    )


def select_start_candidate(candidates, score_at_zero, verify_fn=None, logger=None):
    """
    Full start-position selection: correlation heuristics first, then — when a
    position-0 candidate and a delayed candidate genuinely compete — audio
    verification arbitrates.

    Falls back to the heuristic winner whenever verification is unavailable
    or inconclusive, so behaviour is unchanged for the common single-peak case.
    """
    best = select_best_candidate(candidates, score_at_zero, logger=logger)
    if best is None or verify_fn is None:
        return best

    zero_pool = [c for c in candidates if c["time"] == 0.0]
    zero_best = max(zero_pool, key=lambda c: c["raw"], default=None)
    delayed_pool = [c for c in candidates if c["time"] >= DELAYED_OVERRIDE_MIN_TIME]
    delayed_best = max(delayed_pool, key=lambda c: c["raw"], default=None)

    if zero_best is None or delayed_best is None:
        return best  # no competition to arbitrate

    zero_evidence = max(float(score_at_zero), float(zero_best["raw"]))

    if best["time"] == 0.0:
        return _maybe_switch_to_delayed(
            best, zero_best, delayed_best, zero_evidence, verify_fn, logger,
        )
    return _maybe_revert_to_zero(
        best, zero_best, zero_evidence, verify_fn, logger,
    )


def _maybe_switch_to_delayed(
    best, zero_best, delayed_best, zero_evidence, verify_fn, logger,
):
    """0s won on weighting — check whether a strong delayed peak is the
    real intro (the classic 'delayed intro marked as 0:00' failure)."""
    if (
        float(delayed_best["raw"]) < VERIFY_MIN_DELAYED_RAW
        or float(delayed_best["raw"]) < zero_evidence - VERIFY_RAW_MARGIN
    ):
        return best  # delayed peak too weak to challenge

    zero_q = verify_fn(zero_best["time"])
    if zero_q and zero_q["coverage"] >= VERIFY_STRONG_COVERAGE:
        if logger:
            logger.info(
                f"0s match verified strongly (coverage {zero_q['coverage']:.2f}) "
                f"— keeping 0s despite delayed peak at {delayed_best['time']:.2f}s"
            )
        return best

    delayed_q = verify_fn(delayed_best["time"])
    if not zero_q or not delayed_q:
        return best  # verification unavailable — keep heuristic winner

    if (
        delayed_q["coverage"] >= VERIFY_MIN_WINNER_COVERAGE
        and delayed_q["coverage"] >= zero_q["coverage"] + VERIFY_COVERAGE_LEAD
    ):
        if logger:
            logger.info(
                f"Verification override: delayed candidate {delayed_best['time']:.2f}s "
                f"(coverage {delayed_q['coverage']:.2f}) beats 0s "
                f"(coverage {zero_q['coverage']:.2f}) — intro is delayed"
            )
        return delayed_best

    if logger:
        logger.debug(
            f"Verification kept 0s: coverage 0s={zero_q['coverage']:.2f} vs "
            f"delayed={delayed_q['coverage']:.2f}"
        )
    return best


def _maybe_revert_to_zero(best, zero_best, zero_evidence, verify_fn, logger):
    """A delayed candidate won (weighting or heuristic override) — confirm it
    against the audio so the override can't hurt a genuine 0s intro."""
    if zero_evidence < VERIFY_ZERO_RECHECK_MIN:
        return best  # 0s never plausible; nothing to confirm against

    delayed_q = verify_fn(best["time"])
    if delayed_q and delayed_q["coverage"] >= VERIFY_MIN_WINNER_COVERAGE:
        return best  # delayed winner verified fine

    zero_q = verify_fn(zero_best["time"])
    if not zero_q or not delayed_q:
        return best

    if (
        zero_q["coverage"] >= VERIFY_MIN_WINNER_COVERAGE
        and zero_q["coverage"] >= delayed_q["coverage"] + VERIFY_COVERAGE_LEAD
    ):
        if logger:
            logger.info(
                f"Verification revert: delayed candidate {best['time']:.2f}s "
                f"failed verification (coverage {delayed_q['coverage']:.2f}) but 0s "
                f"verified (coverage {zero_q['coverage']:.2f}) — keeping 0s"
            )
        return zero_best

    return best


def resolve_ambiguity(best_time, zero_time, verify_fn, logger=None):
    """
    The correlation is high at BOTH a late position and 0s (intro music may be
    overlaid at the start). Verify both positions and decide:

    Returns "best" (proceed with the late match), "zero" (the intro is at 0s),
    or "skip" (still ambiguous — leave for manual review).
    """
    best_q = verify_fn(best_time)
    zero_q = verify_fn(zero_time)
    if not best_q or not zero_q:
        return "skip"

    if (
        best_q["coverage"] >= AMBIGUITY_RESOLVE_MIN_COVERAGE
        and best_q["coverage"] >= zero_q["coverage"] + AMBIGUITY_RESOLVE_LEAD
    ):
        if logger:
            logger.info(
                f"Ambiguity resolved toward {best_time:.2f}s "
                f"(coverage {best_q['coverage']:.2f} vs 0s {zero_q['coverage']:.2f})"
            )
        return "best"

    if (
        zero_q["coverage"] >= AMBIGUITY_RESOLVE_MIN_COVERAGE
        and zero_q["coverage"] >= best_q["coverage"] + AMBIGUITY_RESOLVE_LEAD
    ):
        if logger:
            logger.info(
                f"Ambiguity resolved toward 0s "
                f"(coverage {zero_q['coverage']:.2f} vs {best_time:.2f}s "
                f"{best_q['coverage']:.2f})"
            )
        return "zero"

    return "skip"
