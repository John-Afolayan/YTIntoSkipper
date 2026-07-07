DELAYED_OVERRIDE_MIN_TIME = 3.0
DELAYED_OVERRIDE_STRONG_ZERO = 0.85
DELAYED_OVERRIDE_HIGH_RAW = 0.95
DELAYED_OVERRIDE_HIGH_GAIN = 0.12
DELAYED_OVERRIDE_MEDIUM_ZERO = 0.78
DELAYED_OVERRIDE_MEDIUM_RAW = 0.90
DELAYED_OVERRIDE_MEDIUM_GAIN = 0.18


def select_best_candidate(candidates, score_at_zero, logger=None):
    """
    Select the intro candidate.

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
