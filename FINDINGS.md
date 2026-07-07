# Detection Hardening — Findings & Results

*July 2026 — investigation into the delayed-intro misidentification and speech-over-intro edge cases.*

## Summary

Three root causes were found and fixed. All fixes are **evidence-based and conservative**: the common case (intro at 0s) keeps its exact previous behavior unless real audio evidence says otherwise, and every new guard fails toward "do nothing" or "ask for manual review" — never toward a wrong submission.

| Problem | Root cause | Fix |
|---|---|---|
| Delayed intro marked as `0:00–10:38` | `find_peaks` distance bug + position weighting rewarding false 0s matches | Peak-distance fix + frame-wise **match verification** that arbitrates 0s vs delayed candidates |
| Speech-over-intro detection unreliable | Energy *fraction* metric mathematically saturated (couldn't exceed ~1.07× on this intro; threshold was 1.8×) | Compare video tail against the **reference intro's own energy envelope**, gain-calibrated |
| Non-speech sounds triggering trims | No actual speech check | **Two-factor confirmation**: energy excess + VAD (or syllabic-modulation fallback) |
| *(bonus)* Talk-over at video start submitted as skippable | Nothing checked whether the creator talks over the matched intro head | **Talk-over guard** caps confidence below auto-submit so it always gets manual review |

## Root Cause 1: The `find_peaks` distance bug (delayed intros)

`automator.py` called:

```python
find_peaks(scores, height=self.peak_height, distance=sr * 2)   # sr = 22050
```

`distance` is measured in **chroma frames** (one per `hop_length=512` samples ≈ 23 ms), not audio samples. `sr * 2 = 44100` frames ≈ **17 minutes** — longer than the whole 120 s scan window. The peak finder could therefore only ever return **one** peak per scan. If anything near 0s out-scored nothing (which it always does), a true delayed intro never even entered the candidate list.

Fixed to `2 * sr / hop_length` ≈ 86 frames (2 seconds). Scans now return full candidate lists (45 candidates on the sample video vs 1 before).

## Root Cause 2: Position weighting can't be fixed by more weighting

Chroma features describe pitch-class content, so *any* harmonic audio at 0s correlates surprisingly well with an intro fingerprint. Measured on real audio: **pure speech scored 0.846 raw correlation** against the intro reference. With the flat +0.40 position bonus at 0s, a false 0s match beats a true delayed match at 0.95+ raw.

Threshold tuning can't fix this because the false score comes from the same signal being thresholded. The fix is a second, independent measurement:

### Match verification (`verify_match_quality`)

For a candidate start position, load the video at that offset for the full reference duration and compute **frame-wise cosine similarity** against the reference chroma. Report:

- `coverage` — fraction of frames ≥ 0.70 similarity
- `mean_similarity`

A true match keeps high similarity for the *entire* intro; a false correlation peak collapses after a few seconds. On the delayed-intro test: true position verified at coverage **1.00**, the false 0s position at **0.30**.

### Arbitration rules (`detection_utils.select_start_candidate`)

Verification only runs when a 0s candidate and a strong delayed candidate (raw ≥ 0.70 and within 0.05 of the 0s evidence) genuinely compete — the common single-peak case does **zero** extra work:

1. If 0s wins on weighting but verifies < 0.85 coverage and the delayed candidate verifies ≥ 0.60 with a ≥ 0.20 lead → **switch to delayed**.
2. If a delayed candidate wins (including via the raw-score heuristic override) but fails verification while 0s verifies well → **revert to 0s** (protects the common case from the override).
3. Verification unavailable or indecisive → keep the heuristic winner (previous behavior).

The "AMBIGUOUS → skip" path (high correlation at both a late position and 0s) now also verifies both positions and proceeds when one side is decisive, instead of always dumping to manual review.

## Root Cause 3: The speech-band ratio could never fire

The old speech-overlay detector compared the *fraction* of energy in the speech band (300 Hz–3 kHz) against the intro's own baseline fraction. Measured on the real intro: **93.6% of the intro music's energy already lives in that band**, so the ratio was capped at ~1.07× baseline — the 1.8× threshold was mathematically unreachable. On bass-heavy intros the same metric could spike on non-speech. This is why `--trim-speech` "broke the common cases" while missing real talk-overs.

### New approach: reference-envelope comparison

The fingerprint now stores the reference intro's per-frame speech-band energy envelope (`speech_env`, cached; **old caches are auto-regenerated**). Detection:

1. Calibrate the video's gain against the reference using the early (speech-free) part of the matched intro.
2. In the tail, compute per-frame `video_energy / (gain × reference_energy)` — energy the intro itself doesn't account for.
3. Sustained excess ≥ 2× = overlay *candidate*.
4. **Confirm it's speech** before trimming (see below). Rejected candidates resume the scan instead of aborting.

## Speech vs non-speech confirmation

Two-factor, in priority order:

1. **webrtcvad** (mode 3, most conservative) when installed. `requirements.txt` now lists `webrtcvad-wheels` (drop-in fork with prebuilt wheels — plain `webrtcvad` fails to build on Windows without MSVC; note neither ships wheels for Python 3.14 yet, where the fallback below is used).
2. **Syllabic-modulation fallback** (no dependencies): speech has characteristic 3–9 Hz amplitude modulation (the syllable rate) in the speech band; music/sweeps/sound-effects have smooth or slow envelopes. Calibrated on this repo's real audio: intro music scores 0.07–0.26, speech 0.31–0.65 → threshold 0.35 rejects all music samples. A false "not speech" only means no trim — the safe direction.

Confirmation windows are clamped inside the intro tail (sliding earlier near the end) so they can't be contaminated by normal post-intro speech.

## Bonus: talk-over-at-start guard

The repo's own `test.m4a` turned out to be a real instance of the hardest case: the creator talks from 0s with intro music underneath (speech detected in every head window), while the clean reference is instrumental everywhere. Previously this detected as a confident `0.00–10.38` skip — which would cut the talking.

New guard (`detect_talkover_at_start`): after a match is selected, scan the matched intro's head in 1.6 s sub-windows. If the **reference** is instrumental there but the **video** has ≥ 2 speech-like windows, the segment gets `talkover_warning`, its confidence is capped below the auto-submit tier (forcing manual review), and the review prompt shows an explicit warning. If the reference intro itself has vocals, the guard disables itself rather than guess.

## Test Results

Unit tests: 16/16 pass (`python -m unittest test_detection_utils`), covering the heuristic selection, verification arbitration (switch/keep/revert/fallback), and ambiguity resolution with stubbed verifiers.

End-to-end on real + synthetic audio (reference = `intro.m4a`, 10.38 s):

| Scenario | Result | Verdict |
|---|---|---|
| Intro placed 60 s into a video | `59.80–70.38`, conf 0.997 | ✅ was the reported failure mode |
| Real speech mixed over last 2.5 s of intro | trimmed to `7.74` (true onset 7.88, −0.15 safety margin) | ✅ 10 ms onset accuracy |
| Synthetic whoosh over last 2.5 s | **no trim** (`0.00–10.38`) | ✅ non-speech correctly ignored |
| Real talk-over video (`test.m4a`) | conf capped to 0.790, `talkover_warning=True` → forced review | ✅ previously auto-submitted 0.832 |

## Other changes

- `.fingerprint_cache/` and `downloads/` added to `.gitignore`.
- Dead/confused commentary code removed from `detect_speech_overlay`.
- `speech_detector.py` (standalone webrtcvad wrapper) remains unused — the VAD logic now lives in `AudioFingerprinter` where it can share loaded audio. Candidate for deletion.

## Honest limitations

- **Verification costs one extra audio decode per arbitrated position** (~0.2 s each). It only triggers when candidates genuinely compete.
- **Intros with vocals** disable both the talk-over guard and weaken speech confirmation (VAD still helps when installed). This is deliberate — guessing wrong there would break the common case.
- **The modulation fallback was calibrated on one channel's audio.** The 0.35 threshold has good margin (0.26 max music vs 0.31 min speech in samples), but a music genre with strong 3–9 Hz rhythm (some EDM/dubstep) could score speech-like. Consequence is capped: worst case is an unnecessary manual review or a skipped trim, never a bad auto-submission.
- **Talk-over spanning the entire intro** (like `test.m4a`) is flagged for review, not auto-resolved — there is no obviously correct segment to submit in that case.
