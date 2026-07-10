# Re-processing Existing Submissions — Findings & Design

*July 2026 — adding an audit mode to fix videos that were automated with poorly analyzed intro submissions.*

## Summary

A new **audit mode** (`--reprocess`) walks a channel (or URL file), compares each video's *live* SponsorBlock intro submission against what the current detector suggests, and interactively fixes divergent ones:

```bash
# Command A — respects the audit cache
python main.py --reprocess --reference-intro intro.m4a --channel "https://www.youtube.com/@ChannelName"

# Command B — re-audits everything, ignoring the cache
python main.py --reprocess --ignore-cache --reference-intro intro.m4a --channel "https://www.youtube.com/@ChannelName"
```

Per video the flow is:

1. **Cache check** — videos already audited (in the `reprocessed_videos` table) are skipped, unless `--ignore-cache` is passed. This cache is completely separate from the main pipeline's `processed_videos` dedup table, so audit runs never interfere with normal runs.
2. **Fetch the live submission** — the video's current intro segment(s) from SponsorBlock. Videos with no intro submission are recorded as `no_submission` and skipped (nothing to audit — the normal pipeline handles those).
3. **Re-analyze** — the audio is downloaded and run through the current detector (same references, same verification/arbitration logic as the main pipeline).
4. **Compare** — if both `|Δstart|` and `|Δend|` are under the threshold (default **0.5s**, tune with `--diff-threshold`), the submission is fine → recorded as `ok`, move on. Sub-tolerance jitter (±0.05s etc.) is deliberately ignored.
5. **Propose** — divergent videos print a comparison and wait for `y`/`yes` or `n`/`no`:

   ```
   ================================================================
   Video: https://www.youtube.com/watch?v=XXXXXXXXXXX
     Current submission:  35.13s - 45.51s  (votes: 0, ours: yes)
     Suggested segment:   0.00s - 10.38s   (confidence 0.99 HIGH)
     Difference:          start Δ 35.13s, end Δ 35.13s  (threshold 0.50s)
   ================================================================
     Apply this correction? (y/n):
   ```

6. **Fix on approval** — how the fix is applied depends on segment *ownership* (see below).

`--dry-run` works in audit mode too: it prints the comparisons but changes nothing and caches nothing. `--clipboard` copies the video URL (with the suggested timestamp) during review.

## The ownership problem

SponsorBlock has **no edit or delete API for segments you don't own**. The only sanctioned mutation primitives are:

| Action | Endpoint | Who can do it |
|---|---|---|
| Remove a segment | `POST /api/voteOnSponsorTime` with `type=0` (downvote) **from the original submitter's userID** | Submitter only — a self-downvote removes the segment entirely |
| Downvote a segment | same endpoint, any userID | Anyone — but one downvote from a normal user does *not* hide a segment; hiding requires the score to drop low enough (or a VIP downvote) |
| Submit a competing segment | `POST /api/skipSegments` | Anyone |

There is no `DELETE /skipSegments/{uuid}` in the public API — the previous `delete_segment()` in this repo called a nonexistent endpoint and has been rewritten on top of the vote endpoint.

### How we detect ownership

`GET /api/segmentInfo?UUIDs=[...]` returns each segment's **public userID**. SponsorBlock derives the public ID from the private one as `sha256` applied **5000 times**. `SponsorBlockAPI.get_public_user_id()` computes ours locally and `is_own_segment()` compares. No network auth involved — ownership is provable purely by hashing.

### What we do per case

| Case | Action on approval | Cache status |
|---|---|---|
| **We own the old segment** (submitted with the current `SPONSORBLOCK_USER_ID`) | Self-downvote (removes it), then submit the corrected segment with `force=True` (bypasses the "similar segment exists" guard, needed for corrections < 2.5s apart) | `corrected` |
| **Foreign segment** (random one-time ID we no longer have) | We **cannot remove it**. Best effort: downvote it with our ID (nudges its score down), submit the corrected segment **alongside** it, and list the video in the end-of-run "manual attention" report | `corrected_unowned` |
| **Locked segment** (a SponsorBlock VIP locked it) | Nothing possible via API — votes are ignored and same-category submissions are rejected. Recorded and listed for manual attention | `locked` |

### Why "submit alongside" is the right fallback for unowned segments

SponsorBlock's client behavior makes this workable: when multiple overlapping segments of the same category exist, the extension prefers the better-voted one, and segments whose score drops below the hide threshold stop being served. So for a foreign bad segment:

- our downvote pushes it toward hidden (it takes more downvotes from other users, or a VIP, to finish the job),
- our corrected segment coexists and can be upvoted (you can upvote it from your own browser extension since you know it's correct),
- viewers reporting the bad skip ("this skipped too much") add further downvotes organically.

The honest limitation: **we cannot guarantee the bad foreign segment disappears immediately.** If a video really matters, the escalation path is manual: ask a VIP in the SponsorBlock Discord (#segment-review) to remove/lock it — the "manual attention" list printed at the end of each audit run (and the `corrected_unowned` rows in the cache) is exactly the list to bring there.

### Recommendation going forward

This situation exists because some submissions were made with random one-time IDs. Always run with a **stable** `SPONSORBLOCK_USER_ID` in `.env` — then every future submission is correctable by this tool. The audit run itself submits corrections under the stable ID, so even `corrected_unowned` videos become self-correctable *for the new segment* from now on.

## The audit cache

A new table `reprocessed_videos` (same SQLite file, `--db`) stores one row per audited video: status, the old segment, the suggested segment, and a timestamp.

- **Permanent statuses** (skipped on future runs): `ok`, `corrected`, `corrected_unowned`, `denied`, `no_submission`, `no_detection`, `locked`.
- **Transient statuses** (`error_*` — download/API failures): retried automatically on the next run.
- `--ignore-cache` re-audits everything regardless (results still update the cache).
- `--reprocess-stats` prints a status breakdown of the cache.

Because it's a separate table, resetting or re-running audits never touches the main pipeline's dedup state, and vice versa.

## Implementation notes

- `reprocessor.py` — `IntroReprocessor` (audit flow) + `ReprocessCache` (thread-safe SQLite, same per-thread-connection pattern as `VideoDB`).
- Detection is **reused**, not duplicated: the reprocessor holds an `IntroSkipperAutomator` and calls its `find_intro_in_video()` — so verification, arbitration, talk-over guard and divergence trimming all apply identically. Adaptive per-channel corrections are *not* applied in audit mode; comparisons are against the pure detector output.
- `sponsorblock_api.py` additions: `get_public_user_id()`, `get_segment_info()`, `is_own_segment()`, `vote_segment()`, and a `force=` parameter on `submit_segment()` to bypass the pre-existing "similar segment already exists" guard (without it, corrections smaller than 2.5s would be silently swallowed).
- Audit mode is always **sequential** — it's interactive by design, so `--workers` doesn't apply.
- If multiple intro segments exist on a video, the one *closest* to the suggestion is treated as "the current submission" for comparison and replacement.
- The `talkover_warning` flag from the detector is surfaced in the comparison prompt so you know to double-check those.

## Failure behavior

- Removing our own old segment but failing to submit the replacement would leave the video with *no* intro segment — so if the self-downvote fails, we **abort before submitting**; if the submission fails after removal, the video is recorded as `error_api` (transient) and retried next run, and the failure is printed.
- Circuit breaker: 10 consecutive failures stop the run (same policy as the main pipeline).
- `Ctrl+C` exits cleanly; everything already audited stays cached.
