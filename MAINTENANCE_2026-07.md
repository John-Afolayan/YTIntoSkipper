# Maintenance Round — Findings, Results & Responses

*July 10, 2026 — library cleanup, the auto-submit-despite-review bug, config file, wrong-submission cleanup, age-restricted downloads, audit override.*

## 1. The `PySoundFile failed. Trying audioread instead` warning

**Finding.** The warning came from exactly one call: `automator.py` used `librosa.get_duration(path=audio_path)` on the downloaded `.m4a`. libsndfile (soundfile) cannot read m4a/aac, so librosa silently fell back to `audioread` — a fallback that is deprecated and will be **removed in librosa 1.0**, at which point the call would start throwing. All other audio loading in the project was already safe: `AudioFingerprinter._load_audio()` pre-converts to WAV via ffmpeg before `librosa.load()` ever sees the file.

**Response.** New `get_audio_duration()` helper (`audio_fingerprint.py`): tries `soundfile.info()` first (native formats), then `ffprobe` (ships with every ffmpeg install; also probes for it next to the resolved ffmpeg binary). No audioread anywhere in the path, no warning, future-proof against librosa 1.0.

**Library audit results** (`requirements.txt`):

| Package | Verdict | Reason |
|---|---|---|
| `pydub` | **removed** | imported nowhere; unmaintained (last release 2021), breaks on Python 3.13 (`audioop` removal) |
| `silero-vad` | **removed** | imported nowhere; drags in the entire PyTorch stack for nothing |
| `audioread` | **removed** (explicit pin) | deprecated librosa fallback; never imported directly (librosa still installs it transitively while it needs it) |
| `numba`, `soxr` | **removed** (explicit pins) | pure transitive dependencies of librosa — pinning them here just creates upgrade friction |
| `speech_detector.py` | **deleted** | dead module (flagged in FINDINGS.md); VAD logic lives in `AudioFingerprinter` |
| `tomli` | **added** (only for Python < 3.11) | TOML config support; 3.11+ uses stdlib `tomllib` |
| everything else | kept | `librosa`, `soundfile`, `scipy`, `numpy`, `yt-dlp`, `imageio-ffmpeg`, `webrtcvad-wheels`, `requests`, `tqdm`, `python-dotenv` are all actively used |

## 2. "Forcing manual review" … and then it submitted anyway

**Finding — this was a real bug, and it had two independent causes.**

1. **The adaptive engine could *un-force* a review.** The talk-over guard (and plain medium confidence) caps the segment's confidence below the auto-submit tier and logs *"Forcing manual review."* But when `--channel-id` is active, `apply_corrections()` then adds a confidence adjustment that can be **positive** — `confidence_bias` from channel history plus a `+0.04` "profile aligned" bonus. A capped 0.79 (medium) plus even +0.011 crosses the 0.80 "high" threshold, and the code recomputed the tier from that adjusted value. Result: the log honestly said review was forced, then the recomputed tier auto-submitted. This is exactly what hit the pre-October-2022 videos: no real intro exists, chroma correlation against random content still lands in the medium band (speech alone measures ~0.85 raw against an intro reference, per FINDINGS.md), and a well-behaved channel profile pushed it over the line.

2. **Parallel workers can't actually review.** With `--workers > 1`, a review prompt runs `input()` inside a worker thread — interleaved with tqdm and other workers, effectively broken; nothing sane could ever come of it.

**Response.**

- Adaptive confidence adjustments are now **downgrade-only** for tier decisions: a positive adjustment can never promote medium → high. (The adjustment still works in the direction it's trustworthy: pushing suspicious detections *down*.)
- `IntroSegment` gained a hard `force_review` flag, set by the talk-over guard. Once set, no downstream arithmetic can clear it.
- Under `--workers > 1`, a video needing review is **parked** with the new transient status `needs_review` instead of prompting (impossible) or submitting (the bug). A sequential re-run picks all parked videos up for interactive review.
- Defense in depth for your actual scenario: the `cutoff_date` config (below) stops pre-intro-era videos from being analyzed at all.

## 3. Config file (per-channel)

**Response.** New TOML config (`config.py`, template in `config.example.toml`), auto-loaded from `config.toml` or passed via `--config`. I went with per-channel sections over a flat config — you asked which is cleaner, and per-channel is: every parameter that exists as a CLI flag can be set in `[defaults]` and overridden per `[channels.X]`, so channel quirks (this channel's cutoff date, that channel's tighter audit threshold) live next to the channel's URL and references instead of in your shell history.

```toml
[defaults]
diff_threshold = 0.5

[channels.MyChannel]
channel_url = "https://www.youtube.com/@MyChannel"
reference_intros = ["intro.m4a"]
cutoff_date = 2022-10-01
```

- Selection: `--channel-name MyChannel`, or automatic when the config has exactly one channel. The section name doubles as the adaptive-learning `channel_id`.
- Precedence: CLI flag > channel section > defaults > built-in default. A flag only "wins" if you actually typed it.
- **`cutoff_date`** (your October 2022 case): channel mode switches to full-metadata listing (which includes upload dates), skips videos uploaded before the cutoff, and stops the feed entirely after 8 consecutive pre-cutoff videos (feeds stream newest-first; the tolerance absorbs stray out-of-order entries). It applies to both normal processing and `--reprocess`. It cannot work with `--urls-file` (no dates there) — a warning is printed and it's ignored.

## 4. Removing the 8 wrong submissions — done ✅

**Result.** All 8 videos had exactly one intro segment each, **all submitted by our own userID** (public-ID match), so all were fully removable via self-downvote. Removed and then verified gone by re-querying the live API:

| Video | Removed segment |
|---|---|
| `hZGO_4whZlI` | 57.66s – 68.24s |
| `fXPBQDubYsQ` | 0.00s – 10.38s |
| `S7Nt2DhrLH8` | 0.00s – 10.38s |
| `1YjdIZ4Xqp8` | 0.00s – 10.38s |
| `d0aNd4yXpeo` | 0.00s – 10.38s |
| `9WzDZgVQSBU` | 0.00s – 10.38s |
| `Fvm2Jz8CV7U` | 0.00s – 10.38s |
| `GJVF4NtyRU4` | 0.00s – 10.38s |

Each video was also marked `no_intro` in the local DB and `removed` in the audit cache, so neither the pipeline nor `--reprocess` will touch them again.

**The helper for next time** — `--remove-intros` takes any mix of URLs and text files of URLs:

```bash
python main.py --remove-intros bad_urls.txt https://www.youtube.com/watch?v=XXXX
python main.py --remove-intros bad_urls.txt --dry-run    # preview first
```

Foreign segments (not ours) can't be removed — they're downvoted and listed for manual escalation to a SponsorBlock VIP. The helper (and all admin commands) now runs without the heavy audio stack: `main.py` imports the detector lazily.

## 5. Age-restricted videos (e.g. `KHMA6y0GUw8`)

**Finding.** yt-dlp fails these with *"Sign in to confirm your age"*; the pipeline lumped that under generic permanent errors and gave up forever.

**Response.** Two mechanisms, config keys `cookies_file` / `cookies_from_browser` (or the matching CLI flags / `YT_DLP_COOKIES_FROM_BROWSER` env var):

- A Netscape-format **cookies file** is used for all downloads when configured (existing behavior, now configurable instead of hardcoded paths).
- **`cookies_from_browser = "brave"`** (or chrome/firefox/…) is used **only as a retry** after a download fails with an age gate — normal bulk downloads stay anonymous, which also protects your account from association with mass downloading.
- Age-gated failures get their own status `error_age_restricted`, which is deliberately *transient*: the moment you configure cookies, the next run retries exactly those videos. Without cookies configured, the error message tells you which knob to set.

Heads-up for `KHMA6y0GUw8` specifically: close the browser before a `--cookies-from-browser` run on Windows (Chromium-based browsers lock the cookie DB while running), or export a cookies file once and use `cookies_file`.

## 6. Audit override (your `0.00 – 16.00s` case)

**Response.** The audit prompt now has three answers instead of two:

```
  [y] Apply suggested  [n] Keep current  [o] Override with my own times:
    My start (Enter for 0.00s):
    My end   (Enter for 10.38s): 16
```

`[o]` asks for start and end (Enter keeps the suggested value, so your example is just `o`, Enter, `16`), validates end > start, and submits *your* times through the same ownership-aware replacement flow (remove-and-resubmit if we own the old segment, downvote-and-submit-alongside if not). The cache records what was actually submitted.

## Verification

- All modules compile; the 16 detection unit tests pass.
- The 8 segment removals were executed against the live SponsorBlock API and verified gone by re-query (see §4).
- Not exercised end-to-end in this round: an age-restricted download retry with real browser cookies (needs your signed-in browser), and a full channel run with `cutoff_date` (needs a long yt-dlp full-metadata listing). Both paths are small deltas over code that already ran; if the cutoff scan misbehaves, the `STOP_AFTER_CONSECUTIVE_OLD = 8` tolerance in `main.py` is the knob to look at.

## Files touched

`requirements.txt`, `audio_fingerprint.py` (duration helper), `automator.py` (review fixes, cookies pass-through, duration call), `models.py` (`force_review`), `video_db.py` (status notes), `youtube_downloader.py` (age-gate retry, module-level `extract_video_id`), `reprocessor.py` (override prompt, `removed` status, lazy imports), `main.py` (config wiring, cutoff, `--remove-intros`, lazy heavy imports), **new:** `config.py`, `config.example.toml`, `segment_admin.py`; **deleted:** `speech_detector.py`; docs: `README.md`, `REPROCESSING.md`, this file.
