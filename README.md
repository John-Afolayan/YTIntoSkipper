# YouTube Intro Skipper

Automated YouTube intro detection and [SponsorBlock](https://sponsor.ajay.app/) submission tool. Given a reference intro audio file, it scans YouTube videos, detects where the intro plays using audio fingerprinting (chroma cross-correlation), and submits the intro segment to SponsorBlock so viewers can skip it automatically.

## How It Works

1. **Reference fingerprinting** — You provide one or more audio files of the channel's intro music. The tool generates a chroma-based fingerprint for each.
2. **Video scanning** — For each target video, the tool downloads the first N seconds of audio (default: 120s) and runs normalized cross-correlation against the reference fingerprint(s).
3. **Candidate ranking** — Correlation peaks are scored with position weighting (earlier matches score higher). Ambiguity detection identifies overlapping intro music at position 0.
4. **Boundary refinement** — Adaptive divergence detection trims the end point by comparing frame-by-frame cosine similarity between the reference and video. Optional speech-overlay detection (`--trim-speech`) detects when the creator starts talking over the intro tail.
5. **Adaptive learning** — When `--channel-id` is provided, the system records feedback from approvals and denials, builds per-channel profiles, and applies statistical corrections to improve future accuracy over time.
6. **Submission** — Confident detections are submitted to SponsorBlock as "intro" segments. Medium-confidence detections require manual review.

## Requirements

### System Dependencies

- **Python 3.10+**
- **ffmpeg** — Required for audio format conversion
  - Linux/WSL: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: `winget install ffmpeg`
- **yt-dlp** — YouTube downloading (installed via pip, but ensure it's on PATH)

### Python Dependencies

```bash
pip install -r requirements.txt
```

Key packages: `librosa`, `scipy`, `numpy`, `yt-dlp`, `requests`, `tqdm`, `python-dotenv`

## Setup

1. Clone the repository and create a virtualenv:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS/WSL
   # or: venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root:
   ```env
   SPONSORBLOCK_USER_ID=your-uuid-here
   CHANNEL_URL=https://www.youtube.com/@YourChannel
   ```
   If `SPONSORBLOCK_USER_ID` is not set, a random UUID is generated (anonymous submissions).

3. Prepare a reference intro audio file — extract the intro music from one of the channel's videos (e.g., using Audacity or ffmpeg):
   ```bash
   ffmpeg -i sample_video.mp4 -ss 0 -t 10.4 -vn -ac 1 -ar 22050 intro.m4a
   ```

## Usage

### Basic Run (File-Based)

```bash
python main.py --reference-intro intro.m4a --urls-file urls.txt
```

### Channel Mode (Live Feed)

Process all videos from a channel directly:
```bash
python main.py --reference-intro intro.m4a --channel "https://www.youtube.com/@ChannelName"
```

### With Adaptive Learning

Enable per-channel learning so the system improves over time:
```bash
python main.py \
  --reference-intro intro.m4a \
  --channel "https://www.youtube.com/@ChannelName" \
  --channel-id ChannelName \
  --manual-approval
```

When `--channel-id` is set:
- Approved submissions are recorded as positive feedback
- Denied submissions prompt for structured feedback (reason, correct timestamps)
- After 5+ data points, the engine applies corrections to future detections

### Multiple Reference Intros

Some channels have intro variants (e.g., short vs long version). Supply multiple references:
```bash
python main.py \
  --reference-intro intro_v1.m4a \
  --reference-intro intro_v2.m4a \
  --urls-file urls.txt
```

### Parallel Processing

Speed up batch runs with multiple workers (max 5 to avoid rate-limiting):
```bash
python main.py --reference-intro intro.m4a --urls-file urls.txt --workers 3
```

### Dry Run

Detect intros without submitting to SponsorBlock:
```bash
python main.py --reference-intro intro.m4a --urls-file urls.txt --dry-run
```

## CLI Reference

### Input Options

| Flag | Description |
|------|-------------|
| `--reference-intro PATH` | Path to reference intro audio (can specify multiple times) |
| `--urls-file PATH` | Text file with one YouTube URL per line |
| `--channel URL` | YouTube channel URL (processes all videos) |

### Behaviour

| Flag | Description |
|------|-------------|
| `--intro-duration SECS` | Override intro duration (default: auto-detect from reference) |
| `--manual-approval` | Require confirmation before each submission |
| `--dry-run` | Detect intros but don't submit |
| `--clipboard` | Copy video URL with timestamp during manual approval |
| `--trim-speech` | Detect speech over intro tail and trim the skip endpoint |

### Tuning

| Flag | Default | Description |
|------|---------|-------------|
| `--peak-height` | 0.25 | Minimum raw correlation peak height |
| `--weighted-threshold` | 0.60 | Minimum weighted score to accept a match |
| `--search-limit` | 120.0 | Seconds of each video to scan |
| `--early-exit` | 0.90 | Score above which to stop scanning early |

### Performance

| Flag | Default | Description |
|------|---------|-------------|
| `--workers` | 1 | Parallel workers (max 5) |

### Channel & Adaptive Learning

| Flag | Description |
|------|-------------|
| `--channel-id NAME` | Channel identifier for per-channel learning |
| `--show-profile` | Display learned channel profile and exit |
| `--recalibrate` | Force recalculation of channel profile from feedback |
| `--feedback-stats` | Show feedback statistics for the channel |

### Database

| Flag | Default | Description |
|------|---------|-------------|
| `--db PATH` | `intro_skipper.db` | SQLite database path |
| `--db-stats` | | Print database statistics and exit |
| `--db-reset` | | Clear the entire database |
| `--db-reset-errors` | | Clear transient errors (retry failed videos) |

### SponsorBlock Admin

| Flag | Description |
|------|-------------|
| `--user-id UUID` | Override SponsorBlock user ID |
| `--list-segments VIDEO_ID` | List all intro segments for a video |
| `--delete-video VIDEO_ID` | Delete intro segments for a video (not yet implemented) |

## Adaptive Learning System

The adaptive feedback loop allows the script to learn from your corrections and improve over time. This is critical for running confidently on thousands of videos without manual intervention.

### How It Works

1. **Feedback collection** — When you deny a detection during `--manual-approval`, the system prompts for:
   - Reason(s): wrong start, wrong end, no intro, too short, too long, other (multiple allowed)
   - Correct start/end timestamps (optional)
   - Free-text note (optional)

2. **Profile building** — After 5+ feedback entries for a channel, the engine computes:
   - Average intro duration and start offset (with standard deviations)
   - Systematic end-time and start-time errors
   - False positive rate
   - Common denial patterns

3. **Corrections applied** — On subsequent runs:
   - End-time bias correction (bounded ±2s)
   - Start-time offset correction (bounded ±1s)
   - Confidence adjustment based on channel accuracy history
   - Pattern rejection (auto-skips detections matching repeatedly-denied patterns)
   - Deviation flagging (forces review if detection is >2σ from channel norm)

4. **Safe defaults** — With <5 data points, no corrections are applied. All corrections are bounded to prevent feedback spirals.

### Viewing Your Channel Profile

```bash
# Show the learned profile
python main.py --show-profile --channel-id ChannelName --db intro_skipper.db

# Show feedback statistics
python main.py --feedback-stats --channel-id ChannelName --db intro_skipper.db

# Force recalculation after manual DB edits
python main.py --recalibrate --channel-id ChannelName --db intro_skipper.db
```

## Populating URLs

The `populate_urls.py` utility generates a `urls.txt` from a YouTube channel:

```bash
# Fast mode (flat playlist, no date filtering)
python populate_urls.py --channel "https://www.youtube.com/@ChannelName"

# With date filtering (uses full metadata mode)
python populate_urls.py --channel "https://www.youtube.com/@ChannelName" --from 2023-01-01 --to 2024-12-31

# Custom output file
python populate_urls.py --channel "https://www.youtube.com/@ChannelName" --output my_urls.txt
```

Environment variables (`CHANNEL_URL`, `DATE_FROM`, `DATE_TO`) can also be set in `.env`.

## Project Structure

```
.
├── main.py                 # CLI entry point and argument parsing
├── automator.py            # Core orchestration (detection + submission pipeline)
├── audio_fingerprint.py    # Chroma fingerprinting, cross-correlation, divergence detection
├── sponsorblock_api.py     # SponsorBlock API client with retry logic
├── youtube_downloader.py   # yt-dlp wrapper for audio downloading
├── video_db.py             # SQLite tracker for processed videos (deduplication)
├── feedback_store.py       # Feedback and channel profile storage
├── adaptive_engine.py      # Learning engine (corrections, pattern rejection)
├── models.py               # Data models (IntroSegment)
├── populate_urls.py        # Channel URL list generator via yt-dlp
├── logger.py               # Logging setup (file + console)
└── requirements.txt        # Python dependencies
```

## Confidence Tiers

| Tier | Raw Score | Behaviour |
|------|-----------|-----------|
| High | >= 0.80 | Auto-submit (unless `--manual-approval`) |
| Medium | 0.45 - 0.80 | Always requires manual review |
| Low | < 0.45 | Rejected automatically (no intro detected) |

## Tips

- **Start with `--manual-approval --dry-run`** to verify detection quality before submitting anything.
- **Use `--clipboard`** during manual review to quickly open the video at the detected timestamp.
- **Enable `--channel-id` from the start** so the learning system accumulates data even during initial manual runs.
- **Multiple references help** when a channel has intro variations (different lengths, remixes, etc.).
- **If detection quality is poor**, try adjusting `--peak-height` (lower = more sensitive) and `--weighted-threshold` (lower = more permissive).
- **Rate limiting**: YouTube rate-limits aggressive downloading. Use `--workers 2-3` for a balance of speed and reliability. The system has built-in adaptive backoff and a circuit breaker (stops after 10 consecutive failures).
