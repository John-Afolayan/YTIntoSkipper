# youtube_downloader.py
import os
import subprocess
from pathlib import Path
from logger import logger

class YouTubeDownloader:
    """Handles audio downloading from YouTube with robust fallbacks."""

    def __init__(self, cookies_file=None):
        self.cookies_file = cookies_file
        self._check_dependencies()

    def _check_dependencies(self):
        """Check if yt-dlp is installed"""
        try:
            subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("yt-dlp not found. Please install yt-dlp.")

    def extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL"""
        import re
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?]*)',
            r'youtube\.com/shorts/([^&\n?]*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise ValueError(f"Could not extract video ID from URL: {url}")

    def download_audio(self, url, output_dir=os.path.dirname(os.path.abspath(__file__)), audio_filename_base=None):
        """
        Download audio from a URL with fallback strategies.
        Returns tuple (path_to_downloaded_file, video_id).
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Determine video_id early and use as default filename base
        video_id = self.extract_video_id(url)
        if not audio_filename_base:
            audio_filename_base = video_id

        output_template = str(Path(output_dir) / f"{audio_filename_base}.%(ext)s")

        attempts = []

        # 1) Android client + extract & convert to m4a (preferred)
        attempts.append({
            "name": "android-extract-m4a",
            "cmd": [
                "yt-dlp",
                "--extractor-args", "youtube:player_client=android",
                "-x", "--audio-format", "m4a", "--audio-quality", "0",
                "-o", output_template,
                url
            ],
        })

        # 2) Android client + cookies (if provided)
        if self.cookies_file:
            attempts.append({
                "name": "android-extract-m4a+cookies",
                "cmd": [
                    "yt-dlp",
                    "--cookies", self.cookies_file,
                    "--extractor-args", "youtube:player_client=android",
                    "-x", "--audio-format", "m4a", "--audio-quality", "0",
                    "-o", output_template,
                    url
                ],
            })

        # 3) Direct m4a-only format (bestaudio[ext=m4a]) fallback
        attempts.append({
            "name": "bestaudio_m4a",
            "cmd": [
                "yt-dlp",
                "-f", "bestaudio[ext=m4a]",
                "-x", "--audio-format", "m4a",
                "-o", output_template,
                url
            ],
        })

        # 4) m4a + cookies
        if self.cookies_file:
            attempts.append({
                "name": "bestaudio_m4a+cookies",
                "cmd": [
                    "yt-dlp",
                    "--cookies", self.cookies_file,
                    "-f", "bestaudio[ext=m4a]",
                    "-x", "--audio-format", "m4a",
                    "-o", output_template,
                    url
                ],
            })

        last_err = None
        for attempt in attempts:
            logger.info(f"Attempting yt-dlp download (method: {attempt['name']})")
            try:
                proc = subprocess.run(
                    attempt["cmd"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    timeout=600  # adjust if you need longer
                )
            except subprocess.TimeoutExpired as e:
                logger.error("yt-dlp timed out for %s: %s", attempt['name'], e)
                last_err = str(e)
                continue
            except Exception as e:
                logger.exception("Exception during %s download for %s: %s", attempt['name'], url, e)
                last_err = str(e)
                continue

            # Debug logs
            logger.debug("yt-dlp stdout (%s):\n%s", attempt['name'], proc.stdout)
            logger.debug("yt-dlp stderr (%s):\n%s", attempt['name'], proc.stderr)

            if proc.returncode == 0:
                # Look for expected audio file (common extensions)
                for ext in ("m4a", "webm", "opus", "mka", "mp3", "wav"):
                    candidate = Path(output_dir) / f"{audio_filename_base}.{ext}"
                    if candidate.exists():
                        downloaded_file = str(candidate)
                        logger.info("Download succeeded with method '%s': %s", attempt['name'], downloaded_file)
                        return downloaded_file, video_id

                # If yt-dlp returned 0 but file not found
                logger.error("yt-dlp exited 0 but no audio file was produced (method: %s). Stdout/stderr in DEBUG logs.", attempt['name'])
                last_err = proc.stderr or proc.stdout
            else:
                logger.error(
                    "Download attempt '%s' failed for %s\nSTDOUT: %s\nSTDERR: %s",
                    attempt['name'], url, proc.stdout, proc.stderr
                )
                last_err = proc.stderr or proc.stdout

        logger.critical("All yt-dlp download attempts failed for %s", url)
        raise RuntimeError(last_err or f"Could not download audio for {url} using any fallback.")
