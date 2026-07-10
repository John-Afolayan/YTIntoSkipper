import os
import shutil
import subprocess
import sys
from pathlib import Path
from logger import logger


def extract_video_id(url: str) -> str:
    """Extract the 11-char video ID from a YouTube URL (module-level so
    admin tools can use it without instantiating the downloader)."""
    import re
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?]*)',
        r'youtube\.com/shorts/([^&\n?]*)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    # Fallback: if the URL is just an ID (sometimes happens in lists)
    if len(url) == 11 and ' ' not in url:
        return url
    raise ValueError(f"Could not extract video ID from URL: {url}")


def resolve_yt_dlp_command():
    root = Path(__file__).resolve().parent
    candidates = []

    if os.name == "nt":
        candidates.append([str(root / "venv" / "Scripts" / "yt-dlp.exe")])
    else:
        candidates.append([str(root / "venv" / "bin" / "yt-dlp")])

    candidates.append([sys.executable, "-m", "yt_dlp"])

    base_executable = getattr(sys, "_base_executable", None)
    if base_executable and base_executable != sys.executable:
        candidates.append([base_executable, "-m", "yt_dlp"])

    candidates.append(["python", "-m", "yt_dlp"])

    path_cmd = shutil.which("yt-dlp")
    if path_cmd:
        candidates.append([path_cmd])
    candidates.append(["yt-dlp"])

    for cmd in candidates:
        executable = Path(cmd[0]) if os.path.sep in cmd[0] else None
        if executable and not executable.exists():
            continue
        try:
            subprocess.run([*cmd, "--version"], capture_output=True, check=True)
            return cmd
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
            logger.debug(f"yt-dlp candidate failed ({' '.join(cmd)}): {e}")

    raise RuntimeError("yt-dlp not found. Please install yt-dlp.")


# Substrings in yt-dlp stderr that identify an age-gated video
AGE_RESTRICTION_MARKERS = (
    "Sign in to confirm your age",
    "age-restricted",
    "age_verification",
    "This video may be inappropriate",
)


class YouTubeDownloader:
    """Handles audio downloading from YouTube with robust fallbacks."""

    def __init__(self, cookies_file=None, cookies_from_browser=None):
        self.cookies_file = self._resolve_cookies_file(cookies_file)
        # Fallback for age-restricted videos: pull cookies straight from a
        # local browser profile (yt-dlp --cookies-from-browser). Only used
        # on retry after an age-gate failure, to keep normal bulk downloads
        # anonymous.
        self.cookies_from_browser = (
            cookies_from_browser
            or os.getenv("YT_DLP_COOKIES_FROM_BROWSER")
        )
        self._check_dependencies()

    @staticmethod
    def _windows_path_from_wsl(path: str) -> Path:
        if os.name != "nt" or not path.startswith("/mnt/") or len(path) < 7:
            return Path(path)

        drive = path[5]
        rest = path[7:].replace("/", "\\")
        return Path(f"{drive.upper()}:\\{rest}")

    def _resolve_cookies_file(self, cookies_file=None):
        candidates = [
            cookies_file,
            os.getenv("YT_DLP_COOKIES_FILE"),
            os.getenv("COOKIES_FILE"),
        ]

        if os.name == "nt":
            candidates.extend([
                str(Path.home() / "Downloads" / "brave_cookies.txt"),
                "/mnt/c/Users/John/Downloads/brave_cookies.txt",
            ])
        else:
            candidates.extend([
                str(Path.home() / "Downloads" / "brave_cookies.txt"),
                "/mnt/c/Users/John/Downloads/brave_cookies.txt",
            ])

        for candidate in candidates:
            if not candidate:
                continue
            path = self._windows_path_from_wsl(str(candidate))
            if path.exists():
                return str(path)

        return None

    def _check_dependencies(self):
        self.yt_dlp_cmd = resolve_yt_dlp_command()

    def extract_video_id(self, url: str) -> str:
        return extract_video_id(url)

    def download_audio(self, url, output_dir=None):
        """
        Download audio. Returns (absolute_path_to_file, video_id).
        """
        if output_dir is None:
            # Use a 'downloads' subdir relative to this script to avoid clutter
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloads")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        video_id = self.extract_video_id(url)
        
        # We use a specific filename structure so we can easily delete it later
        output_template = str(Path(output_dir) / f"{video_id}.%(ext)s")

        # Simplified attempt logic for brevity - prioritizing m4a
        base_cmd = [
            *self.yt_dlp_cmd,
            "-f", "bestaudio[ext=m4a]/bestaudio", # Prefer m4a, take whatever is best audio otherwise
            "-o", output_template,
            url
        ]

        cookie_args = []
        if self.cookies_file and Path(self.cookies_file).exists():
            cookie_args = ["--cookies", self.cookies_file]
            logger.info(f"Using cookies file: {self.cookies_file}")
        elif self.cookies_file:
            logger.warning(f"Cookies file not found, continuing without cookies: {self.cookies_file}")

        def _run(extra_args):
            cmd = [*self.yt_dlp_cmd, *extra_args, *base_cmd[len(self.yt_dlp_cmd):]]
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )

        logger.info(f"Downloading audio for {video_id}...")
        try:
            _run(cookie_args)
        except subprocess.CalledProcessError as e:
            stderr = e.stderr or ""
            age_gated = any(marker in stderr for marker in AGE_RESTRICTION_MARKERS)

            # Age-gated: retry with browser cookies if configured and the
            # first attempt didn't already have working sign-in cookies.
            if age_gated and self.cookies_from_browser:
                logger.warning(
                    f"{video_id} is age-restricted — retrying with cookies "
                    f"from browser '{self.cookies_from_browser}'..."
                )
                try:
                    _run(["--cookies-from-browser", self.cookies_from_browser])
                except subprocess.CalledProcessError as e2:
                    logger.error(f"yt-dlp failed (with browser cookies): {e2.stderr}")
                    raise RuntimeError(
                        f"age-restricted: download failed for {url} even with "
                        f"cookies from '{self.cookies_from_browser}'. Make sure "
                        f"you are signed in to YouTube in that browser."
                    )
            elif age_gated:
                logger.error(f"yt-dlp failed (age-restricted, no cookies configured): {stderr[:300]}")
                raise RuntimeError(
                    f"age-restricted: {url} requires sign-in. Configure "
                    f"cookies_from_browser in config.toml (or "
                    f"--cookies-from-browser brave), or export a cookies "
                    f"file and set cookies_file / YT_DLP_COOKIES_FILE."
                )
            else:
                logger.error(f"yt-dlp failed: {stderr}")
                raise RuntimeError(f"Download failed for {url}")

        # Find the file (yt-dlp might have used m4a)
        expected_path = Path(output_dir) / f"{video_id}.m4a"
        if expected_path.exists():
            return str(expected_path.absolute()), video_id

        raise FileNotFoundError("yt-dlp ran but output file is missing")
