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
        # Prefer venv/bin/yt-dlp if it exists (local to this project)
        local_yt_dlp = Path(__file__).resolve().parent / "venv" / "bin" / "yt-dlp"
        self.yt_dlp_cmd = "yt-dlp"
        if local_yt_dlp.exists():
            self.yt_dlp_cmd = str(local_yt_dlp)

        try:
            subprocess.run([self.yt_dlp_cmd, "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            if self.yt_dlp_cmd != "yt-dlp":
                 # Try global as fallback
                 try:
                     subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
                     self.yt_dlp_cmd = "yt-dlp"
                     return
                 except (subprocess.CalledProcessError, FileNotFoundError):
                     pass
            raise RuntimeError("yt-dlp not found. Please install yt-dlp.")

    def extract_video_id(self, url: str) -> str:
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
        cmd = [
            self.yt_dlp_cmd,
            "-f", "bestaudio[ext=m4a]/bestaudio", # Prefer m4a, take whatever is best audio otherwise
            "-x", "--audio-format", "m4a",        # Force convert to m4a for consistency
            "--audio-quality", "0",
            "--cookies", "/mnt/c/Users/John/Downloads/brave_cookies.txt",
            "-o", output_template,
            url
        ]

        if self.cookies_file:
            cmd.insert(1, "--cookies")
            cmd.insert(2, self.cookies_file)

        try:
            logger.info(f"Downloading audio for {video_id}...")
            subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                check=True, 
                text=True
            )
            
            # Find the file (yt-dlp might have used m4a)
            expected_path = Path(output_dir) / f"{video_id}.m4a"
            if expected_path.exists():
                return str(expected_path.absolute()), video_id
            
            raise FileNotFoundError("yt-dlp ran but output file is missing")

        except subprocess.CalledProcessError as e:
            logger.error(f"yt-dlp failed: {e.stderr}")
            raise RuntimeError(f"Download failed for {url}")