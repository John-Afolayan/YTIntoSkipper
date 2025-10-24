import os
import shutil
import subprocess
import tempfile

from logger import logger

class AudioFingerprinter:
    """Handles audio fingerprinting using Chromaprint/fpcalc"""

    def __init__(self, fpcalc_path: str = "fpcalc"):
        self.fpcalc_path = fpcalc_path
        self._check_dependencies()
        self.temp_dir = tempfile.mkdtemp(prefix="fpcalc_")

    def _check_dependencies(self):
        try:
            subprocess.run([self.fpcalc_path, "-v"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.critical("fpcalc not found. Please install Chromaprint.")
            raise RuntimeError("fpcalc not found. Please install Chromaprint.")

        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.critical("ffmpeg not found. Please install ffmpeg.")
            raise RuntimeError("ffmpeg not found. Please install ffmpeg.")

    def generate_fingerprint(self, audio_path: str, start_time: float = 0, duration: float = None) -> str:
        """Generate audio fingerprint for a segment"""
        # logger.info(f"Generating fingerprint: {audio_path}, start_time={start_time}, duration={duration}")
        audio_to_fingerprint = audio_path

        # If we need to extract a segment, use ffmpeg first
        if start_time > 0 or duration:
            temp_segment = os.path.join(self.temp_dir, f"segment_{os.getpid()}_{start_time}.wav")
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-ss", str(start_time),
            ]
            if duration:
                ffmpeg_cmd.extend(["-t", str(duration)])
            ffmpeg_cmd.extend([
                "-ar", "16000",
                "-ac", "1",
                temp_segment
            ])

            try:
                subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                audio_to_fingerprint = temp_segment
                logger.debug(f"FFmpeg successfully created temp segment: {temp_segment}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to extract audio segment with ffmpeg: {e}")
                return None

        cmd = [self.fpcalc_path, "-raw", "-length", str(int(duration or 120)), audio_to_fingerprint]
        logger.debug(f"Running fpcalc: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.debug(f"fpcalc stdout: {result.stdout}")
            logger.debug(f"fpcalc stderr: {result.stderr}")

            for line in result.stdout.split('\n'):
                if line.startswith('FINGERPRINT='):
                    fingerprint = line.split('=')[1]

                    if start_time > 0 or duration:
                        try:
                            os.remove(audio_to_fingerprint)
                        except Exception as e:
                            logger.warning(f"Failed to remove temp segment: {audio_to_fingerprint} ({e})")

                    return fingerprint

            logger.error("No fingerprint found in fpcalc output")
            raise ValueError("No fingerprint found in fpcalc output")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate fingerprint: {e}")
            if start_time > 0 or duration:
                try:
                    os.remove(audio_to_fingerprint)
                except Exception as e:
                    logger.warning(f"Failed to remove temp segment: {audio_to_fingerprint} ({e})")
            return None

    def compare_fingerprints(self, fp1: str, fp2: str) -> float:
        """Compare two fingerprints and return similarity score (0-1)"""
        if not fp1 or not fp2:
            logger.warning("One or both fingerprints are empty/null")
            return 0.0

        try:
            arr1 = [int(x) for x in fp1.split(',')]
            arr2 = [int(x) for x in fp2.split(',')]

            scores = []
            min_len = min(len(arr1), len(arr2))
            if min_len > 0:
                matches = sum(1 for i in range(min_len) if arr1[i] == arr2[i])
                scores.append(matches / min_len)

            for offset in range(-2, 3):
                total_distance = 0
                comparisons = 0
                for i in range(max(0, -offset), min(len(arr1), len(arr2) - offset)):
                    j = i + offset
                    if 0 <= j < len(arr2):
                        xor_result = arr1[i] ^ arr2[j]
                        bit_diff = bin(xor_result).count('1')
                        similarity = 1.0 - (bit_diff / 32.0)
                        total_distance += similarity
                        comparisons += 1
                if comparisons > 0:
                    scores.append(total_distance / comparisons)

            score = max(scores) if scores else 0.0
            logger.debug(f"Fingerprint similarity score: {score}")
            return score

        except Exception as e:
            logger.error(f"Error comparing fingerprints: {e}")
            return 0.0

    def cleanup(self):
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.debug(f"Cleaned up temp directory: {self.temp_dir}")