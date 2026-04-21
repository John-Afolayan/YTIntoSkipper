# sponsorblock_api.py
import json
import os
import time
import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

from logger import logger


class SponsorBlockAPI:
    """Interface to SponsorBlock API with retry on transient failures."""

    MAX_RETRIES = 3
    RETRY_BACKOFF = [2, 5, 15]  # seconds to wait between retries

    def __init__(self, user_id: str = None):
        self.base_url = "https://sponsor.ajay.app/api"
        self.user_id = user_id or os.getenv("SPONSORBLOCK_USER_ID") or self._generate_user_id()

    def _generate_user_id(self):
        import uuid
        return str(uuid.uuid4())

    @staticmethod
    def _is_retryable(exc_or_status):
        """Determine if an error is transient and worth retrying."""
        if isinstance(exc_or_status, int):
            # HTTP status codes: retry on server errors and rate limits
            return exc_or_status in (429, 500, 502, 503, 504)
        if isinstance(exc_or_status, requests.RequestException):
            return True  # network errors are always retryable
        return False

    def _request_with_retry(self, method: str, url: str, **kwargs):
        """
        Make an HTTP request with retry on transient failures.
        Returns the response object or raises on permanent failure.
        """
        last_exc = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.request(method, url, timeout=30, **kwargs)
                if not self._is_retryable(response.status_code):
                    return response
                # Retryable status code
                wait = self.RETRY_BACKOFF[min(attempt, len(self.RETRY_BACKOFF) - 1)]
                logger.warning(
                    f"SponsorBlock returned {response.status_code}. "
                    f"Retry {attempt + 1}/{self.MAX_RETRIES} in {wait}s..."
                )
                time.sleep(wait)
            except requests.RequestException as e:
                last_exc = e
                wait = self.RETRY_BACKOFF[min(attempt, len(self.RETRY_BACKOFF) - 1)]
                logger.warning(
                    f"SponsorBlock request failed: {e}. "
                    f"Retry {attempt + 1}/{self.MAX_RETRIES} in {wait}s..."
                )
                time.sleep(wait)

        # All retries exhausted
        if last_exc:
            raise last_exc
        return response  # return last response even if bad status

    def submit_segment(self, video_id: str, start_time: float, end_time: float,
                       category: str = "intro", action_type: str = "skip") -> bool:
        url = f"{self.base_url}/skipSegments"
        data = {
            "videoID": video_id,
            "segments": [{
                "segment": [start_time, end_time],
                "category": category,
                "actionType": action_type,
            }],
            "userID": self.user_id,
        }

        try:
            logger.info(f"Submitting segment to SponsorBlock: {start_time:.2f}s - {end_time:.2f}s")
            response = self._request_with_retry("POST", url, json=data)
            if response.status_code == 200:
                logger.info("Segment submitted successfully!")
                return True
            else:
                logger.error(f"Failed to submit segment: {response.status_code} - {response.text}")
                return False
        except requests.RequestException as e:
            logger.error(f"API request failed after {self.MAX_RETRIES} retries: {e}")
            return False

    def get_segments(self, video_id: str):
        url = f"{self.base_url}/skipSegments"
        params = {"videoID": video_id, "categories": json.dumps(["intro"])}

        try:
            response = self._request_with_retry("GET", url, params=params)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Found {len(data)} existing intro segment(s) for video {video_id}")
                return data
            elif response.status_code == 404:
                logger.info(f"No existing segments found for video {video_id}")
                return []
            else:
                logger.warning(f"Unexpected response when checking segments: {response.status_code}")
                return []
        except requests.RequestException as e:
            logger.error(f"API request failed after retries: {e}")
            return []

    def delete_segment(self, segment_uuid: str) -> bool:
        url = f"{self.base_url}/api/skipSegments/{segment_uuid}"
        data = {"userID": self.user_id}

        try:
            logger.info(f"Deleting segment: {segment_uuid}")
            response = self._request_with_retry("DELETE", url, json=data)
            if response.status_code == 200:
                logger.info("Segment deleted successfully!")
                return True
            else:
                logger.error(f"Failed to delete segment: {response.status_code} - {response.text}")
                return False
        except requests.RequestException as e:
            logger.error(f"API request failed after retries: {e}")
            return False
