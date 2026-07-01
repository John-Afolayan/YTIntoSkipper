# sponsorblock_api.py
import json
import os
import time
import requests
import hashlib
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

from logger import logger


class SponsorBlockAPI:
    """Interface to SponsorBlock API with retry on transient failures."""

    MAX_RETRIES = 3
    RETRY_BACKOFF = [2, 5, 15]  # seconds to wait between retries
    USER_AGENT = "YTIntroSkipper/1.1 (github.com/your-username/YTIntroSkipper)"

    def __init__(self, user_id: str = None):
        self.base_url = "https://sponsor.ajay.app/api"
        raw_user_id = user_id or os.getenv("SPONSORBLOCK_USER_ID") or self._generate_user_id()
        
        # Ensure it's a 64-char hex string (SHA256 hash of a secret string)
        # as required by SponsorBlock API.
        if len(raw_user_id) != 64:
            self.user_id = hashlib.sha256(raw_user_id.encode()).hexdigest()
            logger.info(f"Using hashed SponsorBlock userID: {self.user_id} (derived from {raw_user_id[:8]}...)")
        else:
            self.user_id = raw_user_id

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
                       category: str = "intro", action_type: str = "skip",
                       video_duration: float = None) -> bool:
        
        # Guard: Check if a similar segment already exists
        existing = self.get_segments(video_id)
        for s in existing:
            if s.get("category") == category:
                seg = s.get("segment", [0, 0])
                if abs(seg[0] - start_time) < 2.5 and abs(seg[1] - end_time) < 2.5:
                    logger.info(f"Similar {category} segment already exists. Skipping.")
                    return True

        url = f"{self.base_url}/skipSegments"
        
        # Rounding to exactly 3 decimal places as per best practices
        payload = {
            "videoID": video_id,
            "segments": [{
                "segment": [round(start_time, 3), round(end_time, 3)],
                "category": category,
                "actionType": action_type,
            }],
            "userID": self.user_id,
            "userAgent": self.USER_AGENT,
        }
        
        if video_duration:
            payload["videoDuration"] = float(video_duration)

        try:
            logger.info(f"Submitting {category} segment for {video_id} ({start_time:.2f}s - {end_time:.2f}s)")
            response = self._request_with_retry("POST", url, json=payload)
            
            if response.status_code == 200:
                logger.info(f"Segment submitted successfully! UUID: {response.text[:100]}")
                return True
            elif response.status_code == 409:
                logger.info(f"Duplicate/Conflict on server for {video_id}")
                return True
            elif response.status_code == 403:
                logger.error(f"Submission forbidden. Your ID may be blocked: {response.text}")
                return False
            else:
                logger.error(f"Failed to submit: {response.status_code} - {response.text}")
                return False
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            return False

    def get_segments(self, video_id: str):
        url = f"{self.base_url}/skipSegments"
        # The API expects a JSON array for categories
        params = {"videoID": video_id, "categories": '["intro"]'}

        try:
            response = self._request_with_retry("GET", url, params=params)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Found {len(data)} existing intro segment(s) for video {video_id}")
                return data
            elif response.status_code == 404:
                return []
            else:
                logger.warning(f"Unexpected response when checking segments: {response.status_code}")
                return []
        except requests.RequestException as e:
            logger.error(f"API request failed after retries: {e}")
            return []

    def delete_segment(self, segment_uuid: str) -> bool:
        # base_url already includes /api
        url = f"{self.base_url}/skipSegments/{segment_uuid}"
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
