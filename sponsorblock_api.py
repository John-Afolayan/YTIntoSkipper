# sponsorblock_api.py
import json
import logging
import os
import requests
from dotenv import load_dotenv

# Load .env if it exists
load_dotenv()

from logger import logger

class SponsorBlockAPI:
    """Interface to SponsorBlock API"""

    def __init__(self, user_id: str = None):
        self.base_url = "https://sponsor.ajay.app/api"
        self.user_id = user_id or os.getenv("SPONSORBLOCK_USER_ID") or self._generate_user_id()

    def _generate_user_id(self):
        import uuid
        return str(uuid.uuid4())

    def submit_segment(self, video_id: str, start_time: float, end_time: float,
                       category: str = "intro", action_type: str = "skip") -> bool:
        url = f"{self.base_url}/skipSegments"

        data = {
            "videoID": video_id,
            "segments": [{
                "segment": [start_time, end_time],
                "category": category,
                "actionType": action_type
            }],
            "userID": self.user_id
        }

        try:
            logger.info(f"Submitting segment to SponsorBlock: {start_time:.2f}s - {end_time:.2f}s")
            response = requests.post(url, json=data)

            if response.status_code == 200:
                logger.info("Segment submitted successfully!")
                return True
            else:
                logger.error(f"Failed to submit segment: {response.status_code} - {response.text}")
                return False

        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            return False

    def get_segments(self, video_id: str):
        url = f"{self.base_url}/skipSegments"
        params = {"videoID": video_id, "categories": json.dumps(["intro"]) }

        try:
            response = requests.get(url, params=params)
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
            logger.error(f"API request failed: {e}")
            return []

    def delete_segment(self, segment_uuid: str) -> bool:
        url = f"{self.base_url}/api/skipSegments/{segment_uuid}"
        data = {"userID": self.user_id}

        try:
            logger.info(f"Deleting segment: {segment_uuid}")
            response = requests.delete(url, json=data)

            if response.status_code == 200:
                logger.info("Segment deleted successfully!")
                return True
            else:
                logger.error(f"Failed to delete segment: {response.status_code} - {response.text}")
                return False

        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            return False
        