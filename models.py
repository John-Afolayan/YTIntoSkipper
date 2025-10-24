# models.py
from dataclasses import dataclass


@dataclass
class IntroSegment:
    """Represents a detected intro segment"""
    start_time: float
    end_time: float
    confidence: float
    video_id: str = ""
    