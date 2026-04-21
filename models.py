# models.py
from dataclasses import dataclass


@dataclass
class IntroSegment:
    """Represents a detected intro segment"""
    start_time: float
    end_time: float
    confidence: float
    video_id: str = ""

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def __str__(self) -> str:
        return (
            f"Intro({self.start_time:.2f}s-{self.end_time:.2f}s, "
            f"dur={self.duration:.2f}s, conf={self.confidence:.2f})"
        )
    