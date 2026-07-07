# models.py
from dataclasses import dataclass


@dataclass
class IntroSegment:
    """Represents a detected intro segment"""
    start_time: float
    end_time: float
    confidence: float
    video_id: str = ""
    video_duration: float = 0.0
    # True when speech was detected over the START of the matched intro
    # (while the reference is instrumental there) — submitting this skip
    # would cut the creator's talking, so it must be manually reviewed.
    talkover_warning: bool = False

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def __str__(self) -> str:
        return (
            f"Intro({self.start_time:.2f}s-{self.end_time:.2f}s, "
            f"dur={self.duration:.2f}s, conf={self.confidence:.2f})"
        )
    