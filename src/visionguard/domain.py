from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    ZONE_ENTER = "zone_enter"
    ZONE_EXIT = "zone_exit"
    DWELL = "dwell"
    LINE_CROSS = "line_cross"
    NEAR_MISS = "near_miss"
    OBJECT_COUNT = "object_count"


@dataclass(slots=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(slots=True)
class Track:
    track_id: int
    detection: Detection
    history: list[tuple[float, float]] = field(default_factory=list)

    @property
    def center(self) -> tuple[float, float]:
        return self.detection.center


@dataclass(slots=True)
class AnalyticsEvent:
    event_type: EventType
    camera_id: str
    track_id: int | None
    class_name: str | None
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "class_name": self.class_name,
            "message": self.message,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(slots=True)
class ZoneRule:
    name: str
    points: list[tuple[float, float]]
    classes: set[str]
    dwell_seconds: float = 0.0


@dataclass(slots=True)
class LineRule:
    name: str
    start: tuple[float, float]
    end: tuple[float, float]
    classes: set[str]
