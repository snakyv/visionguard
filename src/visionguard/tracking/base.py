from __future__ import annotations

from abc import ABC, abstractmethod

from visionguard.domain import Detection, Track


class Tracker(ABC):
    @abstractmethod
    def update(self, detections: list[Detection]) -> list[Track]:
        raise NotImplementedError
