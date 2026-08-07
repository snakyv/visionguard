from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot

from visionguard.domain import Detection, Track
from visionguard.tracking.base import Tracker


@dataclass(slots=True)
class _State:
    track_id: int
    detection: Detection
    missed: int = 0
    history: list[tuple[float, float]] = field(default_factory=list)


class CentroidTracker(Tracker):
    def __init__(self, max_distance: float = 90.0, max_missed: int = 12, history_size: int = 40) -> None:
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.history_size = history_size
        self._next_id = 1
        self._states: dict[int, _State] = {}

    def update(self, detections: list[Detection]) -> list[Track]:
        unmatched = set(range(len(detections)))
        assignments: dict[int, int] = {}
        candidates: list[tuple[float, int, int]] = []

        for track_id, state in self._states.items():
            sx, sy = state.detection.center
            for idx, detection in enumerate(detections):
                if detection.class_id != state.detection.class_id:
                    continue
                dx, dy = detection.center
                distance = hypot(dx - sx, dy - sy)
                if distance <= self.max_distance:
                    candidates.append((distance, track_id, idx))

        used_tracks: set[int] = set()
        for _, track_id, idx in sorted(candidates):
            if track_id in used_tracks or idx not in unmatched:
                continue
            assignments[track_id] = idx
            used_tracks.add(track_id)
            unmatched.remove(idx)

        for track_id, state in list(self._states.items()):
            idx = assignments.get(track_id)
            if idx is None:
                state.missed += 1
                if state.missed > self.max_missed:
                    del self._states[track_id]
                continue
            state.detection = detections[idx]
            state.missed = 0
            state.history.append(state.detection.center)
            state.history = state.history[-self.history_size :]

        for idx in sorted(unmatched):
            detection = detections[idx]
            state = _State(self._next_id, detection, history=[detection.center])
            self._states[self._next_id] = state
            self._next_id += 1

        return [
            Track(track_id=s.track_id, detection=s.detection, history=list(s.history))
            for s in self._states.values()
            if s.missed == 0
        ]
