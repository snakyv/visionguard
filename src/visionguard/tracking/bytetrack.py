from __future__ import annotations

import numpy as np

from visionguard.domain import Detection, Track
from visionguard.tracking.base import Tracker


class ByteTrackTracker(Tracker):
    def __init__(self, frame_rate: float = 30.0) -> None:
        try:
            import supervision as sv
            from trackers import ByteTrackTracker as RoboflowByteTrackTracker
        except ImportError as exc:
            raise RuntimeError("Install VisionGuard with the cv extra to use ByteTrack") from exc
        self.sv = sv
        self.tracker = RoboflowByteTrackTracker(frame_rate=frame_rate)
        self.histories: dict[int, list[tuple[float, float]]] = {}

    def update(self, detections: list[Detection]) -> list[Track]:
        if not detections:
            self.tracker.update(self.sv.Detections.empty())
            return []
        xyxy = np.array([d.xyxy for d in detections], dtype=np.float32)
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        class_id = np.array([d.class_id for d in detections], dtype=np.int32)
        sv_detections = self.sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        tracked = self.tracker.update(sv_detections)
        result: list[Track] = []
        if tracked.tracker_id is None:
            return result
        for idx, tracker_id in enumerate(tracked.tracker_id):
            if tracker_id is None:
                continue
            tid = int(tracker_id)
            cid = int(tracked.class_id[idx]) if tracked.class_id is not None else 0
            conf = float(tracked.confidence[idx]) if tracked.confidence is not None else 1.0
            box = tuple(float(v) for v in tracked.xyxy[idx])
            class_name = next((d.class_name for d in detections if d.class_id == cid), str(cid))
            detection = Detection(box, conf, cid, class_name)
            history = self.histories.setdefault(tid, [])
            history.append(detection.center)
            del history[:-40]
            result.append(Track(tid, detection, list(history)))
        return result
