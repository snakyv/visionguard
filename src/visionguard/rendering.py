from __future__ import annotations

import cv2
import numpy as np

from visionguard.domain import LineRule, Track, ZoneRule


def render_frame(frame: np.ndarray, tracks: list[Track], zones: list[ZoneRule], lines: list[LineRule], fps: float) -> np.ndarray:
    output = frame.copy()
    for zone in zones:
        points = np.array(zone.points, dtype=np.int32)
        cv2.polylines(output, [points], True, (40, 210, 255), 2, cv2.LINE_AA)
        if zone.points:
            cv2.putText(output, zone.name, tuple(map(int, zone.points[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 210, 255), 2, cv2.LINE_AA)
    for line in lines:
        cv2.line(output, tuple(map(int, line.start)), tuple(map(int, line.end)), (230, 200, 40), 2, cv2.LINE_AA)
        cv2.putText(output, line.name, tuple(map(int, line.start)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 200, 40), 2, cv2.LINE_AA)
    for track in tracks:
        x1, y1, x2, y2 = (int(v) for v in track.detection.xyxy)
        cv2.rectangle(output, (x1, y1), (x2, y2), (80, 240, 120), 2)
        label = f"{track.detection.class_name} #{track.track_id} {track.detection.confidence:.2f}"
        cv2.putText(output, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 240, 120), 2, cv2.LINE_AA)
        if len(track.history) > 1:
            history = np.array(track.history, dtype=np.int32)
            cv2.polylines(output, [history], False, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(output, (10, 10), (155, 42), (15, 15, 18), -1)
    cv2.putText(output, f"FPS {fps:5.1f}", (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 2, cv2.LINE_AA)
    return output
