from __future__ import annotations

import cv2
import numpy as np

from visionguard.detection.base import Detector
from visionguard.domain import Detection


class SyntheticDetector(Detector):
    def __init__(self, min_area: int = 250) -> None:
        self.min_area = min_area
        self.specs = [
            (0, "person", np.array([0, 0, 180]), np.array([90, 130, 255])),
            (1, "forklift", np.array([0, 120, 180]), np.array([80, 230, 255])),
            (2, "car", np.array([150, 60, 0]), np.array([255, 190, 130])),
        ]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        detections: list[Detection] = []
        for class_id, class_name, lower, upper in self.specs:
            mask = cv2.inRange(frame, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                detections.append(
                    Detection(
                        xyxy=(float(x), float(y), float(x + w), float(y + h)),
                        confidence=0.99,
                        class_id=class_id,
                        class_name=class_name,
                    )
                )
        return detections
