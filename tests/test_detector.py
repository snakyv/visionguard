import cv2
import numpy as np

from visionguard.detection.synthetic import SyntheticDetector


def test_synthetic_detector_finds_three_classes():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(frame, (20, 40), (55, 130), (40, 70, 230), -1)
    cv2.rectangle(frame, (100, 80), (190, 140), (20, 170, 240), -1)
    cv2.rectangle(frame, (210, 150), (300, 205), (220, 120, 60), -1)
    detections = SyntheticDetector().detect(frame)
    assert {d.class_name for d in detections} == {"person", "forklift", "car"}
