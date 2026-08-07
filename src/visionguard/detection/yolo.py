from __future__ import annotations

import numpy as np

from visionguard.detection.base import Detector
from visionguard.domain import Detection


class YoloDetector(Detector):
    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        confidence: float = 0.35,
        iou: float = 0.6,
        imgsz: int = 640,
        device: str | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install VisionGuard with the cv extra to use YOLO") from exc
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self.device = device

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        names = result.names
        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is None:
            return detections
        for xyxy, confidence, class_id in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy(),
            strict=False,
        ):
            cid = int(class_id)
            detections.append(
                Detection(
                    xyxy=tuple(float(value) for value in xyxy),
                    confidence=float(confidence),
                    class_id=cid,
                    class_name=str(names[cid]),
                )
            )
        return detections
