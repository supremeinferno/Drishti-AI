from __future__ import annotations

from dataclasses import dataclass

from ultralytics import YOLO

from device import pick_device


@dataclass
class Detection:
    label: str
    conf: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    track_id: int | None = None     # same number for the same object across frames


class Detector:
    def __init__(self, weights: str, conf: float, tracker: str = "bytetrack.yaml"):
        self.model = YOLO(weights)
        self.conf = conf
        self.tracker = tracker
        self.device = pick_device()

    def __call__(self, frame) -> list[Detection]:
        # track() runs the same detector, then ByteTrack links boxes across frames.
        result = self.model.track(frame, persist=True, tracker=self.tracker, conf=self.conf,
                                  device=self.device, verbose=False)[0]
        boxes = result.boxes
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
        detections = []
        for xyxy, conf, cls, track_id in zip(boxes.xyxy.tolist(), boxes.conf.tolist(),
                                             boxes.cls.tolist(), ids):
            x1, y1, x2, y2 = map(int, xyxy)
            detections.append(Detection(result.names[int(cls)], conf, (x1, y1, x2, y2), track_id))
        return detections
