from dataclasses import dataclass

from ultralytics import YOLO

from device import pick_device


@dataclass
class Detection:
    label: str
    conf: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2


class Detector:
    def __init__(self, weights: str, conf: float):
        self.model = YOLO(weights)
        self.conf = conf
        self.device = pick_device()

    def __call__(self, frame) -> list[Detection]:
        result = self.model.predict(frame, conf=self.conf, device=self.device, verbose=False)[0]
        detections = []
        for xyxy, conf, cls in zip(result.boxes.xyxy.tolist(),
                                   result.boxes.conf.tolist(),
                                   result.boxes.cls.tolist()):
            x1, y1, x2, y2 = map(int, xyxy)
            detections.append(Detection(result.names[int(cls)], conf, (x1, y1, x2, y2)))
        return detections
