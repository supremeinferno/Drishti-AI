"""Two ways to estimate how far an object is.

PinholeEstimator  - fast, no extra model, uses known object heights + camera focal length.
DepthEstimator    - neural monocular depth (Depth Anything V2), slower but works for any object.
"""
import cv2
import numpy as np

from detector import Detection


class PinholeEstimator:
    def __init__(self, focal_px: float, known_heights: dict[str, float], danger_m: float):
        self.focal_px = focal_px
        self.known_heights = known_heights
        self.danger_m = danger_m

    def estimate(self, frame, detections: list[Detection]) -> list[float | None]:
        frame_h = frame.shape[0]
        distances = []
        for d in detections:
            real_h = self.known_heights.get(d.label)
            x1, y1, x2, y2 = d.box
            pixel_h = y2 - y1
            if real_h is None or pixel_h <= 0:
                distances.append(None)
                continue
            dist = real_h * self.focal_px / pixel_h
            # A box cut off at both the top and bottom edge means the object fills the view,
            # so the pixel height underestimates its size. Treat it as close.
            if y1 <= 2 and y2 >= frame_h - 2:
                dist = min(dist, self.danger_m * 0.8)
            distances.append(dist)
        return distances


class DepthEstimator:
    def __init__(self, model_id: str):
        from PIL import Image
        from transformers import pipeline

        from device import pick_device

        self._to_pil = Image.fromarray
        self.pipe = pipeline("depth-estimation", model=model_id, device=pick_device())

    def estimate(self, frame, detections: list[Detection]) -> list[float | None]:
        if not detections:
            return []
        h, w = frame.shape[:2]
        out = self.pipe(self._to_pil(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        depth = out["predicted_depth"].squeeze().float().cpu().numpy()
        depth = cv2.resize(depth, (w, h))

        distances = []
        for d in detections:
            x1, y1, x2, y2 = d.box
            # Use the central part of the box so background pixels at the edges don't skew it.
            cx1, cx2 = x1 + (x2 - x1) // 4, x2 - (x2 - x1) // 4
            cy1, cy2 = y1 + (y2 - y1) // 4, y2 - (y2 - y1) // 4
            region = depth[max(cy1, 0):cy2, max(cx1, 0):cx2]
            distances.append(float(np.median(region)) if region.size else None)
        return distances
