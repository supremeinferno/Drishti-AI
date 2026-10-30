"""Two ways to estimate how far an object is.

PinholeEstimator  - fast, no extra model, uses known object heights + camera focal length.
DepthEstimator    - neural monocular depth (Depth Anything V2), slower but works for any object.
"""
from __future__ import annotations

import cv2
import numpy as np

from detector import Detection


EDGE_FRACTION = 0.02  # a box within 2% of the frame edge is cut off by it (YOLO stops a few px short)


class PinholeEstimator:
    """distance = real_height * focal / pixel_height, corrected for boxes cut off by the frame.

    A cut-off box is shorter than the object, so the plain formula says it is FARTHER than it
    really is (a chair 2 m away with its legs below the frame comes out as 9 m). For those boxes
    we use the edge that is still visible. With a level camera held at camera_height_m, a point
    h metres above the floor that appears `dy` pixels below the image centre is
    (camera_height_m - h) * focal / dy metres away:
      - feet visible (h = 0): measure where the object stands on the floor
      - top visible (h = known height): measure from the object's top edge
    """

    MIN_HEIGHT_GAP_M = 0.3  # top edge level with the camera says nothing about distance
    FILLS_VIEW = 0.8        # a cut-off box this tall (fraction of the frame) is right in front

    def __init__(self, focal_px: float, known_heights: dict[str, float], danger_m: float,
                 camera_height_m: float, known_widths: dict[str, float] | None = None):
        self.focal_px = focal_px
        self.known_heights = known_heights
        self.known_widths = known_widths or {}
        self.danger_m = danger_m
        self.camera_height_m = camera_height_m

    def distance_from_edge(self, y: float, height_m: float, frame_h: int) -> float | None:
        """Distance to a point height_m above the floor seen at image row y."""
        drop_m = self.camera_height_m - height_m  # how far below the camera the point is
        dy = y - frame_h / 2                      # pixels below the horizon (image centre)
        if abs(drop_m) < self.MIN_HEIGHT_GAP_M or abs(dy) < 1 or (drop_m > 0) != (dy > 0):
            return None
        return drop_m * self.focal_px / dy

    def unknown_height_distance(self, y2: int, bottom_cut: bool, pixel_h: int,
                                frame_h: int) -> float | None:
        """Distance for a class with no known height (a cow, a fire hydrant...).

        Dropping it would hide a real obstacle, so measure where it stands on the floor instead.
        If its base is below the frame it is nearer than the closest floor the camera can see,
        so report that distance as an upper bound, or "close" if it fills most of the view.
        """
        if not bottom_cut:
            return self.distance_from_edge(y2, 0.0, frame_h)
        if pixel_h >= self.FILLS_VIEW * frame_h:
            return self.danger_m * 0.8
        return self.distance_from_edge(frame_h, 0.0, frame_h)

    def estimate(self, frame, detections: list[Detection]) -> list[float | None]:
        frame_h, frame_w = frame.shape[:2]
        edge_px = EDGE_FRACTION * frame_h
        distances = []
        for d in detections:
            real_h = self.known_heights.get(d.label)
            x1, y1, x2, y2 = d.box
            pixel_h = y2 - y1
            if pixel_h <= 0:
                distances.append(None)
                continue
            top_cut, bottom_cut = y1 <= edge_px, y2 >= frame_h - edge_px
            if real_h is None:
                distances.append(self.unknown_height_distance(y2, bottom_cut, pixel_h, frame_h))
                continue
            dist = real_h * self.focal_px / pixel_h
            if top_cut and bottom_cut:
                # The object fills the view from top to bottom. Treat it as close.
                dist = min(dist, self.danger_m * 0.8)
            elif top_cut or bottom_cut:
                # Measure from whichever edge is still in view: the feet, or the top.
                edge = self.distance_from_edge(y2, 0.0, frame_h) if top_cut \
                    else self.distance_from_edge(y1, real_h, frame_h)
                if edge is not None:
                    dist = min(dist, edge)
            # Width doesn't change when someone sits down or is cut off at the top or bottom.
            # Taking the smaller estimate errs on the side of "closer", the safe mistake.
            real_w = self.known_widths.get(d.label)
            side_cut = x1 <= EDGE_FRACTION * frame_w or x2 >= frame_w * (1 - EDGE_FRACTION)
            if real_w and x2 > x1 and not side_cut:
                dist = min(dist, real_w * self.focal_px / (x2 - x1))
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
