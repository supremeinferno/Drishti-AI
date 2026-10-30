"""Per-object memory across frames, keyed by the tracker's ID.

A single frame's distance jumps around by tens of centimetres. Averaging it over a few frames
steadies the spoken number, and the change over time tells us how fast the object is coming.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Track:
    distance_m: float
    speed_mps: float  # positive = getting closer
    seen_at: float
    first_seen: float


class DistanceSmoother:
    def __init__(self, alpha: float, forget_s: float = 2.0):
        self.alpha = alpha        # weight of the newest frame, 0..1 (lower = smoother)
        self.forget_s = forget_s  # drop tracks not seen for this long
        self._tracks: dict[int, _Track] = {}

    def update(self, track_id: int | None, distance_m: float, now: float) -> tuple[float, float]:
        """Return (smoothed distance, approach speed in m/s) for this object."""
        if track_id is None:
            return distance_m, 0.0
        t = self._tracks.get(track_id)
        if t is None:
            self._tracks[track_id] = _Track(distance_m, 0.0, now, now)
            return distance_m, 0.0
        dt = now - t.seen_at
        smoothed = self.alpha * distance_m + (1 - self.alpha) * t.distance_m
        if dt > 0:
            speed = (t.distance_m - smoothed) / dt
            t.speed_mps = self.alpha * speed + (1 - self.alpha) * t.speed_mps
        t.distance_m, t.seen_at = smoothed, now
        return smoothed, t.speed_mps

    def seen_for(self, track_id: int | None, now: float) -> float:
        """How long this object has been tracked. 0 for an object the tracker hasn't confirmed."""
        t = self._tracks.get(track_id)
        return now - t.first_seen if t else 0.0

    def forget_old(self, now: float):
        self._tracks = {i: t for i, t in self._tracks.items() if now - t.seen_at < self.forget_s}
