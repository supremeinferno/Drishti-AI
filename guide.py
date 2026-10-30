from __future__ import annotations

import time
from dataclasses import dataclass

from speech import Speaker


@dataclass
class Obstacle:
    label: str
    conf: float
    box: tuple[int, int, int, int]
    distance_m: float
    direction: str
    track_id: int | None = None
    approach_mps: float = 0.0  # positive = getting closer


def direction_of(box, frame_width: int) -> str:
    cx = (box[0] + box[2]) / 2
    if cx < frame_width / 3:
        return "on your left"
    if cx > 2 * frame_width / 3:
        return "on your right"
    return "ahead"


def is_approaching(o: Obstacle, min_speed_mps: float, time_to_contact_s: float) -> bool:
    """Coming at the user faster than walking pace and will reach them soon."""
    return o.approach_mps >= min_speed_mps and o.distance_m / o.approach_mps < time_to_contact_s


def describe(o: Obstacle, danger_m: float, approaching: bool, still_there: bool = False) -> str:
    if approaching:
        return f"Careful! {o.label} approaching {o.direction}, {o.distance_m:.1f} meters"
    if still_there:
        msg = f"{o.label} still {o.direction}, {o.distance_m:.1f} meters"
    else:
        msg = f"{o.label}, {o.distance_m:.1f} meters {o.direction}"
    return f"Careful! {msg}" if o.distance_m < danger_m else msg


class Announcer:
    """Decides what is worth saying out loud, so the user isn't flooded with speech.

    Each object is announced once, then repeated every cooldown_s while it is still there.
    It is repeated sooner if it gets dangerously close or starts coming at the user.
    """

    def __init__(self, speaker: Speaker, cooldown_s: float, danger_m: float, max_items: int,
                 approach_mps: float, time_to_contact_s: float):
        self.speaker = speaker
        self.cooldown_s = cooldown_s
        self.danger_m = danger_m
        self.max_items = max_items
        self.approach_mps = approach_mps
        self.time_to_contact_s = time_to_contact_s
        # object key -> (time, distance, approaching) when it was last announced
        self._last: dict = {}

    def update(self, obstacles: list[Obstacle]):
        now = time.monotonic()
        # An object still in view is re-announced every cooldown_s, which refreshes its entry.
        # One missing for two cooldowns has gone, so it counts as new if it comes back.
        self._last = {k: v for k, v in self._last.items() if now - v[0] < 2 * self.cooldown_s}
        approaching = {id(o): is_approaching(o, self.approach_mps, self.time_to_contact_s)
                       for o in obstacles}
        # Anything coming at the user goes first, then the nearest.
        ranked = sorted(obstacles, key=lambda o: (not approaching[id(o)], o.distance_m))
        messages, urgent = [], False
        for o in ranked[:self.max_items]:
            # With a track ID the same chair stays the same chair when it moves from "ahead"
            # to "on your left", so it isn't announced again just for crossing a zone line.
            key = o.track_id if o.track_id is not None else (o.label, o.direction)
            prev = self._last.get(key)
            if prev:
                recent = now - prev[0] < self.cooldown_s
                closing_in = o.distance_m < self.danger_m and o.distance_m < prev[1] - 0.5
                started_approaching = approaching[id(o)] and not prev[2]
                if recent and not closing_in and not started_approaching:
                    continue
            self._last[key] = (now, o.distance_m, approaching[id(o)])
            # "still" only when the tracker confirms it is the same object as last time.
            still_there = prev is not None and o.track_id is not None
            messages.append(describe(o, self.danger_m, approaching[id(o)], still_there))
            # Same rule as the "Careful!" prefix in describe(): close, or coming at the user.
            urgent |= approaching[id(o)] or o.distance_m < self.danger_m
        if messages:
            self.speaker.say(". ".join(messages), urgent=urgent)
