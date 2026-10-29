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


def direction_of(box, frame_width: int) -> str:
    cx = (box[0] + box[2]) / 2
    if cx < frame_width / 3:
        return "on your left"
    if cx > 2 * frame_width / 3:
        return "on your right"
    return "ahead"


def describe(o: Obstacle, danger_m: float) -> str:
    msg = f"{o.label}, {o.distance_m:.1f} meters {o.direction}"
    return f"Careful! {msg}" if o.distance_m < danger_m else msg


class Announcer:
    """Decides what is worth saying out loud, so the user isn't flooded with speech."""

    def __init__(self, speaker: Speaker, cooldown_s: float, danger_m: float, max_items: int):
        self.speaker = speaker
        self.cooldown_s = cooldown_s
        self.danger_m = danger_m
        self.max_items = max_items
        self._last: dict[tuple[str, str], tuple[float, float]] = {}  # key -> (time, distance)

    def update(self, obstacles: list[Obstacle]):
        now = time.monotonic()
        messages = []
        for o in sorted(obstacles, key=lambda o: o.distance_m)[:self.max_items]:
            key = (o.label, o.direction)
            prev = self._last.get(key)
            if prev:
                recent = now - prev[0] < self.cooldown_s
                closing_in = o.distance_m < self.danger_m and o.distance_m < prev[1] - 0.5
                if recent and not closing_in:
                    continue
            self._last[key] = (now, o.distance_m)
            messages.append(describe(o, self.danger_m))
        if messages:
            self.speaker.say(". ".join(messages))
