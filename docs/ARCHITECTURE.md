# Architecture: how Drishti AI works

This document explains every module, what it does, and why it was designed that way. For known problems in these modules, see [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Overview

```
                ┌──────────────┐
  camera ──────►│   main.py    │  read frame → resize to 640 px wide
                └──────┬───────┘  (camera lost → speak alert, retry 10 s)
                       │ frame
                       ▼
                ┌──────────────┐
                │ detector.py  │  YOLOv8 + ByteTrack → [Detection(label, conf, box, track_id)]
                └──────┬───────┘
                       │ detections
                       ▼
                ┌──────────────┐
                │ distance.py  │  Pinhole or Depth → [metres | None]
                └──────┬───────┘
                       ▼
                ┌──────────────┐
                │ tracking.py  │  smoothed distance, approach speed, seconds tracked
                └──────┬───────┘
                       │ + direction_of() from guide.py
                       ▼
       [Obstacle(label, conf, box, distance_m, direction, track_id, approach_mps, seen_s)]
                       │
                       ├──────────────────────────────┐
                       ▼                              ▼
         ┌───────────────────────────┐        ┌──────────────┐
         │ guide.py  AnnounceFilter  │        │  main.draw() │  red / green / grey boxes,
         │ (real obstacle? sure?     │        └──────────────┘  zones, FPS
         │  seen long enough?)       │
         └─────────────┬─────────────┘
                       ▼
         ┌───────────────────────────┐
         │ guide.py  Announcer       │  what to say, and is it urgent?
         └─────────────┬─────────────┘
                       ▼
         ┌───────────────────────────┐
         │ speech.py  Speaker        │  urgent / normal slots, beep, interrupt
         └───────────────────────────┘

   Key C ──► currency.py CurrencyClassifier ──► "500 rupees note" ──► speech.py
```

Everything runs in one loop on the main thread, except speech, which has its own thread so the video never freezes while a sentence is spoken.

## Modules

### `config.py`: settings
All tunable values live in one place: camera, detection, distance, which classes are spoken, voice guidance, tracking and currency. **Change behaviour here, not in the code.** The most important values:

| Setting | Value | Meaning |
|---|---|---|
| `FOCAL_LENGTH_PX` | 600 (placeholder) | from `calibrate.py`, per camera |
| `CAMERA_HEIGHT_M` | 1.3 | camera height above the floor (chest-mounted) |
| `CONF_THRESHOLD` | 0.45 | minimum confidence to detect and track |
| `ANNOUNCE_MIN_CONF` | 0.55 | minimum confidence to speak |
| `CONFIRM_S` / `CONFIRM_DANGER_S` | 0.5 s / 0.2 s | time tracked before speaking |
| `DANGER_DISTANCE_M` | 1.0 | "Careful!", red box, urgent speech |
| `ANNOUNCE_COOLDOWN_S` | 10 | repeat a still-present object after this |
| `MAX_ANNOUNCEMENTS` | 2 | objects per sentence |
| `APPROACH_SPEED_MPS` / `TIME_TO_CONTACT_S` | 1.5 / 3.0 | when something counts as "approaching" |
| `CAMERA_RETRY_S` | 10 | how long to retry a lost camera |

### `device.py`: hardware selection
Returns `cuda` (NVIDIA), `mps` (Apple Silicon) or `cpu`.

### `detector.py`: detection + tracking
- Uses Ultralytics `YOLO("yolov8n.pt")`, the nano model trained on the 80 COCO classes.
- Calls `model.track(..., tracker="bytetrack.yaml", persist=True)`. ByteTrack links boxes across frames, so the same chair keeps the same `track_id`.
- Returns `Detection(label, conf, box=(x1, y1, x2, y2), track_id)`. `track_id` is `None` until the tracker confirms the object.
- **Limitations:** COCO has no stairs, potholes, drains, poles, walls, doors or auto-rickshaws. The nano model sometimes mislabels objects (a purse as "book" or "laptop").

### `distance.py`: how far is it?

**1. `PinholeEstimator` (default)**

The pinhole camera model:

```
distance (m) = real_height (m) × focal_length (px) / box_height (px)
```

`real_height` comes from `KNOWN_HEIGHTS_M` and `focal_length` from `calibrate.py`. On top of the basic formula there are four corrections:

| Case | What we do |
|---|---|
| Box cut off at top **and** bottom (fills the view) | Treat as close: `0.8 × DANGER_DISTANCE_M` |
| Box cut off at **one** edge (within 2% of the frame) | Measure from the edge that is still visible (see the ground-plane formula below) and keep the smaller distance |
| Class has a known **width** (person: 0.45 m shoulders) and the box isn't cut off at the sides | `width × f / box_width`, and keep the smaller distance. This works for a seated person, whose height is wrong |
| Class has **no known height** | Measure from where it touches the floor. If its base is below the frame, report the nearest visible floor distance, or "close" if the box fills ≥ 80% of the frame height. If its base is above the horizon (on a table or shelf), return `None` |

**Ground-plane formula:** with a level camera at height `CAMERA_HEIGHT_M`, a point `h` metres above the floor that appears `dy` pixels below the image centre is

```
distance = (CAMERA_HEIGHT_M − h) × focal / dy
```

For feet on the floor, `h = 0`. For the top of an object, `h` = its known height.

Wherever two estimates exist, we keep the **smaller** one. If we must be wrong, "closer than it is" is the safe mistake.

**2. `DepthEstimator` (`--depth`)**
- Runs `Depth-Anything-V2-Metric-Indoor-Small-hf`, a neural network that predicts depth in metres for every pixel.
- For each box, it takes the **median depth of the central 50%** of the box, which ignores the background around the edges.
- **Pros:** works for any object. **Cons:** much slower, and trained on *indoor* scenes.

### `tracking.py`: memory per object
`DistanceSmoother` keeps one record per `track_id`:
- **Smoothed distance:** an exponential moving average (`DISTANCE_SMOOTHING = 0.4`, the weight of the newest frame).
- **Approach speed** (m/s, positive = getting closer), from the change in smoothed distance.
- **`seen_for()`:** how long the object has been tracked, used to ignore one-frame ghost detections.
- Tracks not seen for 2 s are forgotten.

### `guide.py`: what to say

**`direction_of(box, width)`** splits the frame into thirds by box centre: `"on your left"`, `"ahead"` or `"on your right"`.

**`AnnounceFilter`** decides which obstacles are worth speaking at all:
1. The class is in `ANNOUNCE_CLASSES`: person, vehicles, animals, furniture, fire hydrant, and so on. Phones, cups, books and laptops are drawn but never spoken.
2. Confidence ≥ `ANNOUNCE_MIN_CONF` (0.55).
3. Tracked for ≥ `CONFIRM_S` (0.5 s), or ≥ `CONFIRM_DANGER_S` (0.2 s) when it is already closer than 1 m.

**`Announcer.update()`** runs every frame on the filtered obstacles:
1. Ranks **approaching** objects first (speed ≥ 1.5 m/s and contact within 3 s), then the nearest. It keeps the top 2.
2. Each object is keyed by `track_id`, so it isn't announced again just for moving from "ahead" to "on your left".
3. An object is spoken when it is new, then again every 10 s while it is still there (*"chair still ahead, 2.0 meters"*). It is spoken sooner if it comes 0.5 m closer inside the danger zone, or starts approaching.
4. Messages:
   - `"chair, 2.0 meters ahead"`: new object.
   - `"chair still ahead, 2.0 meters"`: repeat for a tracked object.
   - `"Careful! chair, 0.8 meters ahead"`: closer than 1 m.
   - `"Careful! car approaching on your right, 3.0 meters"`: coming at the user.
5. The sentence is sent as **urgent** if any part of it is "Careful!".

### `speech.py`: text to speech
- A background thread with **two slots, urgent and normal.** Each slot holds only the newest message, so speech never lags behind reality.
- **Urgent messages** are spoken before normal ones, can never be replaced by a normal message, **interrupt** a normal message that is already playing, and play an **instant beep** (macOS `Funk.aiff`; Windows `MessageBeep`; terminal bell elsewhere).
- An urgent message never interrupts another urgent one.
- `wait_until_quiet()` lets the app finish its last sentence before it exits.
- macOS uses the built-in `say` command, because pyttsx3 is unreliable outside the main thread on macOS. Windows and Linux use `pyttsx3`, where interrupting uses `engine.stop()` and is best effort.
- `--mute` prints messages instead of speaking them (`[voice]` / `[voice URGENT]`).

### `main.py`: the live loop
1. Opens the camera. If it can't, it says *"Camera not found. Drishti cannot start."* and exits.
2. Each frame: read → resize → detect + track → distance → smooth → filter → announce → draw.
3. **If the camera stops giving frames,** it says *"Camera lost. Guidance stopped."* and retries for `CAMERA_RETRY_S`. If the camera comes back: *"Camera back. Guidance resumed."*
4. **However the app ends** (Q, video finished, camera gone, crash), it speaks why, because silence would sound like a clear path.
5. **Drawing:** red = danger, green = will be spoken, grey = detected but not spoken. Labels show `#id label conf% distance`.
6. Keys: **C** classifies the current frame as a currency note, **Q** quits.

## Currency pipeline

```
collect_currency.py  ──►  data/raw/<label>/*.jpg
split_dataset.py     ──►  data/currency/{train,val}/<label>/*.jpg   (80/20, seed 42)
train_currency.py    ──►  models/currency.pt  + classification report + confusion matrix
main.py (key C)      ──►  CurrencyClassifier.predict(frame) → (label, confidence)
```

- **Classes:** `10, 20, 50, 100, 200, 500, background`. The `background` class lets the model say "no note in view".
- **Models (transfer learning from ImageNet):** MobileNetV3-Large (fast, for a phone), EfficientNet-B0 (balanced), ResNet50 (strong baseline).
- **Training augmentation:** random resized crop, ±20° rotation, colour jitter and perspective. There is **no horizontal flip**, because a mirrored note never appears in real life.
- **Training setup:** AdamW, cosine learning-rate schedule, label smoothing 0.1, and automatic mixed precision on CUDA. The best epoch by validation accuracy is saved.
- **At inference:** if confidence < 0.60, or if the predicted class is `background`, the app says *"Not sure. Hold the note closer to the camera."*

## Support tools

| Script | Purpose |
|---|---|
| `calibrate.py` | Stand at a known distance, press SPACE to sample `focal = box_h × distance / height` (on the same 640 px frame as `main.py`), and get the median |
| `collect_currency.py` | SPACE saves one photo, B toggles burst mode (saves every 5th frame) |
| `split_dataset.py` | Random per-class train/val split with copies of the files |
