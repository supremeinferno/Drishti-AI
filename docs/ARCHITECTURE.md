# Architecture: how Drishti AI works

This document explains every module, what it does, and why it was designed that way. For the bugs in these modules, see [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Overview

```
                ┌──────────────┐
  camera ──────►│   main.py    │  read frame → resize to 640 px wide
                └──────┬───────┘
                       │ frame
                       ▼
                ┌──────────────┐
                │ detector.py  │  YOLOv8 → [Detection(label, conf, box)]
                └──────┬───────┘
                       │ detections
                       ▼
                ┌──────────────┐
                │ distance.py  │  Pinhole or Depth → [metres | None]
                └──────┬───────┘
                       │ + direction_of() from guide.py
                       ▼
                [Obstacle(label, conf, box, distance_m, direction)]
                       │
           ┌───────────┴────────────┐
           ▼                        ▼
   ┌──────────────┐         ┌──────────────┐
   │  guide.py    │         │  main.draw() │  boxes, zones, FPS on screen
   │  Announcer   │         └──────────────┘
   └──────┬───────┘
          │ text
          ▼
   ┌──────────────┐
   │  speech.py   │  background thread, latest message only
   └──────────────┘

   Key C ──► currency.py CurrencyClassifier ──► "500 rupees note" ──► speech.py
```

Everything runs in one loop on the main thread, except speech, which has its own thread so the video never freezes while a sentence is spoken.

## Modules

### `config.py`: settings
All tunable values live in one place: camera index, frame width, YOLO weights and confidence, focal length, known object heights, depth model ID, danger distance, cooldown, maximum announcements, and the currency model path and confidence. **Change behaviour here, not in the code.**

### `device.py`: hardware selection
Returns `cuda` (NVIDIA), `mps` (Apple Silicon) or `cpu`. YOLO, the depth model, the currency classifier and training all use it.

### `detector.py`: object detection
- Wraps Ultralytics `YOLO("yolov8n.pt")`, the nano model trained on the 80 COCO classes.
- Each frame returns a list of `Detection(label, conf, box=(x1, y1, x2, y2))`.
- Detections below `CONF_THRESHOLD = 0.45` are discarded.
- **Limitation:** COCO has no stairs, potholes, drains, poles, walls, doors or auto-rickshaws.

### `distance.py`: how far is it?

**1. `PinholeEstimator` (default)**

The pinhole camera model:

```
distance (m) = real_height (m) × focal_length (px) / box_height (px)
```

- `real_height` comes from `KNOWN_HEIGHTS_M` in `config.py`, for example person = 1.70 m and chair = 0.90 m.
- `focal_length` is measured once per webcam with `calibrate.py`.
- If a box touches both the top and bottom edges of the frame, the object fills the view, so it is clamped to "close" (`0.8 × DANGER_DISTANCE_M`).
- Classes without a known height return `None` and are dropped.
- **Pros:** instant, no extra model, easy to explain. **Cons:** needs the object's real height, breaks when the box is cut off or the object is partly hidden, and assumes the object is upright.

**2. `DepthEstimator` (`--depth`)**
- Runs `Depth-Anything-V2-Metric-Indoor-Small-hf`, a neural network that predicts depth in metres for every pixel.
- For each box, it takes the **median depth of the central 50%** of the box, which ignores the background around the edges.
- **Pros:** works for any object and doesn't need known heights. **Cons:** much slower, and it was trained on *indoor* scenes, so outdoor accuracy is uncertain.

### `guide.py`: what to say
- `direction_of(box, width)`: splits the frame into thirds by box centre and returns `"on your left"`, `"ahead"` or `"on your right"`.
- `describe()`: builds `"chair, 0.8 meters ahead"` and adds the prefix `"Careful!"` when the object is closer than `DANGER_DISTANCE_M` (1.0 m).
- `Announcer.update()` runs every frame:
  1. Sorts obstacles by distance and keeps the nearest `MAX_ANNOUNCEMENTS` (2).
  2. Skips any `(label, direction)` already spoken in the last `ANNOUNCE_COOLDOWN_S` (4 s)…
  3. …unless it is inside the danger zone **and** has come at least 0.5 m closer since it was last spoken.
  4. Joins the remaining messages into one sentence and sends it to the speaker.

### `speech.py`: text to speech
- A background thread with a queue of size 1. A new message **replaces** a message that hasn't been spoken yet, so speech never lags behind reality.
- macOS uses the built-in `say` command, because pyttsx3 is unreliable outside the main thread on macOS. Windows and Linux use `pyttsx3`.
- `--mute` prints messages instead of speaking them.

### `main.py`: the live loop
1. Read a frame and resize it to `FRAME_WIDTH` (640 px).
2. Detect → estimate distance → build `Obstacle`s → `announcer.update()`.
3. Draw boxes (red when inside the danger zone, green otherwise), the left/ahead/right zone lines and the FPS counter.
4. Keys: **C** classifies the current frame as a currency note, **Q** quits.

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
| `calibrate.py` | Stand at a known distance, press SPACE to sample `focal = box_h × distance / height`, and get the median |
| `collect_currency.py` | SPACE saves one photo, B toggles burst mode (saves every 5th frame) |
| `split_dataset.py` | Random per-class train/val split with copies of the files |
