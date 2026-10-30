# Evaluation plan: how we prove it works

Claims need numbers. This document defines **what** we measure, **how**, and **where the results go**. All tables are empty until we run the experiments; we never fill them with estimates.

---

## 1. Distance accuracy

**Setup**
- Calibrate first with `calibrate.py`, and set the real `CAMERA_HEIGHT_M` (issue #14).
- Mount the camera **level at chest height** (about 1.3 m), the way it will be worn, not on a desk (issue #21).
- Mark the floor at **1, 2, 3, 4 and 5 m** from the camera with a measuring tape.
- Objects: person standing, **person seated**, chair, and one floor object with no known height (for example a box or a bucket).
- For each object and distance, record about 30 frames and take the median estimate (the on-screen value, after smoothing).
- Repeat with the object partly cut off at the bottom of the frame (tests issue #1).

**Metrics:** mean absolute error (m), relative error (%), and the share of estimates within ±20%.

| Object | True (m) | Pinhole (m) | Error | Depth model (m) | Error |
|---|---|---|---|---|---|
| person, standing | 1.0 | | | | |
| person, standing | 2.0 | | | | |
| person, seated | 1.0 | | | | |
| … | | | | | |

An early check (uncalibrated, laptop on a desk, read from a screenshot): a seated person at about 0.6 m was shown as **3.4 m** before the fixes for issue #1 and **0.84 m** after. This is not a benchmark result; the table above replaces it once filled in.

## 2. Obstacle detection

**Test set:** our own labelled frames from **locations not used in training**: indoor corridor, home, footpath, market road, stairs.

**Metrics:** mAP@0.5, mAP@0.5:0.95, and precision / recall **per class**. For safety, **recall on hazards** matters most, because a missed pothole is worse than a false alarm.

| Model | mAP@0.5 | Hazard recall | FPS (laptop) | FPS (target device) | Size (MB) |
|---|---|---|---|---|---|
| YOLOv8n (COCO, baseline) | | | | | |
| YOLOv8n fine-tuned | | | | | |
| YOLO11n fine-tuned | | | | | |

## 3. Currency recognition

### Currency dataset rules
These rules keep the accuracy number honest (see issues #9–#11):

1. **Several sessions per class:** at least 3 different days, places or lighting setups.
2. **Same backgrounds for every class**, so the background can't give the answer away.
3. **Multiple physical notes per value**, and both old and new designs.
4. **Hard cases:** folded, crumpled, partly covered, low light, held at an angle.
5. **Split by session, not by image.** The test set must come from a session never seen in training.
6. **Test set used once**, only for the final report.

**Metrics:** test accuracy, macro F1, per-class precision and recall, confusion matrix, **false-confident rate** (wrong answers given with confidence ≥ 0.60), latency.

| Model | Test acc | Macro F1 | False-confident rate | Latency (ms) | Size (MB) |
|---|---|---|---|---|---|
| MobileNetV3-Large | | | | | |
| EfficientNet-B0 | | | | | |
| ResNet50 | | | | | |

## 4. Voice guidance quality

Run `python main.py --mute --source walk.mp4` on the fixed chest-height test video and save the `[voice]` / `[voice URGENT]` lines. Compare them with a hand-made list of the real obstacles in the video.

| Metric | Meaning | Result |
|---|---|---|
| Missed obstacles | real obstacle in the path, never spoken | |
| False announcements / min | spoken, but not real or not an obstacle | |
| Late warnings | "Careful!" spoken after the object was already closer than 1 m | |
| Sentences / min | how talkative the app is | |

Re-run after every change to the announce filter, cooldown or detector, so we can see whether it improved.

## 5. System speed

| Mode | Device | FPS | Detection → speech latency (ms) |
|---|---|---|---|
| Pinhole | Laptop CPU | | |
| Pinhole | Laptop GPU / MPS | | |
| Depth model | Laptop GPU / MPS | | |
| Pinhole | Phone / Pi / Jetson | | |

Latency = time from the frame an obstacle first appears to the moment the warning starts playing.

## 6. Real-world walking test

- **Course:** a corridor or footpath with about 10 placed obstacles, including at least one at head height and one step.
- **Participants:** blindfolded sighted volunteers first (for safety), then visually impaired volunteers with informed consent and a sighted spotter always present.
- **Conditions:** cane only vs cane + Drishti.
- **Metrics:** collisions, near misses, completion time, false alarms, missed warnings.
- **Record:** video of every run, with consent.

## 7. User feedback

Short interview after each session:
- Was the voice clear? Too much speech or too little?
- Was the direction understandable?
- Which language do you prefer?
- Would you carry this every day? What would stop you?
- System Usability Scale (SUS) score.

## 8. Limitations to report honestly

- The pinhole method needs known object heights (or widths) and an upright object.
- Floor-based distances assume a level camera at a known height and objects on the same floor.
- The COCO-trained detector cannot see stairs, potholes, drains or poles, and the nano model mislabels some objects.
- The depth model is slower and was trained on indoor scenes.
- Low light, rain, glare and motion blur reduce accuracy.
- A single camera has a limited field of view.
- The prototype runs on a laptop until Phase 5.
