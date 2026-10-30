# Known issues

An honest list of problems in the code, both **open** and **fixed**. Each open issue lists **what goes wrong**, **why**, **where** and **how we plan to fix it**. Issue numbers stay the same once given, so other documents can refer to them.

Severity:
- 🔴 **Safety:** can give the user wrong or missing danger information.
- 🟠 **Correctness:** wrong results or misleading numbers.
- 🟡 **Usability / engineering:** works, but not good enough for real users or a real release.

## Open

| # | Issue | Severity | Area |
|---|---|---|---|
| 7 | COCO has no Indian street hazards | 🔴 | Detection |
| 14 | Focal length and camera height still placeholders, distance accuracy never measured | 🟠 | Distance |
| 21 | Floor-based distances assume a level camera at chest height and objects on the floor | 🟠 | Distance |
| 22 | Screen and speech show different distances | 🟠 | UX |
| 23 | Small detector mislabels objects (purse → "book" / "laptop") | 🟠 | Detection |
| 8 | Currency is classified on the annotated frame | 🟠 | Currency |
| 9 | Near-duplicate photos leak between train and val | 🟠 | Currency |
| 10 | No test set: the best epoch is chosen and reported on val | 🟠 | Currency |
| 11 | The model may learn background and lighting instead of the note | 🟠 | Currency |
| 12 | Whole-frame, single-frame classification | 🟠 | Currency |
| 13 | Old / new note designs and coins not planned | 🟠 | Currency |
| 15 | Depth model is indoor-only and slow | 🟠 | Distance |
| 24 | Speech interrupt untested on Windows / Linux | 🟡 | Speech |
| 16 | Keyboard trigger (C key) is not usable by blind users | 🟡 | UX |
| 17 | Spoken distance is too precise; no steps or clock directions *(partly fixed)* | 🟡 | UX |
| 18 | English only | 🟡 | UX |
| 19 | Needs a desktop window, no headless or wearable mode | 🟡 | Deployment |
| 20 | No tests, unpinned dependencies, old YOLO version | 🟡 | Engineering |

## Fixed

| # | Issue | How it was fixed |
|---|---|---|
| 1 | Nearby objects cut off at the bottom reported as far away | Measure from the visible edge with the ground-plane formula; the "cut off" margin is now 2% of the frame instead of 2 px (YOLO stops a few px short of the edge); shoulder-width estimate for people. A seated person at ~0.6 m went from 3.4 m to 0.84 m (uncalibrated) |
| 2 | Calibration and the live app used different resolutions | `calibrate.py` resizes to `FRAME_WIDTH` like `main.py` |
| 3 | Objects without a known height dropped silently | Measured from where they touch the floor; heights added for cow, horse, sheep, elephant, fire hydrant, parking meter, train |
| 4 | Danger warnings could be overwritten or delayed | Urgent / normal speech slots; urgent beeps, interrupts normal speech and can't be replaced |
| 5 | App stopped silently when the camera failed | Spoken alerts, 10 s reconnect, and every way of stopping is announced |
| 6 | No frame-to-frame tracking or smoothing | ByteTrack IDs, smoothed distance, approach speed, "still there" repeats |
| 17 *(part)* | Announced cups, books, laptops and the phone in the user's hand; one-frame ghosts spoken | `AnnounceFilter`: only obstacle classes, ≥ 55% confidence, tracked ≥ 0.5 s (0.2 s when close). Others are drawn in grey |

---

## 🔴 Safety

### 7. COCO has no Indian street hazards
- **Where:** [config.py:8](../config.py#L8) (`yolov8n.pt`)
- **What happens:** the most dangerous things for a blind pedestrian in India, such as **stairs, potholes, open drains, poles, walls, glass doors, speed breakers, auto-rickshaws and e-rickshaws**, are not COCO classes, so they cannot be detected at all.
- **Fix:** collect and label an Indian street dataset (Roboflow / CVAT) and fine-tune YOLO. Add a **free-path / walkable-ground** check using the depth map or segmentation, so even *unknown* obstacles in front of the user trigger a warning.

---

## 🟠 Correctness

### 14. Focal length and camera height still placeholders
- **Where:** [config.py:13](../config.py#L13), [config.py:18](../config.py#L18)
- **What happens:** `FOCAL_LENGTH_PX = 600` and `CAMERA_HEIGHT_M = 1.3` are guesses. A MacBook webcam at 640 px wide is probably nearer to 450–500 px, so every distance comes out about 20–30% too far. No distance-error numbers exist yet.
- **Fix:** run `calibrate.py` on each camera, measure the camera height, then follow the distance benchmark in [EVALUATION.md](EVALUATION.md).

### 21. Floor-based distances assume a level camera at chest height and objects on the floor
- **Where:** [distance.py](../distance.py) (`distance_from_edge`, `unknown_height_distance`)
- **What happens:** the ground-plane formula assumes (a) the camera is level and at `CAMERA_HEIGHT_M`, and (b) the object stands on the same floor as the user. When testing on a **laptop on a desk**, both are false: the screen is tilted back, and most objects are on the table. Table-top objects then come out too far away. Walking with a chest-mounted camera, the assumptions hold.
- **Fix:** test with a chest-height phone video instead of a desk. Add a camera-tilt setting (or read it from the phone's motion sensors), and only use floor-based distances when the camera setup matches.

### 22. Screen and speech show different distances
- **Where:** [main.py](../main.py) (`draw`), [guide.py](../guide.py) (`Announcer`)
- **What happens:** the screen updates every frame, about 20 times a second. Speech says the distance at the moment of announcement, takes 2–3 s to speak, and then repeats only every 10 s. The screen also shows every object, while speech covers only the nearest 2.
- **Fix:** show the last spoken sentence as a subtitle on screen, round spoken distances (for example to 0.5 m or steps), and re-announce when the distance changes a lot.

### 23. Small detector mislabels objects
- **Where:** [config.py:8](../config.py#L8)
- **What happens:** `yolov8n` is the smallest YOLO model (about 37 mAP on COCO). A purse was shown as "book" and then as "laptop". A wrong label also means the wrong known height, so the distance is wrong too. Non-obstacle labels are no longer spoken (#17), but a mislabelled *obstacle* still would be.
- **Fix:** compare `yolov8n`, `yolo11n` and `yolo11s` on accuracy and FPS on our laptops. Long term, fine-tune on our own data (#7).

### 8. Currency is classified on the annotated frame
- **Where:** [main.py:152-157](../main.py#L152-L157)
- **What happens:** `draw()` paints boxes, zone lines and FPS text **onto `frame` in place**, and pressing C then classifies that same frame. The model was trained on clean photos.
- **Fix:** keep a clean copy (`raw = frame.copy()`) before drawing and classify `raw`.

### 9. Near-duplicate photos leak between train and val
- **Where:** [collect_currency.py:36](../collect_currency.py#L36), [split_dataset.py:22](../split_dataset.py#L22)
- **What happens:** burst mode saves every 5th frame, so neighbouring photos are almost identical. A random split puts near-twins in both train and val, and validation accuracy looks far better than real-world accuracy.
- **Fix:** split by **recording session** (or by physical note), not by individual image. Save a session ID in the filename.

### 10. No test set: the best epoch is chosen and reported on val
- **Where:** [train_currency.py:92-103](../train_currency.py#L92-L103)
- **What happens:** the checkpoint is chosen by best validation accuracy, and the final report is computed on the same validation set, so the number is optimistically biased.
- **Fix:** use a three-way **train / val / test** split. Keep the test set untouched and report on it once, ideally collected on a different day and camera.

### 11. The model may learn background and lighting instead of the note
- **What happens:** if all the ₹500 photos are taken in one session at one table under one light, the model can learn "this table = ₹500". It scores highly on val but fails at a shop counter.
- **Fix:** for **every** class, collect in several sessions, places, lights and hands. Use the same backgrounds across classes. Check with Grad-CAM that the model looks at the note.

### 12. Whole-frame, single-frame classification
- **Where:** [currency.py:61](../currency.py#L61)
- **What happens:** the full frame is resized to 224 px, so a note that is small in the frame loses its details. The decision comes from one frame, so a single blurry frame gives a wrong answer.
- **Fix:** first **locate** the note, then classify the crop. Vote over 5–10 consecutive frames before speaking.

### 13. Old / new note designs and coins not planned
- **What happens:** ₹10, ₹20, ₹50 and ₹100 have both old and new-series designs in circulation, but the dataset plan doesn't separate them. ₹2000 notes are no longer printed, but they are still valid and can still turn up. Coins are not handled.
- **Fix:** collect both designs for each value, while still predicting the same value label. Decide explicitly on ₹2000 and coins.

### 15. Depth model is indoor-only and slow
- **Where:** [config.py:59](../config.py#L59)
- **What happens:** the *Indoor* metric model is used for street navigation, and it runs on every frame, which lowers FPS a lot.
- **Fix:** switch models by scene (Indoor / Outdoor), run depth every Nth frame or in a separate thread, and consider mixing it with the pinhole estimate.

---

## 🟡 Usability and engineering

### 24. Speech interrupt untested on Windows / Linux
- **Where:** [speech.py:71](../speech.py#L71)
- On macOS an urgent message stops the `say` process at once, and this is tested. On Windows and Linux, interrupting uses pyttsx3's `engine.stop()` from another thread, which doesn't work reliably with every driver. Priority still works there; only the cut-off may not.
- **Fix:** test on Windows. If it fails, use a subprocess voice there too (PowerShell SAPI or `espeak`).

### 16. Keyboard trigger (C key) is not usable by blind users
A blind user holding a note can't find a key on a laptop. **Fix:** a voice command ("Drishti, kitna hai?"), a gesture, a physical button on the wearable, or automatic currency mode when a note fills the frame.

### 17. Spoken distance is too precise *(partly fixed)*
Non-obstacles and ghost detections are no longer spoken (see Fixed). Still open: "0.8 meters" sounds more precise than it is. **Fix:** round to steps ("one step", "two steps") and consider clock directions ("at 2 o'clock").

### 18. English only
Most target users in India will be more comfortable in Hindi or a regional language. **Fix:** multilingual text-to-speech, with the language set in `config.py`.

### 19. Needs a desktop window, no headless or wearable mode
`cv2.imshow` is required and everything runs on a laptop. **Fix:** a `--headless` flag, then a phone app or a Raspberry Pi / Jetson wearable with a chest camera and bone-conduction earphones.

### 20. No tests, unpinned dependencies, old YOLO version
- No unit tests for `distance`, `guide`, `tracking` or `speech`. The fixes so far were checked with throwaway scripts, which should become pytest tests.
- `requirements.txt` sets only minimum versions, so builds are not reproducible.
- YOLOv8 is superseded by newer Ultralytics models (YOLO11).
- Ultralytics is AGPL-3.0 (see the license note in the README).
