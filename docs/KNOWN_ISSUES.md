# Known issues and mistakes

An honest review of the current code. Each issue lists **what goes wrong**, **why**, **where** and **how we plan to fix it**.

Severity:
- 🔴 **Safety:** can give the user wrong or missing danger information.
- 🟠 **Correctness:** wrong results or misleading numbers.
- 🟡 **Usability / engineering:** works, but not good enough for real users or a real release.

| # | Issue | Severity | Area |
|---|---|---|---|
| 1 | Nearby objects cut off at the bottom are reported as far away | 🔴 | Distance |
| 2 | Calibration and the live app use different resolutions | 🔴 | Distance |
| 3 | Objects without a known height are dropped silently | 🔴 | Distance |
| 4 | Danger warnings can be overwritten or delayed | 🔴 | Speech |
| 5 | App stops silently when the camera fails | 🔴 | Main loop |
| 6 | No frame-to-frame tracking or smoothing | 🔴 | Detection |
| 7 | COCO has no Indian street hazards | 🔴 | Detection |
| 8 | Currency is classified on the annotated frame | 🟠 | Currency |
| 9 | Near-duplicate photos leak between train and val | 🟠 | Currency |
| 10 | No test set: the best epoch is chosen and reported on val | 🟠 | Currency |
| 11 | The model may learn background and lighting instead of the note | 🟠 | Currency |
| 12 | Whole-frame, single-frame classification | 🟠 | Currency |
| 13 | Old / new note designs and coins not planned | 🟠 | Currency |
| 14 | Calibration and distance accuracy never measured | 🟠 | Evaluation |
| 15 | Depth model is indoor-only and slow | 🟠 | Distance |
| 16 | Keyboard trigger (C key) is not usable by blind users | 🟡 | UX |
| 17 | Says too much and too precisely | 🟡 | UX |
| 18 | English only | 🟡 | UX |
| 19 | Needs a desktop window, no headless or wearable mode | 🟡 | Deployment |
| 20 | No tests, unpinned dependencies, old YOLO version | 🟡 | Engineering |

---

## 🔴 Safety

### 1. Nearby objects cut off at the bottom are reported as far away
- **Where:** [distance.py:31](../distance.py#L31)
- **What happens:** when a person stands close, their legs go out of the bottom of the frame. The box gets *shorter*, and `distance = H × f / h` becomes *larger*. A person 0.7 m away can be reported as 1.5 m or more, which is no "Careful!" warning.
- **Why:** the "object fills the view" check only fires when the box touches **both** the top **and** bottom edges.
- **Fix:** treat any box touching the bottom edge as partly visible. Use the box width or a depth-model value instead, or clamp the result to the danger zone. Optionally estimate distance from where the object meets the ground (with a known camera height).

### 2. Calibration and the live app use different resolutions
- **Where:** [calibrate.py:31](../calibrate.py#L31) vs [main.py:87-88](../main.py#L87-L88)
- **What happens:** `calibrate.py` measures boxes on the full camera frame (often 1280 or 1920 px wide). `main.py` first resizes to 640 px. A focal length calibrated at 1280 px is **2× too large** at 640 px, so every distance comes out about **2× too far**.
- **Fix:** resize to `FRAME_WIDTH` in `calibrate.py` exactly as `main.py` does, or store the focal length together with the resolution and rescale it.

### 3. Objects without a known height are dropped silently
- **Where:** [distance.py:22-27](../distance.py#L22-L27), [main.py:94](../main.py#L94)
- **What happens:** in pinhole mode, YOLO detects a **cow**, **horse** or **fire hydrant**, but because it is not in `KNOWN_HEIGHTS_M` the distance is `None` and the obstacle is thrown away. The user hears nothing.
- **Fix:** add heights for every relevant COCO class. For unknown classes, fall back to the depth model or a "close / far" estimate from box size, and never drop a large object in the "ahead" zone.

### 4. Danger warnings can be overwritten or delayed
- **Where:** [speech.py:17-32](../speech.py#L17-L32)
- **What happens:** the queue holds one message and new messages replace old ones. A pending *"Careful! chair ahead"* can be replaced by *"person, 3 meters on your left"* before it is spoken. A warning also has to wait until the current sentence finishes, which can take about 2 seconds.
- **Fix:** add message priority (danger > normal). Danger messages interrupt current speech; normal messages never replace danger messages. Add a short warning beep or vibration that plays instantly.

### 5. App stops silently when the camera fails
- **Where:** [main.py:76-86](../main.py#L76-L86)
- **What happens:** if the camera can't open, or disconnects, `cap.read()` fails and the loop simply `break`s. A blind user hears nothing and may assume the path is clear.
- **Fix:** check `cap.isOpened()`, retry, and **say** *"Camera lost, guidance stopped"*. Optionally add a quiet periodic "heartbeat" tone so the user knows the system is alive.

### 6. No frame-to-frame tracking or smoothing
- **Where:** [detector.py](../detector.py), [guide.py](../guide.py)
- **What happens:** every frame is processed on its own. A single wrong detection gets spoken. Distances jump between frames. Two people on the left share one cooldown key `("person", "on your left")`. "Approaching" is judged from only two noisy readings.
- **Fix:** use YOLO tracking (`model.track`, ByteTrack) so each object has an ID. Only speak an object that has been seen in several frames. Smooth distance per ID, and estimate approach speed (time until collision) from the trend.

### 7. COCO has no Indian street hazards
- **Where:** [config.py:7](../config.py#L7) (`yolov8n.pt`)
- **What happens:** the most dangerous things for a blind pedestrian in India, such as **stairs, potholes, open drains, poles, walls, glass doors, speed breakers, auto-rickshaws and e-rickshaws**, are not COCO classes, so they cannot be detected at all.
- **Fix:** collect and label an Indian street dataset (Roboflow / CVAT) and fine-tune YOLO. Add a **free-path / walkable-ground** check using the depth map or segmentation, so even *unknown* obstacles in front of the user trigger a warning.

---

## 🟠 Correctness

### 8. Currency is classified on the annotated frame
- **Where:** [main.py:101-106](../main.py#L101-L106)
- **What happens:** `draw()` paints boxes, zone lines and FPS text **onto `frame` in place**, and pressing C then classifies that same frame. The model was trained on clean photos, so the drawings are noise it has never seen.
- **Fix:** keep a clean copy (`raw = frame.copy()`) before drawing and classify `raw`.

### 9. Near-duplicate photos leak between train and val
- **Where:** [collect_currency.py:36](../collect_currency.py#L36), [split_dataset.py:22-24](../split_dataset.py#L22-L24)
- **What happens:** burst mode saves every 5th frame, so neighbouring photos are almost identical. A random split puts near-twins in both train and val, and validation accuracy looks far better than real-world accuracy.
- **Fix:** split by **recording session** (or by physical note), not by individual image. Save a session ID in the filename.

### 10. No test set: the best epoch is chosen and reported on val
- **Where:** [train_currency.py:92-103](../train_currency.py#L92-L103)
- **What happens:** the checkpoint is chosen by best validation accuracy, and the final report is computed on the same validation set. The reported number is optimistically biased.
- **Fix:** use a three-way **train / val / test** split. Keep the test set untouched and report on it once. Better still, use a test set collected on a different day, with a different camera and different notes.

### 11. The model may learn background and lighting instead of the note
- **What happens:** if all the ₹500 photos are taken in one session at one table under one light, the model can learn "this table = ₹500". It scores highly on val but fails at a shop counter.
- **Fix:** for **every** class, collect in several sessions, places, lights and hands. Use the same backgrounds across classes. Check with Grad-CAM that the model looks at the note.

### 12. Whole-frame, single-frame classification
- **Where:** [currency.py:61-66](../currency.py#L61-L66)
- **What happens:** the full frame is resized to 224 px, so a note that is small in the frame loses its details. The decision comes from one frame, so a single blurry frame gives a wrong answer.
- **Fix:** first **locate** the note (a small detector, or a contour / largest-rectangle crop), then classify the crop. Vote over 5–10 consecutive frames before speaking.

### 13. Old / new note designs and coins not planned
- **What happens:** ₹10, ₹20, ₹50 and ₹100 have both old and new-series designs in circulation, but the dataset plan doesn't separate them. ₹2000 notes are no longer printed, but they are still valid and can still turn up. Coins are not handled.
- **Fix:** collect both designs for each value, while still predicting the same value label. Decide explicitly on ₹2000 and coins.

### 14. Calibration and distance accuracy never measured
- **What happens:** `FOCAL_LENGTH_PX = 600.0` is a placeholder, and no distance-error numbers exist yet.
- **Fix:** follow the distance benchmark in [EVALUATION.md](EVALUATION.md).

### 15. Depth model is indoor-only and slow
- **Where:** [config.py:44](../config.py#L44), [distance.py:47-53](../distance.py#L47-L53)
- **What happens:** the *Indoor* metric model is used for street navigation, and it runs on every frame, which lowers FPS a lot.
- **Fix:** switch models by scene (Indoor / Outdoor), run depth every Nth frame or in a separate thread, and consider mixing it with the pinhole estimate.

---

## 🟡 Usability and engineering

### 16. Keyboard trigger (C key) is not usable by blind users
A blind user holding a note can't find a key on a laptop. **Fix:** a voice command ("Drishti, kitna hai?"), a gesture, a physical button on the wearable, or automatic currency mode when a note fills the frame.

### 17. Says too much and too precisely
It announces cups, books, laptops and even the phone in the user's own hand. "0.8 meters" sounds more precise than it is. **Fix:** whitelist only the obstacles that matter, ignore objects held by the user, round distances to steps ("one step", "two steps"), and consider clock directions ("at 2 o'clock").

### 18. English only
Most target users in India will be more comfortable in Hindi or a regional language. **Fix:** multilingual text-to-speech (for example Indic voices), with the language set in `config.py`.

### 19. Needs a desktop window, no headless or wearable mode
`cv2.imshow` is required and everything runs on a laptop. **Fix:** a `--headless` flag, then a phone app or a Raspberry Pi / Jetson wearable with a chest or head camera and bone-conduction earphones.

### 20. No tests, unpinned dependencies, old YOLO version
- No unit tests for `distance`, `guide` or `speech`.
- `requirements.txt` sets only minimum versions, so builds are not reproducible.
- YOLOv8 is superseded by newer Ultralytics models (YOLO11).
- Ultralytics is AGPL-3.0 (see the license note in the README).
- **Fix:** pytest for the pure-logic modules, a lock file, and a benchmark of newer detectors.
