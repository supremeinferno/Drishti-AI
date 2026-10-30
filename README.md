# Drishti AI (दृष्टि): Vision Assistant for the Visually Impaired

*Drishti* means "vision" in Hindi. Drishti AI is a real-time assistant that uses a camera to tell a blind or low-vision person **what is around them, how far away it is, and in which direction**, and **reads Indian currency notes out loud**.

> 🔔 "Careful! chair, 0.8 meters ahead." · "person still on your left, 2.4 meters" · "Careful! car approaching on your right, 3.0 meters"

**Status:** working prototype on a laptop webcam. Detection, tracking, distance estimation and prioritised voice guidance run end to end, and the first round of safety fixes is done. The currency classifier code is complete, but the model has not been trained yet because the dataset still needs to be collected. See [Current status](#4-current-status).

---

## 1. The problem

India has one of the largest populations of blind and low-vision people in the world. Moving around independently is hard and often dangerous for them:

- **A white cane only senses what it touches.** It reaches about a metre ahead at ground level. It misses obstacles at head and chest height, such as a truck's side mirror, a low branch or a hanging signboard, and it gives no warning of what is further away.
- **Indian streets are unpredictable.** Open drains, potholes, broken footpaths, parked two-wheelers, stray cattle, auto-rickshaws and unexpected stairs are everyday hazards.
- **Handling cash depends on trust.** Notes of different values look and feel similar, especially worn, folded or new-series notes, so blind users often have to rely on the shopkeeper.
- **Existing tools have gaps.** Guide dogs are rare and expensive in India. Smart canes are costly. General-purpose apps such as Seeing AI and Lookout are not designed for Indian street hazards, and many need the internet.

## 2. Our goal

Build a **low-cost, offline, voice-first assistant** that:

1. Detects obstacles in the user's path in real time.
2. Estimates how far each obstacle is, whether it is to the left, ahead or to the right, and whether it is coming towards the user.
3. Speaks **only what matters**, with danger first, without flooding the user with speech.
4. Recognises Indian currency notes on demand.
5. Eventually runs on a **phone or small wearable** and speaks **Hindi and regional languages**.

## 3. How it works

```
Camera frame (resized to 640 px wide)
   │
   ├─► YOLOv8 + ByteTrack ──► boxes, labels, confidence, and a track ID per object
   │        │
   │        ├─► Distance estimation
   │        │     • Pinhole: d = H_real × f / h_pixels, corrected for boxes cut off by
   │        │       the frame, plus a shoulder-width estimate for people
   │        │     • No known height: measured from where the object touches the floor
   │        │     • --depth: Depth Anything V2 metric depth (neural, any object)
   │        │
   │        ├─► Tracking: smoothed distance + approach speed per track ID
   │        ├─► Direction: left / ahead / right (frame split into thirds)
   │        │
   │        ├─► Announce filter: only real obstacles, ≥ 55% confidence,
   │        │                    tracked ≥ 0.5 s (0.2 s when closer than 1 m)
   │        │
   │        └─► Announcer: approaching objects first, then nearest 2;
   │                       repeats every 10 s, sooner if it gets close or comes at the user
   │                 │
   │                 └─► Speaker: urgent ("Careful!") beeps, interrupts normal speech,
   │                              and can never be overwritten by a normal message
   │
   ├─► Camera lost? ──► "Camera lost. Guidance stopped." → retry for 10 s
   │
   └─► [C key] Currency classifier (MobileNetV3 / EfficientNet-B0 / ResNet50,
                transfer learning on our own dataset) ──► "500 rupees note"
```

Each stage is a separate module. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how every part works, including the formulas and design choices.

**On screen:** 🔴 red box = danger (closer than 1 m) · 🟢 green = will be spoken · ⚪ grey = detected but not spoken (not an obstacle, low confidence, or not confirmed yet). Each label shows `#track-id label confidence% distance`.

## 4. Current status

| Feature | Status |
|---|---|
| Object detection (YOLOv8n, 80 COCO classes) | ✅ Working (small model, sometimes mislabels, e.g. a purse as "book") |
| Object tracking (ByteTrack) + distance smoothing | ✅ Working |
| Approach detection ("car approaching") | ✅ Working |
| Distance: pinhole + cut-off / seated-person corrections | ✅ Working, ⚠️ needs calibration per camera |
| Distance for objects with no known height | ✅ Working (assumes they stand on the floor) |
| Distance: Depth Anything V2 | ✅ Working, slow on CPU |
| Speak only real, confirmed obstacles | ✅ Working |
| Priority speech (urgent beep + interrupt) | ✅ Working on macOS, ⚠️ interrupt untested on Windows / Linux |
| Camera-lost / app-stopped alerts | ✅ Working |
| Focal-length calibration tool | ✅ Working |
| Currency dataset collection tool | ✅ Working |
| Currency classifier training (3 architectures) | ✅ Code ready, ⏳ model not trained yet |
| Screen and speech showing the same distance | ❌ Not yet (next step) |
| Indian-specific hazards (stairs, potholes, drains…) | ❌ Not yet |
| Hindi / regional language voice | ❌ Not yet |
| Phone / wearable deployment | ❌ Not yet |
| Measured accuracy and user testing | ❌ Not yet |

## 5. Known issues (summary)

**Fixed so far:** cut-off boxes reported far away · calibration resolution mismatch · objects without a known height dropped silently · danger warnings overwritten · silent stop on camera failure · no tracking · announcing phones, cups and one-frame ghost detections · seated person reported 3.4 m away instead of about 0.6 m.

**Most important open problems:**
1. 🔴 **COCO has no Indian street hazards** (stairs, potholes, drains, poles).
2. 🟠 **Distances need calibration.** `FOCAL_LENGTH_PX` and `CAMERA_HEIGHT_M` are still placeholders.
3. 🟠 **Floor-based distances assume a level camera at chest height.** On a laptop on a desk, objects on the table come out too far away.
4. 🟠 **The screen and the speech show different distances.**
5. 🟠 **The small model mislabels objects**, for example a purse as "book" or "laptop".
6. 🟠 **The currency classifier receives the annotated frame**, and its validation accuracy will be inflated.

Full list with causes, file locations and fixes: **[docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)**.

## 6. What we will add

Next: match the screen and the speech → fix distance for real camera setups → compare bigger detectors → detect Indian street hazards → make currency recognition robust and hands-free → run offline on a phone or wearable in Hindi → measure everything and test with real users.

Phased plan: **[docs/ROADMAP.md](docs/ROADMAP.md)**.
How we will prove it works: **[docs/EVALUATION.md](docs/EVALUATION.md)**.

---

## 7. Setup

Use Python 3.11 or 3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
# NVIDIA laptop: install the CUDA build of torch first, from https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

If `yolov8n.pt` fails to download (some college or office networks block GitHub release files), get it from the Hugging Face mirror instead:

```bash
curl -L -o yolov8n.pt https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.pt
```

## 8. Usage

1. **Calibrate first.** Distances are wrong until you do.
   - Stand at a measured 2 m from the camera with your full body visible, run `python calibrate.py --distance 2.0 --height <your height in m>`, press SPACE 5–10 times, then Q. Paste the printed `FOCAL_LENGTH_PX` into `config.py`.
   - Measure how high the camera is above the floor and set `CAMERA_HEIGHT_M` in `config.py`. Keep the camera roughly level.
2. **Run the assistant:** `python main.py` (the YOLO weights download on first run).
   - `python main.py --depth`: use the neural depth model instead of the pinhole formula.
   - `python main.py --source walk.mp4 --mute`: run on a video file and print guidance instead of speaking it. Recording one walk at chest height and replaying it after every change is the easiest way to compare versions.
   - `python main.py --source http://<phone-ip>:8080/video`: use a phone camera through the IP Webcam app.
   - Keys: **C** = read the currency note, **Q** = quit.
3. **Collect a currency dataset**, about 150–300 photos per class:
   ```bash
   python collect_currency.py --label 10
   python collect_currency.py --label 20
   ...                         # 50, 100, 200, 500
   python collect_currency.py --label background   # no note in view
   ```
   Vary the angle, distance, lighting and background, and include folded or partly covered notes. Follow the [currency dataset rules](docs/EVALUATION.md#currency-dataset-rules) so the accuracy numbers are honest.
4. **Split the data:** `python split_dataset.py`
5. **Train the classifier:**
   ```bash
   python train_currency.py --arch mobilenet_v3_large
   python train_currency.py --arch efficientnet_b0 --out models/currency_effb0.pt
   python train_currency.py --arch resnet50 --batch 8 --out models/currency_r50.pt
   ```
   Mixed precision turns on automatically on CUDA. If you run out of memory, lower `--batch`. The script prints a classification report and a confusion matrix.
6. **Full demo:** `python main.py`, then hold a note in front of the camera and press **C**.

All thresholds (danger distance, cooldown, confidence, which classes are spoken…) are in [config.py](config.py).

## 9. Project structure

```
Drishti-AI/
├── main.py              # live app: camera loop, camera-lost handling, drawing, keys
├── config.py            # all tunable settings (thresholds, heights, spoken classes, paths)
├── detector.py          # YOLOv8 + ByteTrack wrapper → list of Detection with track IDs
├── tracking.py          # per-object distance smoothing, approach speed, time tracked
├── distance.py          # PinholeEstimator and DepthEstimator
├── guide.py             # direction, announce filter, Announcer (what to say, when)
├── speech.py            # background text-to-speech with urgent / normal priority
├── device.py            # picks cuda / mps / cpu
├── currency.py          # currency model definition, transforms, live classifier
├── calibrate.py         # measures the webcam focal length in pixels
├── collect_currency.py  # webcam tool to photograph notes
├── split_dataset.py     # data/raw → data/currency/{train,val}
├── train_currency.py    # trains and evaluates the currency classifier
└── docs/
    ├── ARCHITECTURE.md  # how each module works
    ├── KNOWN_ISSUES.md  # every known problem, fixed and open
    ├── ROADMAP.md       # what we will build next, in phases
    └── EVALUATION.md    # how we measure and prove results
```

Datasets (`data/`), trained weights (`models/`, `*.pt`) and training runs (`runs/`) are kept out of git.

## 10. Tech stack

Python · PyTorch · torchvision · Ultralytics YOLOv8 · ByteTrack · OpenCV · Hugging Face Transformers (Depth Anything V2) · scikit-learn · macOS `say` / pyttsx3

> **License note:** Ultralytics YOLO is AGPL-3.0. That is fine for academic work, but a commercial or closed-source release would need an Ultralytics enterprise license or a differently licensed detector.
