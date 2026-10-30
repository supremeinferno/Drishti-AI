# Drishti AI (दृष्टि): Vision Assistant for the Visually Impaired

*Drishti* means "vision" in Hindi. Drishti AI is a real-time assistant that uses a camera to tell a blind or low-vision person **what is around them, how far away it is, and in which direction**, and **reads Indian currency notes out loud**.

> "Careful! chair, 0.8 meters ahead. person, 2.4 meters on your left."

**Status:** working prototype on a laptop webcam. Obstacle detection, distance estimation and voice guidance run end to end. The currency classifier code is complete, but the model has not been trained yet because the dataset still needs to be collected. See [Current status](#current-status).

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
2. Estimates how far each obstacle is and whether it is to the left, ahead or to the right.
3. Speaks **only what matters**: the nearest and most dangerous things, without flooding the user with speech.
4. Recognises Indian currency notes on demand.
5. Eventually runs on a **phone or small wearable** and speaks **Hindi and regional languages**.

## 3. How it works

```
Camera frame
   │
   ├─► YOLOv8 object detection ──► bounding boxes + labels
   │        │
   │        ├─► Distance estimation
   │        │     • Pinhole model: d = H_real × f / h_pixels   (fast, explainable)
   │        │     • Depth Anything V2 metric depth (--depth)   (neural, works for any object)
   │        │
   │        ├─► Direction: left / ahead / right (frame split into thirds)
   │        │
   │        └─► Announcer: nearest N obstacles, cooldown, "Careful!" under 1 m
   │                 │
   │                 └─► Text-to-speech (background thread, never blocks the video)
   │
   └─► [C key] Currency classifier (MobileNetV3 / EfficientNet-B0 / ResNet50,
                transfer learning on our own dataset) ──► "500 rupees note"
```

Each stage is a separate module. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how every part works, including the formulas and design choices.

## 4. Current status

| Feature | Status |
|---|---|
| Object detection (YOLOv8n, 80 COCO classes) | ✅ Working |
| Distance: pinhole formula | ✅ Working (has known bugs, see below) |
| Distance: Depth Anything V2 | ✅ Working, slow on CPU |
| Left / ahead / right direction | ✅ Working |
| Voice guidance with cooldown | ✅ Working (English only) |
| Focal-length calibration tool | ✅ Working (has a resolution bug) |
| Currency dataset collection tool | ✅ Working |
| Currency classifier training (3 architectures) | ✅ Code ready, ⏳ model not trained yet |
| Indian-specific hazards (stairs, potholes, drains, cattle…) | ❌ Not yet |
| Hindi / regional language voice | ❌ Not yet |
| Phone / wearable deployment | ❌ Not yet |
| Measured accuracy and user testing | ❌ Not yet |

## 5. Known mistakes (summary)

We reviewed the whole codebase honestly. The most important problems:

1. 🔴 **Nearby people can be reported as far away.** When only the bottom of the box is cut off by the frame, the pinhole formula overestimates distance.
2. 🔴 **The calibration resolution doesn't match the live app.** Calibration measures on the full-size camera frame, but the app resizes to 640 px, so distances can be off by about 2×.
3. 🔴 **Objects without a known height are dropped silently**, for example cows, horses and fire hydrants.
4. 🔴 **A "Careful!" warning can be replaced by a less important message**, because the speech queue has no priority.
5. 🔴 **The app stops silently** if the camera disconnects.
6. 🟠 **The currency classifier receives the annotated frame** (with boxes and text drawn on it), not the clean camera image.
7. 🟠 **Currency validation accuracy will be inflated** because near-duplicate burst photos land in both the train and validation sets.

Full list with causes, file locations and planned fixes: **[docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)**.

## 6. What we will add

In short: fix the safety bugs → detect Indian street hazards → make currency recognition robust and hands-free → run offline on a phone or wearable in Hindi → measure everything and test with real users.

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

1. **Run the obstacle detector:** `python main.py` (the YOLO weights download on first run).
   - `python main.py --depth`: use the neural depth model instead of the pinhole formula.
   - `python main.py --source walk.mp4 --mute`: run on a video file and print guidance instead of speaking it.
   - `python main.py --source http://<phone-ip>:8080/video`: use a phone camera through the IP Webcam app.
   - Keys: **C** = read the currency note, **Q** = quit.
2. **Calibrate distance:** stand at a measured 2 m from the camera, run `python calibrate.py --distance 2.0 --height 1.70`, press SPACE 5–10 times, then Q. Paste the printed `FOCAL_LENGTH_PX` into `config.py`.
3. **Collect a currency dataset**, about 150–300 photos per class:
   ```bash
   python collect_currency.py --label 10
   python collect_currency.py --label 20
   ...                         # 50, 100, 200, 500
   python collect_currency.py --label background   # no note in view
   ```
   Vary the angle, distance, lighting and background, and include folded or partly covered notes. See the data rules in [docs/EVALUATION.md](docs/EVALUATION.md#currency-dataset-rules) so the accuracy numbers are honest.
4. **Split the data:** `python split_dataset.py`
5. **Train the classifier:**
   ```bash
   python train_currency.py --arch mobilenet_v3_large
   python train_currency.py --arch efficientnet_b0 --out models/currency_effb0.pt
   python train_currency.py --arch resnet50 --batch 8 --out models/currency_r50.pt
   ```
   Mixed precision turns on automatically on CUDA. If you run out of memory, lower `--batch`. The script prints a classification report and a confusion matrix.
6. **Full demo:** `python main.py`, then hold a note in front of the camera and press **C**.

## 9. Project structure

```
Drishti-AI/
├── main.py              # live app: camera loop, drawing, key handling
├── config.py            # all tunable settings (thresholds, heights, paths)
├── detector.py          # YOLOv8 wrapper → list of Detection
├── distance.py          # PinholeEstimator and DepthEstimator
├── guide.py             # direction, message text, Announcer (what to say, when)
├── speech.py            # background text-to-speech
├── device.py            # picks cuda / mps / cpu
├── currency.py          # currency model definition, transforms, live classifier
├── calibrate.py         # measures the webcam focal length in pixels
├── collect_currency.py  # webcam tool to photograph notes
├── split_dataset.py     # data/raw → data/currency/{train,val}
├── train_currency.py    # trains and evaluates the currency classifier
└── docs/
    ├── ARCHITECTURE.md  # how each module works
    ├── KNOWN_ISSUES.md  # every known mistake, with fixes
    ├── ROADMAP.md       # what we will build next, in phases
    └── EVALUATION.md    # how we measure and prove results
```

Datasets (`data/`), trained weights (`models/`, `*.pt`) and training runs (`runs/`) are kept out of git.

## 10. Tech stack

Python · PyTorch · torchvision · Ultralytics YOLOv8 · OpenCV · Hugging Face Transformers (Depth Anything V2) · scikit-learn · macOS `say` / pyttsx3

> **License note:** Ultralytics YOLO is AGPL-3.0. That is fine for academic work, but a commercial or closed-source release would need an Ultralytics enterprise license or a differently licensed detector.
